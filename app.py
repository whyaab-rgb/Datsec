import os
import math
import requests
import pandas as pd
import streamlit as st

from datetime import date, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_autorefresh import st_autorefresh


# =========================
# CONFIG
# =========================

load_dotenv()

API_KEY = os.getenv("DATASECTOR_API_KEY")
BASE_URL = os.getenv("DATASECTOR_BASE_URL", "https://api.datasectors.com/api")

st.set_page_config(
    page_title="IDX High Prob Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# UI STYLE
# =========================

st.markdown("""
<style>
    .main {
        background-color: #101418;
    }

    div[data-testid="stDataFrame"] {
        background-color: #101418;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    .title-box {
        background: linear-gradient(90deg, #121a24, #1d2733);
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #2b3c4e;
        margin-bottom: 10px;
    }

    .title-text {
        color: #f7f7f7;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .subtitle-text {
        color: #aab7c4;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# HELPER DATASECTORS
# =========================

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


@st.cache_data(ttl=60)
def load_symbols():
    try:
        df = pd.read_csv("idx_symbols.csv")
        return df["symbol"].dropna().astype(str).str.upper().unique().tolist()
    except Exception:
        return ["BBCA", "BBRI", "BMRI", "TLKM", "ASII"]


def extract_chart_payload(raw):
    """
    Dibuat fleksibel karena struktur DataSectors kadang nested.
    Bisa berupa:
    - raw["chartbit"]
    - raw["data"]
    - raw["data"]["chartbit"]
    - list langsung
    """
    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["chartbit", "data", "result", "candles"]:
            val = raw.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                nested = extract_chart_payload(val)
                if nested:
                    return nested

    return []


def fetch_chart(symbol, timeframe="5m", from_date=None, to_date=None, limit=120):
    if not API_KEY:
        raise RuntimeError("DATASECTOR_API_KEY belum diisi di file .env")

    url = f"{BASE_URL}/chart-saham/{symbol}/{timeframe}"

    params = {}

    # Untuk daily DataSectors biasanya butuh from/to.
    if timeframe == "daily":
        if from_date is None:
            from_date = date.today() - timedelta(days=180)
        if to_date is None:
            to_date = date.today()

        params["from"] = str(from_date)
        params["to"] = str(to_date)

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    r = requests.get(url, headers=headers, params=params, timeout=15)

    if r.status_code != 200:
        return {
            "symbol": symbol,
            "ok": False,
            "error": f"{r.status_code} {r.text[:120]}",
            "candles": []
        }

    raw = r.json()
    candles = extract_chart_payload(raw)

    if not candles:
        return {
            "symbol": symbol,
            "ok": False,
            "error": "empty candles",
            "candles": []
        }

    return {
        "symbol": symbol,
        "ok": True,
        "error": None,
        "candles": candles[-limit:]
    }


def normalize_candles(candles):
    rows = []

    for c in candles:
        if not isinstance(c, dict):
            continue

        # Fleksibel untuk kemungkinan nama field berbeda
        open_ = c.get("open", c.get("o"))
        high = c.get("high", c.get("h"))
        low = c.get("low", c.get("l"))
        close = c.get("close", c.get("c"))
        volume = c.get("volume", c.get("v"))
        time_ = c.get("time", c.get("date", c.get("datetime", c.get("timestamp"))))

        rows.append({
            "time": time_,
            "open": safe_float(open_),
            "high": safe_float(high),
            "low": safe_float(low),
            "close": safe_float(close),
            "volume": safe_float(volume)
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df[df["close"] > 0].copy()
    df.reset_index(drop=True, inplace=True)

    return df


# =========================
# INDICATORS
# =========================

def calc_rsi(series, period=14):
    if len(series) < period + 2:
        return 50.0

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, 0.000001)
    rsi = 100 - (100 / (1 + rs))

    return safe_float(rsi.iloc[-1], 50.0)


def calc_support_resistance(df, window=20):
    if len(df) < 5:
        last = safe_float(df["close"].iloc[-1]) if len(df) else 0
        return last, last

    recent = df.tail(window)
    support = safe_float(recent["low"].min())
    resistance = safe_float(recent["high"].max())

    return support, resistance


def analyze_symbol(symbol, timeframe="5m"):
    try:
        intraday = fetch_chart(symbol, timeframe=timeframe, limit=120)

        if not intraday["ok"]:
            return {
                "EMITEN": symbol,
                "FASE": "ERROR",
                "SETUP": intraday["error"],
                "AKSI": "SKIP",
                "GAIN": 0,
                "DAY": "-",
                "ENTRY": "-",
                "TP1 / TP2 / TP3": "-",
                "TRAIL SL": "-",
                "PROFIT": "-",
                "RSI": "-",
                "SINYAL": "DATA ERROR",
                "S1": "-",
                "R1": "-",
                "VAL(M)": "-"
            }

        df = normalize_candles(intraday["candles"])

        if len(df) < 30:
            return {
                "EMITEN": symbol,
                "FASE": "WAIT",
                "SETUP": "DATA KURANG",
                "AKSI": "WAIT",
                "GAIN": 0,
                "DAY": "CANDLE KURANG",
                "ENTRY": "-",
                "TP1 / TP2 / TP3": "-",
                "TRAIL SL": "-",
                "PROFIT": "-",
                "RSI": "-",
                "SINYAL": "BELUM VALID",
                "S1": "-",
                "R1": "-",
                "VAL(M)": "-"
            }

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = safe_float(last["close"])
        prev_close = safe_float(prev["close"])
        high = safe_float(last["high"])
        low = safe_float(last["low"])
        volume = safe_float(last["volume"])

        gain = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        rsi = calc_rsi(df["close"])
        s1, r1 = calc_support_resistance(df, 20)

        vol_ma20 = safe_float(df["volume"].tail(20).mean())
        rvol = volume / vol_ma20 if vol_ma20 > 0 else 0

        ma20 = safe_float(df["close"].rolling(20).mean().iloc[-1])
        ma50 = safe_float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else ma20

        high20 = safe_float(df["high"].tail(20).max())
        low20 = safe_float(df["low"].tail(20).min())

        breakout = close >= high20 * 0.995
        rebound20 = close > ma20 and prev_close <= ma20
        trend_up = close > ma20 > ma50
        strong_volume = rvol >= 1.3
        oversold_rebound = rsi > 45 and rsi < 65 and close > prev_close
        hot = rsi >= 70

        entry = round(close)
        tp1 = round(entry * 1.03)
        tp2 = round(entry * 1.07)
        tp3 = round(entry * 1.12)

        trail_sl = round(max(low20, entry * 0.95))
        profit_tp3 = ((tp3 - entry) / entry * 100) if entry > 0 else 0

        value_m = close * volume / 1_000_000

        # =========================
        # LOGIC SETUP
        # =========================

        if breakout and strong_volume and trend_up:
            fase = "PART"
            setup = "🚀 BREAKOUT"
            aksi = "🏆 PAST TP2 - EXIT" if gain > 10 else "✅ BE - HOLD TRAIL"
            sinyal = "🔥 AMBIL SEMUA - EXIT" if rsi > 82 else "✅ ON TRACK - OKE 👍"

        elif rebound20 and strong_volume:
            fase = "PART"
            setup = "🟢 REBOUND20"
            aksi = "✅ BE - HOLD TRAIL"
            sinyal = "✅ HOLD NYAMAN 💪"

        elif oversold_rebound and close > ma20:
            fase = "PART"
            setup = "🟢 REBOUND50"
            aksi = "✅ BE - HOLD TRAIL"
            sinyal = "✅ HOLD NYAMAN 💪"

        elif hot and gain < 0:
            fase = "TRANS"
            setup = "🟡 WAIT"
            aksi = "⚠️ HOLD"
            sinyal = "⚠️ WASPADAI BALIK"

        elif close < ma20 and gain < -1:
            fase = "AKUM"
            setup = "⏳ WAIT"
            aksi = "☠️ DEAD CROSS!"
            sinyal = "☠️ DEAD CROSS!"

        else:
            fase = "PART" if close > ma20 else "WAIT"
            setup = "🟢 GC ONLY" if trend_up else "⏳ WAIT"
            aksi = "📊 ON TRACK" if trend_up else "HOLD"
            sinyal = "📊 ON TRACK - OKE 👍" if trend_up else "HOLD"

        if gain > 0:
            day = "NAIK, TEKANAN JUAL" if rsi > 70 else "NAIK TIPIS, KOREKSI"
        elif gain < -2:
            day = "TURUN, EKOR BAWAH 👀"
        else:
            day = "STABIL DI ATAS"

        return {
            "EMITEN": symbol,
            "FASE": fase,
            "SETUP": setup,
            "AKSI": aksi,
            "GAIN": round(gain, 2),
            "DAY": day,
            "ENTRY": entry,
            "TP1 / TP2 / TP3": f"{tp1}/{tp2}/{tp3}",
            "TRAIL SL": trail_sl,
            "PROFIT": f"{profit_tp3:.2f}%",
            "RSI": round(rsi, 1),
            "SINYAL": sinyal,
            "S1": round(s1),
            "R1": round(r1),
            "VAL(M)": round(value_m, 2)
        }

    except Exception as e:
        return {
            "EMITEN": symbol,
            "FASE": "ERROR",
            "SETUP": str(e)[:80],
            "AKSI": "SKIP",
            "GAIN": 0,
            "DAY": "-",
            "ENTRY": "-",
            "TP1 / TP2 / TP3": "-",
            "TRAIL SL": "-",
            "PROFIT": "-",
            "RSI": "-",
            "SINYAL": "ERROR",
            "S1": "-",
            "R1": "-",
            "VAL(M)": "-"
        }


# =========================
# COLOR STYLE
# =========================

def style_gain(v):
    try:
        v = float(v)
        if v > 5:
            return "background-color:#7b2cff;color:white;font-weight:bold;"
        if v > 0:
            return "background-color:#009b43;color:white;font-weight:bold;"
        if v < -2:
            return "background-color:#d71920;color:white;font-weight:bold;"
        if v < 0:
            return "background-color:#b00020;color:white;font-weight:bold;"
    except Exception:
        pass

    return "background-color:#1f2933;color:white;"


def style_rsi(v):
    try:
        v = float(v)
        if v >= 80:
            return "background-color:#7b2cff;color:white;font-weight:bold;"
        if v >= 70:
            return "background-color:#d71920;color:white;font-weight:bold;"
        if v >= 55:
            return "background-color:#009b43;color:white;font-weight:bold;"
        if v < 45:
            return "background-color:#34495e;color:white;font-weight:bold;"
    except Exception:
        pass

    return "background-color:#1f2933;color:white;"


def style_phase(v):
    text = str(v)
    if "PART" in text:
        return "background-color:#008a36;color:white;font-weight:bold;"
    if "TRANS" in text:
        return "background-color:#005f99;color:white;font-weight:bold;"
    if "AKUM" in text:
        return "background-color:#ffb000;color:black;font-weight:bold;"
    if "ERROR" in text:
        return "background-color:#d71920;color:white;font-weight:bold;"
    return "background-color:#202936;color:white;font-weight:bold;"


def style_setup(v):
    text = str(v)
    if "BREAKOUT" in text:
        return "background-color:#b58000;color:white;font-weight:bold;"
    if "REBOUND" in text:
        return "background-color:#008a36;color:white;font-weight:bold;"
    if "GC" in text or "GOLDEN" in text:
        return "background-color:#7b2cff;color:white;font-weight:bold;"
    if "DEAD" in text:
        return "background-color:#b00020;color:white;font-weight:bold;"
    return "background-color:#202936;color:white;font-weight:bold;"


def style_signal(v):
    text = str(v)
    if "AMBIL" in text:
        return "background-color:#008a36;color:white;font-weight:bold;"
    if "ON TRACK" in text:
        return "background-color:#009b43;color:white;font-weight:bold;"
    if "HOLD NYAMAN" in text:
        return "background-color:#009b43;color:white;font-weight:bold;"
    if "WASPADAI" in text:
        return "background-color:#d35400;color:white;font-weight:bold;"
    if "DEAD" in text or "ERROR" in text:
        return "background-color:#b00020;color:white;font-weight:bold;"
    return "background-color:#202936;color:white;font-weight:bold;"


def style_dataframe(df):
    return (
        df.style
        .map(style_phase, subset=["FASE"])
        .map(style_setup, subset=["SETUP"])
        .map(style_gain, subset=["GAIN"])
        .map(style_rsi, subset=["RSI"])
        .map(style_signal, subset=["SINYAL"])
        .set_properties(**{
            "background-color": "#111827",
            "color": "white",
            "border-color": "#2b3440",
            "font-size": "13px",
            "font-weight": "600",
            "text-align": "center"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1d2733"),
                    ("color", "white"),
                    ("font-weight", "800"),
                    ("text-align", "center"),
                    ("border", "1px solid #3a4655")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #2b3440")
                ]
            }
        ])
    )


# =========================
# APP
# =========================

st.markdown("""
<div class="title-box">
    <div class="title-text">🔍 IDX HIGH PROB SCREENER — STREAMLIT + DATASECTORS</div>
    <div class="subtitle-text">Realtime watchlist | Rebound | Breakout | Trail SL | RSI | Support Resistance</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Setting Scanner")

    symbols_all = load_symbols()

    selected_symbols = st.multiselect(
        "Pilih Emiten",
        symbols_all,
        default=symbols_all[:30]
    )

    timeframe = st.selectbox(
        "Timeframe DataSectors",
        ["1m", "5m", "10m", "15m", "30m", "1h"],
        index=1
    )

    refresh_sec = st.slider(
        "Auto Refresh Detik",
        min_value=10,
        max_value=300,
        value=30,
        step=5
    )

    max_workers = st.slider(
        "Parallel Request",
        min_value=1,
        max_value=20,
        value=8,
        step=1
    )

    min_value_m = st.number_input(
        "Minimal Value Transaksi, juta",
        min_value=0,
        value=0,
        step=100
    )

    only_signal = st.checkbox("Tampilkan hanya yang ON TRACK / HOLD NYAMAN / AMBIL", value=False)

    st.caption("Pastikan API key DataSectors sudah diisi di file .env")


st_autorefresh(interval=refresh_sec * 1000, key="idx_screener_refresh")

if not API_KEY:
    st.error("DATASECTOR_API_KEY belum ditemukan. Isi dulu file .env.")
    st.stop()

if not selected_symbols:
    st.warning("Pilih minimal 1 emiten.")
    st.stop()


progress = st.progress(0)
status = st.empty()

rows = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(analyze_symbol, symbol, timeframe): symbol
        for symbol in selected_symbols
    }

    total = len(futures)
    done = 0

    for future in as_completed(futures):
        symbol = futures[future]
        done += 1

        try:
            rows.append(future.result())
        except Exception as e:
            rows.append({
                "EMITEN": symbol,
                "FASE": "ERROR",
                "SETUP": str(e),
                "AKSI": "SKIP",
                "GAIN": 0,
                "DAY": "-",
                "ENTRY": "-",
                "TP1 / TP2 / TP3": "-",
                "TRAIL SL": "-",
                "PROFIT": "-",
                "RSI": "-",
                "SINYAL": "ERROR",
                "S1": "-",
                "R1": "-",
                "VAL(M)": "-"
            })

        progress.progress(done / total)
        status.caption(f"Scanning {done}/{total}: {symbol}")

progress.empty()
status.empty()

df = pd.DataFrame(rows)

if df.empty:
    st.warning("Tidak ada data.")
    st.stop()

# Filter value
df["VAL_NUM"] = pd.to_numeric(df["VAL(M)"], errors="coerce").fillna(0)

if min_value_m > 0:
    df = df[df["VAL_NUM"] >= min_value_m]

if only_signal:
    df = df[
        df["SINYAL"].astype(str).str.contains(
            "ON TRACK|HOLD NYAMAN|AMBIL",
            case=False,
            na=False
        )
    ]

# Sort prioritas
df["SORT_GAIN"] = pd.to_numeric(df["GAIN"], errors="coerce").fillna(-999)
df["SORT_RSI"] = pd.to_numeric(df["RSI"], errors="coerce").fillna(0)

df = df.sort_values(
    by=["SORT_GAIN", "SORT_RSI", "VAL_NUM"],
    ascending=[False, False, False]
)

df_show = df.drop(columns=["VAL_NUM", "SORT_GAIN", "SORT_RSI"], errors="ignore")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Emiten", len(df_show))
col2.metric("Timeframe", timeframe)
col3.metric("Refresh", f"{refresh_sec} detik")
col4.metric("Data Source", "DataSectors")

st.dataframe(
    style_dataframe(df_show),
    use_container_width=True,
    height=720
)

st.caption(
    "Catatan: Signal ini adalah screener teknikal otomatis, bukan rekomendasi beli/jual. "
    "Validasi lagi dengan bid-offer, broker summary, running trade, dan market context."
)
