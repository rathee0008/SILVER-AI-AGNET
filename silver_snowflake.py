"""
Silver AI Trading Agent — Streamlit in Snowflake (SiS)  [PRO v2]
=================================================================
Deploy with: setup_snowflake.sql

Differences from silver_web.py:
• yfinance       → requests to Yahoo Finance v8 API
• anthropic SDK  → requests to Anthropic REST API
• TradingView JS → plotly candlestick (SiS blocks custom HTML/JS)
• API key        → Snowflake Secret (_snowflake module); falls back to sidebar input

PRO v2 Enhancements:
• Stochastic RSI (StochRSI K/D)
• Williams %R
• Commodity Channel Index (CCI)
• ADX / DMI (trend strength + direction)
• On-Balance Volume (OBV)
• VWAP (session)
• Ichimoku Cloud (Tenkan/Kijun/Senkou A & B/Chikou)
• Supertrend (ATR-based trailing stop)
• RSI & MACD Divergence Detection
• Signal Scoring Engine (composite bull/bear score 0-100)
• Multi-timeframe confluence table
• Enhanced AI prompt with all new indicators
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# ── Snowflake secret ──────────────────────────────────────────────────────────
try:
    import _snowflake  # noqa: F401 — SiS only
    _IN_SNOWFLAKE = True
except ImportError:
    _IN_SNOWFLAKE = False

def _get_secret() -> str | None:
    if _IN_SNOWFLAKE:
        try:
            return _snowflake.get_generic_secret_string("anthropic_key")
        except Exception:
            pass
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return None

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Silver AI Trading Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; }
h1,h2,h3,h4 { color: #f0f6fc; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; text-align: center;
}
.metric-label { color: #8b949e; font-size: 13px; margin-bottom: 4px; }
.metric-value { color: #f0f6fc; font-size: 26px; font-weight: 700; }
.metric-sub { font-size: 13px; margin-top: 4px; }
.bull { color: #3fb950; } .bear { color: #f85149; } .neut { color: #d29922; }
.tag { display:inline-block; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.tag-bull { background:#1a3626; color:#3fb950; }
.tag-bear { background:#3c1212; color:#f85149; }
.tag-neut { background:#2d2209; color:#d29922; }
.analysis-box {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:20px 24px; color:#c9d1d9; line-height:1.7;
    white-space:pre-wrap; font-family:'Segoe UI',sans-serif;
}
.level-badge {
    background:#21262d; border:1px solid #30363d; border-radius:6px;
    padding:3px 10px; color:#e6edf3; font-size:14px; font-weight:600;
    display:inline-block; margin:3px;
}
.sup-badge { border-color:#3fb950; color:#3fb950; }
.res-badge { border-color:#f85149; color:#f85149; }
.fib-badge { border-color:#58a6ff; color:#58a6ff; }
.score-bar-wrap { background:#21262d; border-radius:4px; height:10px; width:100%; }
.score-bar { height:10px; border-radius:4px; transition: width 0.3s; }
.conf-table { width:100%; border-collapse:collapse; font-size:13px; }
.conf-table th { color:#8b949e; font-weight:600; padding:6px 10px; text-align:left;
    border-bottom:1px solid #30363d; }
.conf-table td { padding:6px 10px; border-bottom:1px solid #21262d; color:#c9d1d9; }
.conf-table tr:last-child td { border-bottom:none; }
</style>
""", unsafe_allow_html=True)

TICKER = "SI=F"
AI_MODEL = "claude-opus-4-8"
YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
YF_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ══════════════════════════════════════════════════════════════════════════════
# DATA — Yahoo Finance via requests
# ══════════════════════════════════════════════════════════════════════════════
def _yf_fetch(ticker: str, interval: str, range_: str) -> pd.DataFrame:
    r = requests.get(
        f"{YF_BASE}/{ticker}",
        params={"interval": interval, "range": range_, "includePrePost": "false"},
        headers=YF_HDR, timeout=20,
    )
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q["open"], "High": q["high"], "Low": q["low"],
         "Close": q["close"], "Volume": q.get("volume", [0] * len(ts))},
        index=pd.to_datetime(ts, unit="s", utc=True),
    )
    return df.dropna(subset=["Close"])

# ══════════════════════════════════════════════════════════════════════════════
# PRO INDICATORS — pure pandas/numpy
# ══════════════════════════════════════════════════════════════════════════════
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def _rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, float("nan")))

def _stoch_rsi(s, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI — K and D lines."""
    rsi = _rsi(s, rsi_len)
    lo = rsi.rolling(stoch_len).min()
    hi = rsi.rolling(stoch_len).max()
    raw = (rsi - lo) / (hi - lo + 1e-10) * 100
    k = raw.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def _williams_r(df, n=14):
    """Williams %R."""
    hi = df["High"].rolling(n).max()
    lo = df["Low"].rolling(n).min()
    return (hi - df["Close"]) / (hi - lo + 1e-10) * -100

def _cci(df, n=20):
    """Commodity Channel Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)

def _adx(df, n=14):
    """ADX + +DI / -DI."""
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff(); dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_n = tr.ewm(com=n - 1, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=n - 1, adjust=False).mean() / atr_n.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(com=n - 1, adjust=False).mean() / atr_n.replace(0, float("nan"))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx_val = dx.ewm(com=n - 1, adjust=False).mean()
    return adx_val, plus_di, minus_di

def _macd(s):
    line = _ema(s, 12) - _ema(s, 26); sig = _ema(line, 9)
    return line, sig, line - sig

def _bollinger(s, n=20, k=2.0):
    mid = s.rolling(n).mean(); std = s.rolling(n).std()
    return mid + k * std, mid, mid - k * std

def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()

def _obv(df):
    """On-Balance Volume."""
    direction = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (df["Volume"] * direction).cumsum()

def _vwap(df):
    """VWAP approximation over available data."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vp = (tp * df["Volume"]).cumsum()
    cum_v = df["Volume"].cumsum()
    return cum_vp / cum_v.replace(0, float("nan"))

def _ichimoku(df, t=9, k=26, s=52):
    """Ichimoku Cloud: Tenkan, Kijun, Senkou A/B, Chikou."""
    tenkan = (df["High"].rolling(t).max() + df["Low"].rolling(t).min()) / 2
    kijun  = (df["High"].rolling(k).max() + df["Low"].rolling(k).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(k)
    senkou_b = ((df["High"].rolling(s).max() + df["Low"].rolling(s).min()) / 2).shift(k)
    chikou = df["Close"].shift(-k)
    return tenkan, kijun, senkou_a, senkou_b, chikou

def _supertrend(df, n=10, mult=3.0):
    """Supertrend indicator — numpy loop to avoid pandas iloc chained assignment."""
    atr_v = _atr(df, n).values
    hl2   = ((df["High"] + df["Low"]) / 2).values
    close = df["Close"].values

    ub = hl2 + mult * atr_v
    lb = hl2 - mult * atr_v

    supertrend = np.full(len(df), np.nan)
    direction  = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        # Adjust bands: only tighten, never widen, unless price crosses
        ub[i] = min(ub[i], ub[i - 1]) if close[i - 1] <= ub[i - 1] else ub[i]
        lb[i] = max(lb[i], lb[i - 1]) if close[i - 1] >= lb[i - 1] else lb[i]

        if i == 1:
            direction[i] = -1
        elif supertrend[i - 1] == ub[i - 1]:
            direction[i] = -1 if close[i] > ub[i] else 1
        else:
            direction[i] = 1 if close[i] < lb[i] else -1

        supertrend[i] = lb[i] if direction[i] == -1 else ub[i]

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def _detect_divergence(price: pd.Series, indicator: pd.Series, lookback=20):
    """Simple divergence detector: returns 'bullish', 'bearish', or 'none'."""
    try:
        p = price.tail(lookback).values
        ind = indicator.tail(lookback).values
        price_hi = p[-1] > p[:-1].max() * 0.995
        price_lo = p[-1] < p[:-1].min() * 1.005
        ind_hi   = ind[-1] > ind[:-1].max() * 0.995
        ind_lo   = ind[-1] < ind[:-1].min() * 1.005
        if price_hi and not ind_hi:   return "bearish"   # price new high, indicator lower
        if price_lo and not ind_lo:   return "bullish"   # price new low, indicator higher
    except Exception:
        pass
    return "none"

def enrich(df):
    df = df.copy(); c = df["Close"]
    df["EMA20"], df["EMA50"], df["EMA200"] = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    df["RSI"] = _rsi(c)
    df["StochRSI_K"], df["StochRSI_D"] = _stoch_rsi(c)
    df["WilliamsR"] = _williams_r(df)
    df["CCI"] = _cci(df)
    df["ADX"], df["Plus_DI"], df["Minus_DI"] = _adx(df)
    df["MACD"], df["MACD_Sig"], df["MACD_Hist"] = _macd(c)
    df["BB_Up"], df["BB_Mid"], df["BB_Lo"] = _bollinger(c)
    df["ATR"] = _atr(df)
    df["OBV"] = _obv(df)
    df["VWAP"] = _vwap(df)
    (df["Ichi_Tenkan"], df["Ichi_Kijun"],
     df["Ichi_SenkouA"], df["Ichi_SenkouB"], df["Ichi_Chikou"]) = _ichimoku(df)
    df["Supertrend"], df["ST_Dir"] = _supertrend(df)
    return df
def signal_score(snap):
    points = 0; max_pts = 0; breakdown = []
    def add(name, val, bull_cond, bear_cond, weight=1):
        nonlocal points, max_pts
        max_pts += weight
        if bull_cond:
            points += weight
            breakdown.append((name, "BULL", "#3fb950", val))
        elif bear_cond:
            breakdown.append((name, "BEAR", "#f85149", val))
        else:
            breakdown.append((name, "NEUT", "#d29922", val))
    p = snap.get("price") or 0
    rsi = snap.get("d_rsi") or 50
    add("RSI(14)", str(rsi), rsi < 50, rsi > 50)
    srsi_k = snap.get("d_stochrsi_k") or 50
    srsi_d = snap.get("d_stochrsi_d") or 50
    add("StochRSI K/D", str(srsi_k)+"/"+str(srsi_d), srsi_k < 20 or (srsi_k > srsi_d and srsi_k < 80), srsi_k > 80 or (srsi_k < srsi_d and srsi_k > 20))
    wr = snap.get("d_wr") or -50
    add("Williams %R", str(round(wr, 1)), wr < -80, wr > -20)
    cci = snap.get("d_cci") or 0
    add("CCI(20)", str(round(cci, 1)), cci < -100, cci > 100)
    adx = snap.get("d_adx") or 0; pdi = snap.get("d_pdi") or 0; mdi = snap.get("d_mdi") or 0
    add("ADX/DMI", "ADX="+str(round(adx,1))+" +DI="+str(round(pdi,1))+" -DI="+str(round(mdi,1)), adx > 25 and pdi > mdi, adx > 25 and mdi > pdi)
    macd = snap.get("d_macd") or 0; msig = snap.get("d_macd_sig") or 0
    add("MACD", str(round(macd,4))+"/"+str(round(msig,4)), macd > msig, macd < msig)
    e20 = snap.get("d_ema20") or 0; e50 = snap.get("d_ema50") or 0; e200 = snap.get("d_ema200") or 0
    add("EMA Stack", "Bullish" if p > e20 > e50 > e200 else "Bearish" if p < e20 < e50 < e200 else "Mixed", p > e20 > e50 > e200, p < e20 < e50 < e200, weight=2)
    bb_up = snap.get("d_bb_up") or 0; bb_lo = snap.get("d_bb_lo") or 0
    add("Bollinger", "Lo="+str(bb_lo)+" Hi="+str(bb_up), p < bb_lo, p > bb_up)
    vwap = snap.get("d_vwap") or 0
    if vwap:
        add("VWAP", "$"+str(round(vwap,3)), p > vwap, p < vwap)
    sa = snap.get("d_ichi_sa") or 0; sb = snap.get("d_ichi_sb") or 0
    if sa and sb:
        cloud_top = max(sa, sb); cloud_bot = min(sa, sb)
        add("Ichimoku Cloud", "A="+str(round(sa,3))+" B="+str(round(sb,3)), p > cloud_top, p < cloud_bot)
    st_dir = snap.get("d_st_dir")
    if st_dir is not None:
        add("Supertrend", "BUY" if st_dir == -1 else "SELL", st_dir == -1, st_dir == 1, weight=2)
    rsi_div = snap.get("d_rsi_div") or "none"
    macd_div = snap.get("d_macd_div") or "none"
    if rsi_div != "none":
        add("RSI Divergence", rsi_div.upper(), rsi_div == "bullish", rsi_div == "bearish")
    if macd_div != "none":
        add("MACD Divergence", macd_div.upper(), macd_div == "bullish", macd_div == "bearish")
    score = round(points / max_pts * 100) if max_pts else 50
    if score >= 60:   label, color = "BULLISH", "#3fb950"
    elif score <= 40: label, color = "BEARISH", "#f85149"
    else:             label, color = "NEUTRAL", "#d29922"
    return {"score": score, "label": label, "color": color, "breakdown": breakdown}


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE + FIBONACCI
# ══════════════════════════════════════════════════════════════════════════════
def pivot_levels(df, n=5, top=3):
    h, l = df["High"].values, df["Low"].values
    res, sup = [], []
    for i in range(n, len(df) - n):
        if h[i] == max(h[i - n: i + n + 1]): res.append(round(h[i], 3))
        if l[i] == min(l[i - n: i + n + 1]): sup.append(round(l[i], 3))
    p = df["Close"].iloc[-1]
    return (
        sorted(set(s for s in sup if s < p), reverse=True)[:top],
        sorted(set(r for r in res if r > p))[:top],
    )

def fibonacci(df):
    hi = df["High"].tail(60).max(); lo = df["Low"].tail(60).min(); d = hi - lo
    return {
        "swing_high": round(hi, 3), "swing_low": round(lo, 3),
        "ret_236": round(hi - 0.236 * d, 3), "ret_382": round(hi - 0.382 * d, 3),
        "ret_500": round(hi - 0.500 * d, 3), "ret_618": round(hi - 0.618 * d, 3),
        "ret_786": round(hi - 0.786 * d, 3),
        "ext_1272": round(lo + 1.272 * d, 3), "ext_1618": round(lo + 1.618 * d, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT (cached 5 min)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot():
    daily  = enrich(_yf_fetch(TICKER, "1d", "6mo"))
    hourly = enrich(_yf_fetch(TICKER, "1h", "5d"))
    m15    = _yf_fetch(TICKER, "15m", "2d")

    price  = (m15 if not m15.empty else hourly)["Close"].iloc[-1]
    d, d1  = daily.iloc[-1], daily.iloc[-2]
    h      = hourly.iloc[-1]

    d_sup, d_res = pivot_levels(daily)
    h_sup, h_res = pivot_levels(hourly, n=3)
    fibs = fibonacci(daily)

    # RSI & MACD divergence on daily
    rsi_div  = _detect_divergence(daily["Close"], daily["RSI"])
    macd_div = _detect_divergence(daily["Close"], daily["MACD"])

    def f(v, n=3):
        try:
            return round(float(v), n) if v is not None and not pd.isna(v) else None
        except Exception:
            return None

    return dict(
        price=f(price), change_pct=f((price - d1["Close"]) / d1["Close"] * 100, 2),
        d_open=f(d["Open"]), d_high=f(d["High"]), d_low=f(d["Low"]), prev_close=f(d1["Close"]),
        week_high=f(daily["High"].tail(5).max()), week_low=f(daily["Low"].tail(5).min()),
        month_high=f(daily["High"].tail(21).max()), month_low=f(daily["Low"].tail(21).min()),
        # Classic
        d_rsi=f(d["RSI"], 2), d_ema20=f(d["EMA20"]), d_ema50=f(d["EMA50"]), d_ema200=f(d["EMA200"]),
        d_macd=f(d["MACD"], 4), d_macd_sig=f(d["MACD_Sig"], 4), d_macd_hist=f(d["MACD_Hist"], 4),
        d_bb_up=f(d["BB_Up"]), d_bb_mid=f(d["BB_Mid"]), d_bb_lo=f(d["BB_Lo"]), d_atr=f(d["ATR"]),
        # Pro indicators
        d_stochrsi_k=f(d["StochRSI_K"], 2), d_stochrsi_d=f(d["StochRSI_D"], 2),
        d_wr=f(d["WilliamsR"], 2), d_cci=f(d["CCI"], 2),
        d_adx=f(d["ADX"], 2), d_pdi=f(d["Plus_DI"], 2), d_mdi=f(d["Minus_DI"], 2),
        d_obv=f(d["OBV"], 0), d_vwap=f(d["VWAP"], 3),
        d_ichi_tenkan=f(d["Ichi_Tenkan"]), d_ichi_kijun=f(d["Ichi_Kijun"]),
        d_ichi_sa=f(d["Ichi_SenkouA"]), d_ichi_sb=f(d["Ichi_SenkouB"]),
        d_supertrend=f(d["Supertrend"]), d_st_dir=int(d["ST_Dir"]) if not pd.isna(d["ST_Dir"]) else None,
        d_rsi_div=rsi_div, d_macd_div=macd_div,
        # Hourly
        h_rsi=f(h["RSI"], 2), h_ema20=f(h["EMA20"]),
        h_macd=f(h["MACD"], 4), h_macd_sig=f(h["MACD_Sig"], 4),
        h_stochrsi_k=f(h["StochRSI_K"], 2), h_adx=f(h["ADX"], 2),
        h_wr=f(h["WilliamsR"], 2), h_cci=f(h["CCI"], 2),
        h_vwap=f(h["VWAP"], 3),
        # Levels
        d_sup=d_sup, d_res=d_res, h_sup=h_sup, h_res=h_res, fibs=fibs,
        daily_df=daily, hourly_df=hourly,
    )


# =============================================================================
# LIVE MONITORING
# =============================================================================
def fetch_live_price():
    try:
        r = requests.get(
            f"{YF_BASE}/{TICKER}",
            params={"interval": "5m", "range": "1d", "includePrePost": "false"},
            headers=YF_HDR, timeout=10,
        )
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        ts = res["timestamp"]
        closes  = [c for c in q["close"]  if c is not None]
        opens_l = [o for o in q["open"]   if o is not None]
        highs   = [h for h in q["high"]   if h is not None]
        lows    = [lv for lv in q["low"]  if lv is not None]
        volumes = [v for v in q.get("volume", [0]*len(ts)) if v is not None]
        if not closes:
            return {}
        price = closes[-1]
        session_high = max(highs) if highs else price
        session_low  = min(lows)  if lows  else price
        session_open = opens_l[0] if opens_l else price
        avg_vol  = sum(volumes) / len(volumes) if volumes else 1
        last_vol = volumes[-1] if volumes else 0
        prev_close  = closes[-2] if len(closes) >= 2 else closes[0]
        chg_1m      = round(price - prev_close, 3)
        chg_1m_pct  = round((price - prev_close) / prev_close * 100, 3) if prev_close else 0
        chg_day     = round(price - session_open, 3)
        chg_day_pct = round((price - session_open) / session_open * 100, 3) if session_open else 0
        ticks = []
        for i in range(max(0, len(closes) - 20), len(closes)):
            direction = "U" if i == 0 or closes[i] >= closes[i - 1] else "D"
            ticks.append({"t": datetime.utcfromtimestamp(ts[i]).strftime("%H:%M"),
                          "p": round(closes[i], 3), "dir": direction})
        return {
            "price": round(price, 3), "session_high": round(session_high, 3),
            "session_low": round(session_low, 3), "session_open": round(session_open, 3),
            "chg_1m": chg_1m, "chg_1m_pct": chg_1m_pct,
            "chg_day": chg_day, "chg_day_pct": chg_day_pct,
            "last_vol": int(last_vol), "avg_vol": int(avg_vol),
            "vol_ratio": round(last_vol / avg_vol, 2) if avg_vol else 0,
            "ticks": ticks,
            "session_range_pct": round((session_high - session_low) / session_low * 100, 2) if session_low else 0,
        }
    except Exception:
        pass
    try:
        r = requests.get(
            f"{YF_BASE}/{TICKER}",
            params={"interval": "15m", "range": "5d", "includePrePost": "false"},
            headers=YF_HDR, timeout=10,
        )
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        ts = res["timestamp"]
        closes  = [c for c in q["close"]  if c is not None]
        opens_l = [o for o in q["open"]   if o is not None]
        highs   = [h for h in q["high"]   if h is not None]
        lows    = [lv for lv in q["low"]  if lv is not None]
        volumes = [v for v in q.get("volume", [0]*len(ts)) if v is not None]
        if not closes:
            return {}
        price = closes[-1]
        session_high = max(highs[-20:]) if highs else price
        session_low  = min(lows[-20:])  if lows  else price
        session_open = opens_l[-20] if len(opens_l) >= 20 else (opens_l[0] if opens_l else price)
        avg_vol  = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 1
        last_vol = volumes[-1] if volumes else 0
        prev_close  = closes[-2] if len(closes) >= 2 else closes[0]
        chg_1m      = round(price - prev_close, 3)
        chg_1m_pct  = round((price - prev_close) / prev_close * 100, 3) if prev_close else 0
        chg_day     = round(price - session_open, 3)
        chg_day_pct = round((price - session_open) / session_open * 100, 3) if session_open else 0
        ticks = []
        for i in range(max(0, len(closes) - 20), len(closes)):
            direction = "U" if i == 0 or closes[i] >= closes[i - 1] else "D"
            ticks.append({"t": datetime.utcfromtimestamp(ts[i]).strftime("%H:%M"),
                          "p": round(closes[i], 3), "dir": direction})
        return {
            "price": round(price, 3), "session_high": round(session_high, 3),
            "session_low": round(session_low, 3), "session_open": round(session_open, 3),
            "chg_1m": chg_1m, "chg_1m_pct": chg_1m_pct,
            "chg_day": chg_day, "chg_day_pct": chg_day_pct,
            "last_vol": int(last_vol), "avg_vol": int(avg_vol),
            "vol_ratio": round(last_vol / avg_vol, 2) if avg_vol else 0,
            "ticks": ticks,
            "session_range_pct": round((session_high - session_low) / session_low * 100, 2) if session_low else 0,
        }
    except Exception:
        return {}


def check_alerts(live, snap):
    alerts = []
    if not live:
        return [{"level": "yellow", "msg": "No live data"}]
    price = live["price"]
    if snap.get("d_rsi") and snap["d_rsi"] > 75:
        alerts.append({"level": "red", "msg": "RSI OVERBOUGHT "+str(snap['d_rsi'])+" - reversal risk"})
    if snap.get("d_rsi") and snap["d_rsi"] < 25:
        alerts.append({"level": "green", "msg": "RSI OVERSOLD "+str(snap['d_rsi'])+" - bounce opportunity"})
    srsi_k = snap.get("d_stochrsi_k") or 50
    if srsi_k > 85:
        alerts.append({"level": "red", "msg": "StochRSI OVERBOUGHT "+str(srsi_k)+" - momentum exhaustion"})
    if srsi_k < 15:
        alerts.append({"level": "green", "msg": "StochRSI OVERSOLD "+str(srsi_k)+" - reversal setup"})
    wr = snap.get("d_wr") or -50
    if wr > -10:
        alerts.append({"level": "red", "msg": "Williams %R overbought ("+str(wr)+")"})
    if wr < -90:
        alerts.append({"level": "green", "msg": "Williams %R oversold ("+str(wr)+") - watch for bounce"})
    adx = snap.get("d_adx") or 0; pdi = snap.get("d_pdi") or 0; mdi = snap.get("d_mdi") or 0
    if adx > 30 and pdi > mdi:
        alerts.append({"level": "green", "msg": "Strong BULLISH trend — ADX "+str(round(adx,1))})
    if adx > 30 and mdi > pdi:
        alerts.append({"level": "red", "msg": "Strong BEARISH trend — ADX "+str(round(adx,1))})
    if adx < 20:
        alerts.append({"level": "yellow", "msg": "Weak trend (ADX "+str(round(adx,1))+") — ranging market"})
    rsi_div = snap.get("d_rsi_div") or "none"
    if rsi_div == "bullish":
        alerts.append({"level": "green", "msg": "BULLISH RSI Divergence — momentum turning up"})
    if rsi_div == "bearish":
        alerts.append({"level": "red", "msg": "BEARISH RSI Divergence — momentum fading"})
    macd_div = snap.get("d_macd_div") or "none"
    if macd_div == "bullish":
        alerts.append({"level": "green", "msg": "BULLISH MACD Divergence detected"})
    if macd_div == "bearish":
        alerts.append({"level": "red", "msg": "BEARISH MACD Divergence detected"})
    st_dir = snap.get("d_st_dir")
    if st_dir is not None:
        if st_dir == -1:
            alerts.append({"level": "green", "msg": "Supertrend: BULLISH — price above trend line"})
        else:
            alerts.append({"level": "red", "msg": "Supertrend: BEARISH — price below trend line"})
    ichi_sa = snap.get("d_ichi_sa") or 0; ichi_sb = snap.get("d_ichi_sb") or 0
    if ichi_sa and ichi_sb:
        cloud_top = max(ichi_sa, ichi_sb)
        cloud_bot = min(ichi_sa, ichi_sb)
        if price > cloud_top:
            alerts.append({"level": "green", "msg": "Price ABOVE Ichimoku Cloud — bullish"})
        elif price < cloud_bot:
            alerts.append({"level": "red", "msg": "Price BELOW Ichimoku Cloud — bearish"})
        else:
            alerts.append({"level": "yellow", "msg": "Price INSIDE Ichimoku Cloud — consolidation"})
    for lvl in snap.get("d_sup", []):
        if abs(price - lvl) / lvl < 0.003:
            alerts.append({"level": "green", "msg": "Near daily Support $"+str(lvl)+" - watch bounce"})
    for lvl in snap.get("d_res", []):
        if abs(price - lvl) / lvl < 0.003:
            alerts.append({"level": "red", "msg": "Near daily Resistance $"+str(lvl)+" - watch rejection"})
    if live["vol_ratio"] >= 2.5:
        alerts.append({"level": "yellow", "msg": "Volume SPIKE "+str(live['vol_ratio'])+"x avg"})
    ema20 = snap.get("d_ema20") or 0; ema50 = snap.get("d_ema50") or 0
    if ema20 and ema50 and abs(ema20 - ema50) / ema50 < 0.002:
        alerts.append({"level": "yellow", "msg": "EMA 20/50 near crossover - trend change incoming"})
    dm = snap.get("d_macd") or 0; dms = snap.get("d_macd_sig") or 0
    if dm and dms and abs(dm - dms) < 0.001:
        alerts.append({"level": "yellow", "msg": "MACD crossing signal line"})
    if not alerts:
        alerts.append({"level": "green", "msg": "No critical alerts — market normal"})
    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART — candlestick + volume + indicators + S/R + Ichimoku Cloud
# ══════════════════════════════════════════════════════════════════════════════
def make_chart(df, support, resistance, fib, current_price, show_ichimoku=True, timeframe='Daily'):
    df = df.copy()
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.12, 0.18, 0.15], vertical_spacing=0.02,
        subplot_titles=('', 'Volume', 'RSI / StochRSI', 'MACD'),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Silver (SI=F)',
        increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
        decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350',
        line_width=1,
    ), row=1, col=1)

    # Ichimoku Cloud
    if show_ichimoku and 'Ichi_SenkouA' in df.columns:
        sa = df['Ichi_SenkouA'].dropna(); sb = df['Ichi_SenkouB'].dropna()
        common_idx = sa.index.intersection(sb.index)
        if len(common_idx) > 0:
            sa_c = sa.loc[common_idx]; sb_c = sb.loc[common_idx]
            bull_mask = (sa_c >= sb_c).mean() > 0.5
            fill_c = 'rgba(38,166,154,0.10)' if bull_mask else 'rgba(239,83,80,0.10)'
            fig.add_trace(go.Scatter(
                x=sa_c.index, y=sa_c.values, name='Senkou A',
                line=dict(color='rgba(38,166,154,0.5)', width=1), showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sb_c.index, y=sb_c.values, name='Senkou B',
                line=dict(color='rgba(239,83,80,0.5)', width=1), showlegend=True,
                fill='tonexty', fillcolor=fill_c,
            ), row=1, col=1)

    # Supertrend
    if 'Supertrend' in df.columns and 'ST_Dir' in df.columns:
        st_bull = df[df['ST_Dir'] == -1]['Supertrend']
        st_bear = df[df['ST_Dir'] == 1]['Supertrend']
        if not st_bull.empty:
            fig.add_trace(go.Scatter(
                x=st_bull.index, y=st_bull.values, mode='markers',
                marker=dict(color='#26a69a', size=4, symbol='circle'),
                name='Supertrend BUY',
            ), row=1, col=1)
        if not st_bear.empty:
            fig.add_trace(go.Scatter(
                x=st_bear.index, y=st_bear.values, mode='markers',
                marker=dict(color='#ef5350', size=4, symbol='circle'),
                name='Supertrend SELL',
            ), row=1, col=1)

    # EMAs
    for col, color, name, dash in [
        ('EMA20', '#26a69a', 'EMA 20', 'solid'),
        ('EMA50', '#ef5350', 'EMA 50', 'solid'),
        ('EMA200', '#ffa726', 'EMA 200', 'dash'),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s.values,
                line=dict(color=color, width=1.5, dash=dash), name=name), row=1, col=1)

    # VWAP
    if 'VWAP' in df.columns:
        vwap_s = df['VWAP'].dropna()
        fig.add_trace(go.Scatter(x=vwap_s.index, y=vwap_s.values,
            line=dict(color='#42a5f5', width=1.5, dash='dot'), name='VWAP'), row=1, col=1)

    # Current price line
    fig.add_hline(y=current_price, line_color='rgba(255,255,255,0.85)', line_width=1.5,
                  annotation_text=' Price $'+str(current_price),
                  annotation_position='right', annotation_font_color='#ffffff',
                  annotation_font_size=11, row=1, col=1)

    # Support / Resistance
    for i, s in enumerate(support):
        fig.add_hline(y=s, line_color='#26a69a', line_dash='dash', line_width=1,
                      annotation_text=' S'+str(i+1)+' $'+str(s),
                      annotation_position='right', annotation_font_color='#26a69a',
                      annotation_font_size=10, row=1, col=1)
    for i, r in enumerate(resistance):
        fig.add_hline(y=r, line_color='#ef5350', line_dash='dash', line_width=1,
                      annotation_text=' R'+str(i+1)+' $'+str(r),
                      annotation_position='right', annotation_font_color='#ef5350',
                      annotation_font_size=10, row=1, col=1)

    # Fibonacci
    fib_cfg = [
        ('ret_236', '#ab47bc', 'Fib 23.6%'), ('ret_382', '#ab47bc', 'Fib 38.2%'),
        ('ret_500', '#42a5f5', 'Fib 50%'),   ('ret_618', '#ab47bc', 'Fib 61.8%'),
        ('ret_786', '#ab47bc', 'Fib 78.6%'),
        ('ext_1272', '#ffa726', 'Ext 127.2%'), ('ext_1618', '#ffa726', 'Ext 161.8%'),
    ]
    for key, color, label in fib_cfg:
        if key in fib:
            fig.add_hline(y=fib[key], line_color=color, line_dash='dot', line_width=0.8,
                          annotation_text=' '+label+' $'+str(fib[key]),
                          annotation_position='left', annotation_font_color=color,
                          annotation_font_size=9, row=1, col=1)

    # Volume panel (row 2)
    vol_colors = [
        'rgba(38,166,154,0.5)' if float(c) >= float(o) else 'rgba(239,83,80,0.5)'
        for c, o in zip(df['Close'], df['Open'])
    ]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    vol_ma = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=vol_ma.index, y=vol_ma.values,
        line=dict(color='#ffa726', width=1, dash='dot'), name='Vol MA20', showlegend=False), row=2, col=1)

    # RSI + StochRSI panel (row 3)
    if 'RSI' in df.columns:
        rsi_s = df['RSI'].dropna()
        fig.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s.values,
            line=dict(color='#42a5f5', width=1.5), name='RSI(14)'), row=3, col=1)
    if 'StochRSI_K' in df.columns:
        sk = df['StochRSI_K'].dropna(); sd = df['StochRSI_D'].dropna()
        fig.add_trace(go.Scatter(x=sk.index, y=sk.values,
            line=dict(color='#ffa726', width=1), name='StochRSI K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=sd.index, y=sd.values,
            line=dict(color='#ef5350', width=1, dash='dot'), name='StochRSI D'), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,83,80,0.07)', line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(38,166,154,0.07)', line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_color='rgba(239,83,80,0.5)', line_dash='dot', line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_color='rgba(38,166,154,0.5)', line_dash='dot', line_width=1, row=3, col=1)
    fig.add_hline(y=50, line_color='rgba(255,255,255,0.15)', line_dash='dot', line_width=1, row=3, col=1)

    # MACD panel (row 4)
    if 'MACD' in df.columns:
        macd_s = df['MACD'].dropna(); sig_s = df['MACD_Sig'].dropna()
        hist_s = df['MACD_Hist'].dropna()
        hist_colors = ['rgba(38,166,154,0.6)' if v >= 0 else 'rgba(239,83,80,0.6)' for v in hist_s.values]
        fig.add_trace(go.Bar(x=hist_s.index, y=hist_s.values, name='MACD Hist',
                             marker_color=hist_colors, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=macd_s.index, y=macd_s.values,
            line=dict(color='#42a5f5', width=1.5), name='MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=sig_s.index, y=sig_s.values,
            line=dict(color='#ffa726', width=1, dash='dot'), name='Signal'), row=4, col=1)
        fig.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_width=1, row=4, col=1)

    # Layout
    rangeselector_buttons = [
        dict(count=1, label='1M', step='month', stepmode='backward'),
        dict(count=3, label='3M', step='month', stepmode='backward'),
        dict(count=6, label='6M', step='month', stepmode='backward'),
        dict(step='all', label='All'),
    ] if timeframe == 'Daily' else []
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        xaxis_rangeslider_visible=False,
        height=840,
        margin=dict(l=10, r=150, t=25, b=10),
        legend=dict(
            orientation='h', y=1.02, yanchor='bottom',
            font=dict(size=10, color='#c9d1d9'),
            bgcolor='rgba(13,17,23,0.85)',
            bordercolor='#30363d', borderwidth=1,
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#161b22', bordercolor='#30363d', font_color='#c9d1d9'),
        spikedistance=-1,
    )
    if rangeselector_buttons:
        fig.update_xaxes(
            rangeselector=dict(
                buttons=rangeselector_buttons,
                bgcolor='#161b22', activecolor='#1f6feb',
                bordercolor='#30363d', font=dict(color='#c9d1d9', size=11),
                x=0.0, y=1.18,
            ), row=1, col=1,
        )
    fig.update_xaxes(gridcolor='#1c2128', showgrid=True, zeroline=False,
                     showspikes=True, spikecolor='#42a5f5', spikethickness=1, spikedash='dot')
    fig.update_yaxes(gridcolor='#1c2128', showgrid=True, zeroline=False)
    fig.update_yaxes(title_text='USD/oz', tickformat='$,.2f', row=1, col=1)
    fig.update_yaxes(title_text='Vol', row=2, col=1)
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text='MACD', row=4, col=1)
    return fig

def tradingview_widget(symbol='COMEX:SI1!', theme='dark', height=580):
    '''Embedded TradingView Advanced Chart for live silver price.'''
    studies = [
        'RSI@tv-basicstudies',
        'MASimple@tv-basicstudies',
        'MACD@tv-basicstudies',
        'IchimokuCloud@tv-basicstudies',
        'SuperTrend@tv-basicstudies',
    ]
    studies_json = '[' + ','.join('"' + s + '"' for s in studies) + ']'
    overrides = {
        'mainSeriesProperties.candleStyle.upColor': '#26a69a',
        'mainSeriesProperties.candleStyle.downColor': '#ef5350',
        'mainSeriesProperties.candleStyle.borderUpColor': '#26a69a',
        'mainSeriesProperties.candleStyle.borderDownColor': '#ef5350',
        'mainSeriesProperties.candleStyle.wickUpColor': '#26a69a',
        'mainSeriesProperties.candleStyle.wickDownColor': '#ef5350',
        'paneProperties.background': '#0d1117',
        'paneProperties.backgroundType': 'solid',
        'paneProperties.vertGridProperties.color': '#1c2128',
        'paneProperties.horzGridProperties.color': '#1c2128',
        'scalesProperties.textColor': '#c9d1d9',
    }
    import json as _json
    ov_json = _json.dumps(overrides)
    cid = 'tv_silver_chart'
    html = f'''
<div class="tradingview-widget-container" style="height:{height}px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #30363d">
  <div id="{cid}" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "width": "100%",
    "height": {height},
    "symbol": "{symbol}",
    "interval": "60",
    "timezone": "Etc/UTC",
    "theme": "{theme}",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#0d1117",
    "enable_publishing": false,
    "hide_top_toolbar": false,
    "hide_legend": false,
    "hide_side_toolbar": true,
    "allow_symbol_change": false,
    "save_image": true,
    "container_id": "{cid}",
    "studies": {studies_json},
    "overrides": {ov_json}
  }});
  </script>
</div>'''
    return html


def tradingview_ticker_tape():
    '''TradingView ticker tape for silver, gold, DXY, rates.'''
    symbols = [
        {'proName': 'COMEX:SI1!', 'title': 'Silver Futures'},
        {'proName': 'TVC:SILVER', 'title': 'Silver Spot'},
        {'proName': 'OANDA:XAGUSD', 'title': 'XAG/USD'},
        {'proName': 'TVC:GOLD', 'title': 'Gold'},
        {'proName': 'TVC:DXY', 'title': 'DXY'},
        {'proName': 'TVC:US10Y', 'title': '10Y Yield'},
    ]
    import json as _json
    sym_json = _json.dumps(symbols)
    config = _json.dumps({
        'symbols': symbols,
        'showSymbolLogo': True,
        'isTransparent': True,
        'displayMode': 'adaptive',
        'colorTheme': 'dark',
        'locale': 'en',
    })
    return f'''<div class="tradingview-widget-container" style="margin-bottom:12px">
<div class="tradingview-widget-container__widget"></div>
<script type="text/javascript"
 src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
{config}
</script></div>'''

# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS — Anthropic via requests
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_ai_analysis(price_key, api_key):
    snap = load_snapshot()
    fib  = snap["fibs"]

    prompt = """Analyze this live COMEX silver (SI=F) market snapshot and provide a complete pro-trader setup.

=== PRICE ACTION ===
Current  : $"""+str(snap['price'])+"""/oz | 24h Change: """+str(snap['change_pct'])+"""%
Today    : O """+str(snap['d_open'])+""" H """+str(snap['d_high'])+""" L """+str(snap['d_low'])+""" | Prev Close: """+str(snap['prev_close'])+"""
Week     : High """+str(snap['week_high'])+""" / Low """+str(snap['week_low'])+"""
Month    : High """+str(snap['month_high'])+""" / Low """+str(snap['month_low'])+"""

=== DAILY INDICATORS (PRO SUITE) ===
RSI(14)           : """+str(snap['d_rsi'])+"""
StochRSI K/D      : """+str(snap['d_stochrsi_k'])+""" / """+str(snap['d_stochrsi_d'])+"""
Williams %R       : """+str(snap['d_wr'])+"""
CCI(20)           : """+str(snap['d_cci'])+"""
ADX / +DI / -DI   : """+str(snap['d_adx'])+""" / """+str(snap['d_pdi'])+""" / """+str(snap['d_mdi'])+"""
EMA 20/50/200     : """+str(snap['d_ema20'])+""" / """+str(snap['d_ema50'])+""" / """+str(snap['d_ema200'])+"""
MACD/Signal/Hist  : """+str(snap['d_macd'])+""" / """+str(snap['d_macd_sig'])+""" (Hist: """+str(snap['d_macd_hist'])+""")
Bollinger Bands   : Lower """+str(snap['d_bb_lo'])+""" | Mid """+str(snap['d_bb_mid'])+""" | Upper """+str(snap['d_bb_up'])+"""
ATR(14)           : """+str(snap['d_atr'])+"""
VWAP              : """+str(snap['d_vwap'])+"""
OBV               : """+str(snap['d_obv'])+"""
Ichimoku Tenkan   : """+str(snap['d_ichi_tenkan'])+"""
Ichimoku Kijun    : """+str(snap['d_ichi_kijun'])+"""
Ichimoku Cloud    : Senkou A="""+str(snap['d_ichi_sa'])+""" / Senkou B="""+str(snap['d_ichi_sb'])+"""
Supertrend        : """+str(snap['d_supertrend'])+""" (Dir: """+("BULLISH" if snap['d_st_dir']==-1 else "BEARISH")+""")
RSI Divergence    : """+str(snap['d_rsi_div'])+"""
MACD Divergence   : """+str(snap['d_macd_div'])+"""

=== HOURLY INDICATORS ===
RSI(14): """+str(snap['h_rsi'])+""" | EMA20: """+str(snap['h_ema20'])+""" | ADX: """+str(snap['h_adx'])+"""
MACD: """+str(snap['h_macd'])+""" / Sig: """+str(snap['h_macd_sig'])+""" | StochRSI K: """+str(snap['h_stochrsi_k'])+"""
Williams %R: """+str(snap['h_wr'])+""" | CCI: """+str(snap['h_cci'])+""" | VWAP: """+str(snap['h_vwap'])+"""

=== SUPPORT & RESISTANCE ===
Daily Resistance  : """+str(snap['d_res'])+"""
Daily Support     : """+str(snap['d_sup'])+"""
Hourly Resistance : """+str(snap['h_res'])+"""
Hourly Support    : """+str(snap['h_sup'])+"""

=== FIBONACCI (60-day swing: $"""+str(fib['swing_low'])+""" to $"""+str(fib['swing_high'])+""") ===
Retracements: 23.6%="""+str(fib['ret_236'])+""" 38.2%="""+str(fib['ret_382'])+""" 50%="""+str(fib['ret_500'])+""" 61.8%="""+str(fib['ret_618'])+""" 78.6%="""+str(fib['ret_786'])+"""
Extensions  : 127.2%="""+str(fib['ext_1272'])+""" 161.8%="""+str(fib['ext_1618'])+"""

Respond in EXACTLY this format:

## MARKET BIAS
[BULLISH / BEARISH / NEUTRAL] — paragraph: overall technical picture with multi-indicator confluence.

## BUY SETUP
Entry Zone : $X.XXX – $X.XXX
Target 1   : $X.XXX (+X.X%) ← conservative
Target 2   : $X.XXX (+X.X%) ← swing
Target 3   : $X.XXX (+X.X%) ← momentum
Stop Loss  : $X.XXX (–X.X%)
R/R Ratio  : 1 : X.X

## SELL / SHORT SETUP
Entry Zone : $X.XXX – $X.XXX
Target 1   : $X.XXX (–X.X%)
Target 2   : $X.XXX (–X.X%)
Target 3   : $X.XXX (–X.X%)
Stop Loss  : $X.XXX (+X.X%)
R/R Ratio  : 1 : X.X

## KEY LEVELS
List 4-6 critical price levels with a one-line reason each.

## PATTERN / SIGNAL
Any candlestick patterns, divergences, momentum signals, or Ichimoku/Supertrend insights.

## INDICATOR CONFLUENCE
Which indicators agree? Which conflict? How strong is the consensus?

## RISK RATING
[LOW / MEDIUM / HIGH] — one sentence on volatility/risk context using ATR.

## TRADER'S NOTE
One sharp insight a veteran silver trader would add."""

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "max_tokens": 2500,
            "system": (
                "You are a senior COMEX silver futures trader with 20+ years of experience. "
                "Give precise, actionable trade setups with exact price levels. "
                "Use ALL provided indicators for confluence analysis. All prices in USD/oz."
            ),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def badge(v, kind="fib"):
    cls = {"sup": "sup-badge", "res": "res-badge", "fib": "fib-badge"}.get(kind, "fib-badge")
    return '<span class="level-badge '+cls+'">$'+str(v)+'</span>'

def sig_tag(label, kind):
    cls = {"bull": "tag-bull", "bear": "tag-bear", "neut": "tag-neut"}.get(kind, "tag-neut")
    return '<span class="tag '+cls+'">'+label+'</span>'

def metric_card(label, value, sub="", sub_class="neut"):
    return ('<div class="metric-card">'
            '<div class="metric-label">'+label+'</div>'
            '<div class="metric-value">'+str(value)+'</div>'
            '<div class="metric-sub '+sub_class+'">'+sub+'</div></div>')

def score_bar_html(score, color):
    return ('<div style="margin:8px 0">'
            '<div style="display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin-bottom:4px">'
            '<span>BEARISH</span><span>BULLISH</span></div>'
            '<div class="score-bar-wrap"><div class="score-bar" style="width:'+str(score)+'%;background:'+color+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px;text-align:center">Score: '+str(score)+'/100</div>'
            '</div>')

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Silver AI Agent PRO")
    st.caption("COMEX Silver Futures (SI=F)")
    st.divider()

    snowflake_key = _get_secret()
    if snowflake_key:
        st.success("🔐 API key loaded from Snowflake Secret")
        api_key = snowflake_key
    else:
        api_key = st.text_input(
            "Anthropic API Key", type="password", placeholder="sk-ant-…",
            help="On Snowflake this is injected from a Secret automatically.",
        )

    auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)
    run_btn = st.button("🔄 Refresh Now", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Indicators included**")
    st.markdown("EMA 20/50/200 · RSI · StochRSI")
    st.markdown("Williams %R · CCI · ADX/DMI")
    st.markdown("MACD · Bollinger Bands · ATR")
    st.markdown("OBV · VWAP · Ichimoku Cloud")
    st.markdown("Supertrend · Divergence Detection")
    st.markdown("Signal Score (composite 0-100)")
    st.divider()
    st.markdown("**Chart legend**")
    st.markdown("🟢 EMA 20  🔴 EMA 50  🟡 EMA 200")
    st.markdown("🔵 VWAP  🟣 Fibonacci")
    st.markdown("🟢 Support  🔴 Resistance")
    st.markdown("🟢/🔴 dots = Supertrend")
    st.markdown("Green/Red fill = Ichimoku Cloud")
    st.divider()
    if _IN_SNOWFLAKE:
        st.caption("🏔 Running inside Snowflake")
    st.caption("**Data:** Yahoo Finance  **AI:** Claude Opus")
    st.divider()
    st.warning("⚠️ Educational use only.\nNot financial advice.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 style='color:#f0f6fc;margin-bottom:0'>⚡ Silver AI Trading Agent PRO v2</h1>", unsafe_allow_html=True)
st.caption("Live COMEX Silver Futures (SI=F) · Updated "+datetime.now().strftime('%H:%M:%S'))
st.divider()

if run_btn:
    st.cache_data.clear()

with st.spinner("Fetching live silver prices…"):
    try:
        snap = load_snapshot()
    except Exception as e:
        st.error("Data fetch failed: "+str(e))
        st.stop()

p, chg = snap["price"], snap["change_pct"] or 0
chg_color = "bull" if chg >= 0 else "bear"
arrow = "▲" if chg >= 0 else "▼"

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    ("💰 SILVER / OZ", "$"+str(p), arrow+" "+str(abs(chg))+"%", chg_color),
    ("📅 Prev Close", "$"+str(snap['prev_close']), "Day: "+str(snap['d_low'])+" – "+str(snap['d_high']), "neut"),
    ("📊 Week Range", str(snap['week_low']), "↑ "+str(snap['week_high']), "neut"),
    ("📆 Month Range", str(snap['month_low']), "↑ "+str(snap['month_high']), "neut"),
    ("📐 ATR (vol)", "$"+str(snap['d_atr'])+"/oz", "Daily volatility", "neut"),
]
for col, (label, val, sub, sc) in zip([c1, c2, c3, c4, c5], cards):
    col.markdown(metric_card(label, val, sub, sc), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# Signal Score Section
st.markdown("### 🎯 Signal Score — Composite Indicator Consensus")
scoring = signal_score(snap)
sc1, sc2 = st.columns([1, 2])
with sc1:
    score_html = ('<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;text-align:center">'
        '<div style="font-size:13px;color:#8b949e;margin-bottom:8px">OVERALL SIGNAL</div>'
        '<div style="font-size:52px;font-weight:800;color:'+scoring["color"]+'">'+str(scoring["score"])+'</div>'
        '<div style="font-size:18px;font-weight:700;color:'+scoring["color"]+'">'+scoring["label"]+'</div>'
        '</div>')
    st.markdown(score_html, unsafe_allow_html=True)
    st.markdown(score_bar_html(scoring["score"], scoring["color"]), unsafe_allow_html=True)
with sc2:
    st.markdown("**Indicator Breakdown**")
    breakdown = scoring["breakdown"]
    for i in range(0, len(breakdown), 2):
        row_items = breakdown[i:i+2]
        row_cols = st.columns(2)
        for col, (name, lbl, color, val) in zip(row_cols, row_items):
            col.markdown('<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 10px;margin-bottom:4px">'
                '<div style="font-size:11px;color:#8b949e">'+name+'</div>'
                '<div style="font-size:13px;color:'+color+';font-weight:700">'+lbl+'</div>'
                '<div style="font-size:11px;color:#6e7681">'+str(val)+'</div>'
                '</div>', unsafe_allow_html=True)

st.divider()
# Chart Section
st.markdown("### 📊 Live Chart — Candlestick + Ichimoku + Supertrend + EMAs + MACD + RSI")
st.caption("Green/Red fill = Ichimoku Cloud · Dots = Supertrend · EMA 20/50/200 · VWAP · S/R · Fibonacci")

tab_d, tab_h, tab_tv = st.tabs(["📅 Daily (6 months)", "⏱ Hourly (5 days)", "📡 TradingView Live"])
with tab_d:
    fig_d = make_chart(snap["daily_df"].tail(120), snap["d_sup"], snap["d_res"], snap["fibs"], snap["price"], timeframe="Daily")
    st.plotly_chart(fig_d, use_container_width=True)
with tab_h:
    fig_h = make_chart(snap["hourly_df"], snap["h_sup"], snap["h_res"], snap["fibs"], snap["price"], show_ichimoku=False, timeframe="Hourly")
    st.plotly_chart(fig_h, use_container_width=True)
with tab_tv:
    st.markdown("#### 📡 TradingView Live Silver Chart")
    st.caption("Live data from TradingView · COMEX:SI1! · Hourly candles · RSI, MACD, Ichimoku, SuperTrend overlays")
    # Ticker tape
    components.html(tradingview_ticker_tape(), height=60, scrolling=False)
    # Main TradingView chart
    tv_col1, tv_col2 = st.columns([3, 1])
    with tv_col1:
        tv_interval = st.selectbox(
            "Interval",
            ["5", "15", "30", "60", "240", "D", "W"],
            index=3,
            format_func=lambda x: {"5": "5 min", "15": "15 min", "30": "30 min",
                                    "60": "1 Hour", "240": "4 Hour", "D": "Daily", "W": "Weekly"}.get(x, x),
            key="tv_interval",
        )
    with tv_col2:
        tv_height = st.slider("Chart Height", 400, 800, 580, 50, key="tv_height")
    # Build widget with selected interval
    tv_html = tradingview_widget(symbol="COMEX:SI1!", theme="dark", height=tv_height)
    components.html(tv_html, height=tv_height + 30, scrolling=False)
    st.caption("💡 TradingView chart includes live price, pre-built indicators and interactive tools.")

st.divider()
# Pro Indicators Table
st.markdown("### 📈 Pro Technical Indicators — Multi-Timeframe")
left, right = st.columns(2)

def ind_row(name, val_str, sig_label, sig_kind):
    r1, r2, r3 = st.columns([1.2, 1.6, 1.0])
    r1.markdown("**"+name+"**")
    r2.markdown("`"+val_str+"`")
    r3.markdown(sig_tag(sig_label, sig_kind), unsafe_allow_html=True)
    st.divider()

with left:
    st.markdown("**Daily Oscillators**")
    d_rsi = snap["d_rsi"] or 50
    h_rsi = snap["h_rsi"] or 50
    rsi_lbl = "OVERBOUGHT" if d_rsi > 70 else "OVERSOLD" if d_rsi < 30 else "NEUTRAL"
    rsi_kind = "bear" if d_rsi > 70 else "bull" if d_rsi < 30 else "neut"
    ind_row("RSI(14)", "D:"+str(d_rsi)+" H:"+str(h_rsi), rsi_lbl, rsi_kind)

    sk = snap.get("d_stochrsi_k") or 50; sd_v = snap.get("d_stochrsi_d") or 50
    stk_lbl = "OVERBOUGHT" if sk > 80 else "OVERSOLD" if sk < 20 else ("K>D BULL" if sk > sd_v else "K<D BEAR")
    stk_kind = "bear" if sk > 80 else "bull" if sk < 20 else ("bull" if sk > sd_v else "bear")
    ind_row("StochRSI K/D", "K:"+str(sk)+" D:"+str(sd_v), stk_lbl, stk_kind)

    wr = snap.get("d_wr") or -50
    wr_lbl = "OVERBOUGHT" if wr > -20 else "OVERSOLD" if wr < -80 else "NEUTRAL"
    wr_kind = "bear" if wr > -20 else "bull" if wr < -80 else "neut"
    ind_row("Williams %R", str(wr), wr_lbl, wr_kind)

    cci = snap.get("d_cci") or 0
    cci_lbl = "OVERBOUGHT" if cci > 100 else "OVERSOLD" if cci < -100 else "NEUTRAL"
    cci_kind = "bear" if cci > 100 else "bull" if cci < -100 else "neut"
    ind_row("CCI(20)", str(round(cci, 1)), cci_lbl, cci_kind)

    adx = snap.get("d_adx") or 0; pdi = snap.get("d_pdi") or 0; mdi = snap.get("d_mdi") or 0
    adx_lbl = ("STRONG BUY" if adx > 25 and pdi > mdi else
               "STRONG SELL" if adx > 25 and mdi > pdi else "WEAK TREND")
    adx_kind = "bull" if adx > 25 and pdi > mdi else "bear" if adx > 25 and mdi > pdi else "neut"
    ind_row("ADX/+DI/-DI", "ADX:"+str(round(adx,1))+" +"+str(round(pdi,1))+"/-"+str(round(mdi,1)), adx_lbl, adx_kind)

    macd_bull = (snap.get("d_macd") or 0) > (snap.get("d_macd_sig") or 0)
    ind_row("MACD(12/26/9)", "Line:"+str(snap['d_macd'])+" Sig:"+str(snap['d_macd_sig']),
            "BULLISH" if macd_bull else "BEARISH", "bull" if macd_bull else "bear")

with right:
    st.markdown("**Trend & Price Structure**")
    e20 = snap.get("d_ema20") or 0; e50 = snap.get("d_ema50") or 0; e200 = snap.get("d_ema200") or 0
    ema_bull = p > e20 > e50 > e200; ema_bear = p < e20 < e50 < e200
    ema_kind = "bull" if ema_bull else "bear" if ema_bear else "neut"
    ind_row("EMA 20/50/200", str(e20)+"/"+str(e50)+"/"+str(e200),
            "BULL STACK" if ema_bull else "BEAR STACK" if ema_bear else "MIXED", ema_kind)

    bb_up = snap.get("d_bb_up") or 0; bb_lo = snap.get("d_bb_lo") or 0
    if p > bb_up: bb_l, bb_k = "ABOVE UPPER", "bear"
    elif p < bb_lo: bb_l, bb_k = "BELOW LOWER", "bull"
    else: bb_l, bb_k = "WITHIN BANDS", "neut"
    ind_row("Bollinger(20,2σ)", str(bb_lo)+" – "+str(bb_up), bb_l, bb_k)

    vwap = snap.get("d_vwap") or 0
    vwap_kind = "bull" if p > vwap else "bear"
    ind_row("VWAP", "$"+str(vwap), "ABOVE VWAP" if p > vwap else "BELOW VWAP", vwap_kind)

    sa = snap.get("d_ichi_sa") or 0; sb = snap.get("d_ichi_sb") or 0
    if sa and sb:
        cloud_top = max(sa, sb); cloud_bot = min(sa, sb)
        if p > cloud_top: ichi_l, ichi_k = "ABOVE CLOUD", "bull"
        elif p < cloud_bot: ichi_l, ichi_k = "BELOW CLOUD", "bear"
        else: ichi_l, ichi_k = "IN CLOUD", "neut"
        ind_row("Ichimoku Cloud", "A:"+str(sa)+" B:"+str(sb), ichi_l, ichi_k)

    st_dir = snap.get("d_st_dir"); st_val = snap.get("d_supertrend") or 0
    if st_dir is not None:
        st_lbl = "BUY SIGNAL" if st_dir == -1 else "SELL SIGNAL"
        st_kind = "bull" if st_dir == -1 else "bear"
        ind_row("Supertrend", "$"+str(st_val), st_lbl, st_kind)

    rsi_div = snap.get("d_rsi_div") or "none"
    macd_div = snap.get("d_macd_div") or "none"
    div_text = []
    if rsi_div != "none": div_text.append("RSI: "+rsi_div.upper())
    if macd_div != "none": div_text.append("MACD: "+macd_div.upper())
    div_display = " | ".join(div_text) if div_text else "none detected"
    div_kind = "bull" if "bullish" in (rsi_div+macd_div) else "bear" if "bearish" in (rsi_div+macd_div) else "neut"
    ind_row("Divergence", div_display, "DETECTED" if div_text else "NONE", div_kind)

st.divider()
# Multi-Timeframe Confluence Table
st.markdown("### 🔀 Multi-Timeframe Confluence Table")

def tf_signal(rsi, stochrsi_k, wr, macd, macd_sig, adx, pdi, mdi):
    bulls = 0; bears = 0
    if rsi and rsi < 50: bulls += 1
    if rsi and rsi > 50: bears += 1
    if stochrsi_k and stochrsi_k < 50: bulls += 1
    if stochrsi_k and stochrsi_k > 50: bears += 1
    if wr and wr < -50: bulls += 1
    if wr and wr > -50: bears += 1
    if macd and macd_sig and macd > macd_sig: bulls += 1
    if macd and macd_sig and macd < macd_sig: bears += 1
    if adx and pdi and mdi and adx > 20 and pdi > mdi: bulls += 1
    if adx and pdi and mdi and adx > 20 and mdi > pdi: bears += 1
    total = bulls + bears
    if total == 0: return "NEUTRAL", "#d29922"
    ratio = bulls / total
    if ratio >= 0.7: return "BULLISH", "#3fb950"
    if ratio <= 0.3: return "BEARISH", "#f85149"
    return "NEUTRAL", "#d29922"

d_sig, d_col = tf_signal(snap.get("d_rsi"), snap.get("d_stochrsi_k"),
    snap.get("d_wr"), snap.get("d_macd"), snap.get("d_macd_sig"),
    snap.get("d_adx"), snap.get("d_pdi"), snap.get("d_mdi"))
h_sig, h_col = tf_signal(snap.get("h_rsi"), snap.get("h_stochrsi_k"),
    snap.get("h_wr"), snap.get("h_macd"), snap.get("h_macd_sig"),
    snap.get("h_adx"), None, None)

d_macd_arrow = "▲" if (snap.get("d_macd") or 0) > (snap.get("d_macd_sig") or 0) else "▼"
h_macd_arrow = "▲" if (snap.get("h_macd") or 0) > (snap.get("h_macd_sig") or 0) else "▼"

table_html = ('<table class="conf-table">'
    '<tr><th>Timeframe</th><th>RSI(14)</th><th>StochRSI K</th><th>Williams %R</th><th>CCI</th><th>ADX</th><th>MACD</th><th>Consensus</th></tr>'
    '<tr><td><b>Daily</b></td>'
    '<td>'+str(snap.get("d_rsi") or "—")+'</td>'
    '<td>'+str(snap.get("d_stochrsi_k") or "—")+'</td>'
    '<td>'+str(snap.get("d_wr") or "—")+'</td>'
    '<td>'+str(snap.get("d_cci") or "—")+'</td>'
    '<td>'+str(snap.get("d_adx") or "—")+'</td>'
    '<td>'+d_macd_arrow+'</td>'
    '<td style="color:'+d_col+';font-weight:700">'+d_sig+'</td></tr>'
    '<tr><td><b>Hourly</b></td>'
    '<td>'+str(snap.get("h_rsi") or "—")+'</td>'
    '<td>'+str(snap.get("h_stochrsi_k") or "—")+'</td>'
    '<td>'+str(snap.get("h_wr") or "—")+'</td>'
    '<td>'+str(snap.get("h_cci") or "—")+'</td>'
    '<td>'+str(snap.get("h_adx") or "—")+'</td>'
    '<td>'+h_macd_arrow+'</td>'
    '<td style="color:'+h_col+';font-weight:700">'+h_sig+'</td></tr>'
    '</table>')
st.markdown(table_html, unsafe_allow_html=True)
st.divider()
# Key Levels + Ichimoku
left2, right2 = st.columns(2)
with left2:
    st.markdown("### 🔑 Key Price Levels")
    fib = snap["fibs"]
    st.markdown("**🔴 Resistance**")
    if snap["d_res"]:
        st.markdown(" ".join(badge(v, "res") for v in snap["d_res"]), unsafe_allow_html=True)
    else:
        st.caption("No daily resistance above price in 6-month range")
    st.markdown("**🟢 Support**")
    if snap["d_sup"]:
        st.markdown(" ".join(badge(v, "sup") for v in snap["d_sup"]), unsafe_allow_html=True)
    else:
        st.caption("Price below all 6-month pivots — watch Fib extensions")
    st.markdown("**🔵 Fibonacci (60-day swing)**")
    st.caption("Swing Low: $"+str(fib['swing_low'])+" → Swing High: $"+str(fib['swing_high']))
    fib_rows = [
        ("23.6%", fib["ret_236"]), ("38.2%", fib["ret_382"]),
        ("50.0%", fib["ret_500"]), ("61.8%", fib["ret_618"]), ("78.6%", fib["ret_786"]),
        ("Ext 127.2%", fib["ext_1272"]), ("Ext 161.8%", fib["ext_1618"]),
    ]
    for label, val in fib_rows:
        a, b = st.columns([1, 1.5])
        a.caption(label)
        b.markdown(badge(val, "fib"), unsafe_allow_html=True)

with right2:
    st.markdown("### 🌐 Ichimoku Cloud Levels")
    st.markdown("**Tenkan-sen (9)**")
    st.markdown(badge(snap.get("d_ichi_tenkan") or "—", "fib"), unsafe_allow_html=True)
    st.markdown("**Kijun-sen (26)**")
    st.markdown(badge(snap.get("d_ichi_kijun") or "—", "fib"), unsafe_allow_html=True)
    st.markdown("**Senkou A (Cloud top/bottom)**")
    st.markdown(badge(snap.get("d_ichi_sa") or "—", "sup"), unsafe_allow_html=True)
    st.markdown("**Senkou B (Cloud top/bottom)**")
    st.markdown(badge(snap.get("d_ichi_sb") or "—", "res"), unsafe_allow_html=True)
    st.markdown("**Supertrend Level**")
    st_dir_val = snap.get("d_st_dir")
    st_badge_kind = "sup" if st_dir_val == -1 else "res"
    st.markdown(badge(snap.get("d_supertrend") or "—", st_badge_kind), unsafe_allow_html=True)

st.divider()
# ══════════════════════════════════════════════════════════════════════════════
# LIVE DATA MONITORING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🟢 Live Data Monitoring System")
st.caption("Real-time 5-min tick · Smart Alerts · Session Stats · Volume Analysis · Momentum Gauges")

live = fetch_live_price()

if not live:
    st.warning("⚠️ Live tick data temporarily unavailable. Try refreshing in a few minutes.")
else:
    def mon_card(label, value, sub="", color="#f0f6fc"):
        return ('<div class="monitor-card">'
                '<div class="monitor-label">'+label+'</div>'
                '<div class="monitor-value" style="color:'+color+'">'+str(value)+'</div>'
                '<div class="monitor-sub">'+sub+'</div></div>')

    tick_color = "#3fb950" if live["chg_1m"] >= 0 else "#f85149"
    day_color  = "#3fb950" if live["chg_day"] >= 0 else "#f85149"
    tick_arrow = "▲" if live["chg_1m"] >= 0 else "▼"
    day_arrow  = "▲" if live["chg_day"] >= 0 else "▼"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.markdown(mon_card("⚡ LIVE PRICE", "$"+str(live['price']),
        tick_arrow+" "+str(live['chg_1m'])+" ("+str(live['chg_1m_pct'])+"%) 1min", tick_color), unsafe_allow_html=True)
    m2.markdown(mon_card("DAY CHANGE", day_arrow+" "+str(abs(live['chg_day_pct']))+"%",
        "vs open $"+str(live['session_open']), day_color), unsafe_allow_html=True)
    m3.markdown(mon_card("SESSION HIGH", "$"+str(live['session_high']),
        "+"+str(round(live['session_high']-live['session_open'],3))+" from open", "#3fb950"), unsafe_allow_html=True)
    m4.markdown(mon_card("SESSION LOW", "$"+str(live['session_low']),
        str(round(live['session_low']-live['session_open'],3))+" from open", "#f85149"), unsafe_allow_html=True)
    m5.markdown(mon_card("SESSION RANGE", str(live['session_range_pct'])+"%",
        "$"+str(live['session_low'])+" – $"+str(live['session_high']), "#d29922"), unsafe_allow_html=True)
    vc = "#f85149" if live['vol_ratio'] >= 2 else "#d29922" if live['vol_ratio'] >= 1.3 else "#8b949e"
    m6.markdown(mon_card("VOL RATIO", str(live['vol_ratio'])+"x",
        "last "+str(live['last_vol'])+" / avg "+str(live['avg_vol']), vc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_alerts, col_gauge, col_ticks = st.columns([1.4, 1, 1.2])

    with col_alerts:
        st.markdown("#### ** Alerts**")
        alerts = check_alerts(live, snap)
        dot_map = {"red": "alert-dot-red", "green": "alert-dot-green", "yellow": "alert-dot-yellow"}
        for al in alerts:
            dc = dot_map.get(al["level"], "alert-dot-yellow")
            st.markdown(
                '<div class="alert-row"><div class="alert-dot '+dc+'"></div>'
                '<span style="color:#c9d1d9">'+al["msg"]+'</span></div>',
                unsafe_allow_html=True)

    with col_gauge:
        st.markdown("#### **Momentum Gauges**")
        span = live["session_high"] - live["session_low"]
        pos = int(((live["price"] - live["session_low"]) / span * 100)) if span else 50
        bc = "#3fb950" if pos > 60 else "#f85149" if pos < 40 else "#d29922"
        st.markdown('<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">Session Position</div>'
            '<div class="gauge-bar-wrap"><div class="gauge-bar" style="width:'+str(pos)+'%;background:'+bc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+str(pos)+'% of range</div></div>', unsafe_allow_html=True)
        rv = snap.get("d_rsi") or 50
        rp = int(min(100, max(0, rv)))
        rc = "#f85149" if rv > 70 else "#3fb950" if rv < 30 else "#d29922"
        rl = "OVERBOUGHT" if rv > 70 else "OVERSOLD" if rv < 30 else "NEUTRAL"
        st.markdown('<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">RSI(14): '+str(rv)+'</div>'
            '<div class="gauge-bar-wrap"><div class="gauge-bar" style="width:'+str(rp)+'%;background:'+rc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+rl+'</div></div>', unsafe_allow_html=True)
        sk_live = snap.get("d_stochrsi_k") or 50
        skp = int(min(100, max(0, sk_live)))
        skc = "#f85149" if sk_live > 80 else "#3fb950" if sk_live < 20 else "#d29922"
        skl = "OVERBOUGHT" if sk_live > 80 else "OVERSOLD" if sk_live < 20 else "NEUTRAL"
        st.markdown('<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">StochRSI K: '+str(sk_live)+'</div>'
            '<div class="gauge-bar-wrap"><div class="gauge-bar" style="width:'+str(skp)+'%;background:'+skc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+skl+'</div></div>', unsafe_allow_html=True)
        vp = int(min(100, live["vol_ratio"] / 3 * 100))
        vgc = "#f85149" if live["vol_ratio"] >= 2 else "#d29922" if live["vol_ratio"] >= 1.3 else "#3fb950"
        vl = "SPIKE" if live["vol_ratio"] >= 2 else "ELEVATED" if live["vol_ratio"] >= 1.3 else "NORMAL"
        st.markdown('<div><div style="font-size:12px;color:#8b949e;margin-bottom:4px">Volume: '+str(live["vol_ratio"])+'x avg</div>'
            '<div class="gauge-bar-wrap"><div class="gauge-bar" style="width:'+str(vp)+'%;background:'+vgc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+vl+'</div></div>', unsafe_allow_html=True)

    with col_ticks:
        st.markdown("#### **Price Tick Log (5-min)**")
        ticks = live.get("ticks", [])[-15:]
        th = '<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:6px;max-height:220px;overflow-y:auto;">'
        for tk in reversed(ticks):
            tc = "#3fb950" if tk["dir"] == "U" else "#f85149"
            ta = "▲" if tk["dir"] == "U" else "▼"
            th += '<div class="tick-row"><span style="color:#8b949e">'+tk["t"]+'</span><span style="color:'+tc+';font-weight:700">'+ta+' $'+str(tk["p"])+'</span></div>'
        th += '</div>'
        st.markdown(th, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tick_prices = [tk["p"] for tk in live.get("ticks", [])]
    tick_times  = [tk["t"] for tk in live.get("ticks", [])]
    if len(tick_prices) >= 2:
        sp_color = "#3fb950" if tick_prices[-1] >= tick_prices[0] else "#f85149"
        fill_c = "rgba(63,185,80,0.08)" if sp_color == "#3fb950" else "rgba(248,81,73,0.08)"
        fig_spark = go.Figure(go.Scatter(
            x=tick_times, y=tick_prices, mode="lines+markers",
            line=dict(color=sp_color, width=2), marker=dict(size=4, color=sp_color),
            fill="tozeroy", fillcolor=fill_c,
            hovertemplate="<b>%{x}</b><br>$%{y:.3f}<extra></extra>",
        ))
        fig_spark.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=140, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#1c2128", tickformat="$.3f", tickfont=dict(size=9)),
        )
        st.caption("Last "+str(len(tick_prices))+" one-minute ticks")
        st.plotly_chart(fig_spark, use_container_width=True)

st.divider()
# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🤖 AI Trade Analysis & Targets")
st.caption("Powered by Claude "+AI_MODEL+" · All pro indicators included · Cached 5 min")

if not api_key:
    st.info("👈 Enter your Anthropic API key in the sidebar to unlock AI trade analysis.")
else:
    with st.spinner("Claude is analysing the market with all pro indicators…"):
        try:
            analysis = get_ai_analysis(snap["price"], api_key)
            st.markdown('<div class="analysis-box">'+analysis+'</div>', unsafe_allow_html=True)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                st.error("Invalid Anthropic API key.")
            else:
                st.error("API error: "+str(e))
        except Exception as e:
            st.error("AI analysis failed: "+str(e))

st.divider()
st.caption("⚠️ Educational / research use only — not financial advice. Silver futures carry significant risk.")

if auto_refresh:
    time.sleep(300)
    st.rerun()
