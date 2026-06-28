"""
Silver AI Trading Agent — Streamlit in Snowflake (SiS)
=======================================================
Deploy with:  setup_snowflake.sql

Differences from silver_web.py:
  • yfinance        → requests to Yahoo Finance v8 API
  • anthropic SDK   → requests to Anthropic REST API
  • TradingView JS  → embedded via st.components.v1.html (full interactive chart)
  • API key         → Snowflake Secret (_snowflake module); falls back to sidebar input
"""

import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Snowflake secret (injected at runtime; ignored when running locally) ───────
try:
    import _snowflake                                          # noqa: F401 — SiS only
    _IN_SNOWFLAKE = True
except ImportError:
    _IN_SNOWFLAKE = False


def _get_secret() -> str | None:
    # 1. Snowflake Secret (when running inside SiS)
    if _IN_SNOWFLAKE:
        try:
            return _snowflake.get_generic_secret_string("anthropic_key")
        except Exception:
            pass
    # 2. Streamlit Community Cloud secret (ANTHROPIC_API_KEY = "..." in app secrets)
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
  [data-testid="stSidebar"]          { background: #161b22; }
  h1,h2,h3,h4                        { color: #f0f6fc; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; text-align: center;
  }
  .metric-label { color: #8b949e; font-size: 13px; margin-bottom: 4px; }
  .metric-value { color: #f0f6fc; font-size: 26px; font-weight: 700; }
  .metric-sub   { font-size: 13px; margin-top: 4px; }
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
</style>
""", unsafe_allow_html=True)

TICKER   = "SI=F"
AI_MODEL = "claude-opus-4-8"
YF_BASE  = "https://query1.finance.yahoo.com/v8/finance/chart"
YF_HDR   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA — Yahoo Finance via requests (no yfinance needed)
# ══════════════════════════════════════════════════════════════════════════════
def _yf_fetch(ticker: str, interval: str, range_: str) -> pd.DataFrame:
    r = requests.get(
        f"{YF_BASE}/{ticker}",
        params={"interval": interval, "range": range_, "includePrePost": "false"},
        headers=YF_HDR,
        timeout=20,
    )
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts     = result["timestamp"]
    q      = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q["open"], "High": q["high"], "Low": q["low"],
         "Close": q["close"], "Volume": q.get("volume", [0] * len(ts))},
        index=pd.to_datetime(ts, unit="s", utc=True),
    )
    return df.dropna(subset=["Close"])


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS — pure pandas
# ══════════════════════════════════════════════════════════════════════════════
def _ema(s, n):   return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d    = s.diff()
    gain = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, float("nan")))

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

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); c = df["Close"]
    df["EMA20"], df["EMA50"], df["EMA200"] = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    df["RSI"] = _rsi(c)
    df["MACD"], df["MACD_Sig"], df["MACD_Hist"] = _macd(c)
    df["BB_Up"], df["BB_Mid"], df["BB_Lo"] = _bollinger(c)
    df["ATR"] = _atr(df)
    return df


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
    daily  = enrich(_yf_fetch(TICKER, "1d",  "6mo"))
    hourly = enrich(_yf_fetch(TICKER, "1h",  "5d"))
    m15    = _yf_fetch(TICKER, "15m", "2d")

    price  = (m15 if not m15.empty else hourly)["Close"].iloc[-1]
    d, d1  = daily.iloc[-1], daily.iloc[-2]
    h      = hourly.iloc[-1]

    d_sup, d_res = pivot_levels(daily)
    h_sup, h_res = pivot_levels(hourly, n=3)
    fibs         = fibonacci(daily)

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
        d_rsi=f(d["RSI"], 2), d_ema20=f(d["EMA20"]), d_ema50=f(d["EMA50"]), d_ema200=f(d["EMA200"]),
        d_macd=f(d["MACD"], 4), d_macd_sig=f(d["MACD_Sig"], 4), d_macd_hist=f(d["MACD_Hist"], 4),
        d_bb_up=f(d["BB_Up"]), d_bb_mid=f(d["BB_Mid"]), d_bb_lo=f(d["BB_Lo"]), d_atr=f(d["ATR"]),
        h_rsi=f(h["RSI"], 2), h_ema20=f(h["EMA20"]),
        h_macd=f(h["MACD"], 4), h_macd_sig=f(h["MACD_Sig"], 4),
        d_sup=d_sup, d_res=d_res, h_sup=h_sup, h_res=h_res, fibs=fibs,
        daily_df=daily, hourly_df=hourly,
    )


# =============================================================================
# LIVE MONITORING
# =============================================================================
def fetch_live_price() -> dict:
    """Fetch live 1-min tick."""
    try:
        r = requests.get(
            f"{YF_BASE}/{TICKER}",
            params={"interval": "5m", "range": "1d", "includePrePost": "false"},
            headers=YF_HDR, timeout=10,
        )
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        ts  = res["timestamp"]
        closes  = [c for c in q["close"]  if c is not None]
        opens_l = [o for o in q["open"]   if o is not None]
        highs   = [h for h in q["high"]   if h is not None]
        lows    = [lv for lv in q["low"]  if lv is not None]
        volumes = [v for v in q.get("volume", [0]*len(ts)) if v is not None]
        if not closes:
            return {}
        price        = closes[-1]
        session_high = max(highs)    if highs    else price
        session_low  = min(lows)     if lows     else price
        session_open = opens_l[0]    if opens_l  else price
        avg_vol      = sum(volumes) / len(volumes) if volumes else 1
        last_vol     = volumes[-1]   if volumes  else 0
        prev_close   = closes[-2]    if len(closes) >= 2 else closes[0]
        chg_1m       = round(price - prev_close, 3)
        chg_1m_pct   = round((price - prev_close) / prev_close * 100, 3) if prev_close else 0
        chg_day      = round(price - session_open, 3)
        chg_day_pct  = round((price - session_open) / session_open * 100, 3) if session_open else 0
        ticks = []
        for i in range(max(0, len(closes) - 20), len(closes)):
            direction = "U" if i == 0 or closes[i] >= closes[i - 1] else "D"
            ticks.append({
                "t": datetime.utcfromtimestamp(ts[i]).strftime("%H:%M"),
                "p": round(closes[i], 3),
                "dir": direction,
            })
        return {
            "price": round(price, 3),
            "session_high": round(session_high, 3),
            "session_low":  round(session_low,  3),
            "session_open": round(session_open, 3),
            "chg_1m": chg_1m, "chg_1m_pct": chg_1m_pct,
            "chg_day": chg_day, "chg_day_pct": chg_day_pct,
            "last_vol": int(last_vol), "avg_vol": int(avg_vol),
            "vol_ratio": round(last_vol / avg_vol, 2) if avg_vol else 0,
            "ticks": ticks,
            "session_range_pct": round((session_high - session_low) / session_low * 100, 2) if session_low else 0,
        }
    except Exception:
        pass
    # Fallback: try 15-min interval
    try:
        r = requests.get(
            f"{YF_BASE}/{TICKER}",
            params={"interval": "15m", "range": "5d", "includePrePost": "false"},
            headers=YF_HDR, timeout=10,
        )
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        ts  = res["timestamp"]
        closes  = [c for c in q["close"]  if c is not None]
        opens_l = [o for o in q["open"]   if o is not None]
        highs   = [h for h in q["high"]   if h is not None]
        lows    = [lv for lv in q["low"]  if lv is not None]
        volumes = [v for v in q.get("volume", [0]*len(ts)) if v is not None]
        if not closes:
            return {}
        price        = closes[-1]
        session_high = max(highs[-20:])    if highs    else price
        session_low  = min(lows[-20:])     if lows     else price
        session_open = opens_l[-20]        if len(opens_l) >= 20 else opens_l[0] if opens_l else price
        avg_vol      = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 1
        last_vol     = volumes[-1]  if volumes else 0
        prev_close   = closes[-2]   if len(closes) >= 2 else closes[0]
        chg_1m       = round(price - prev_close, 3)
        chg_1m_pct   = round((price - prev_close) / prev_close * 100, 3) if prev_close else 0
        chg_day      = round(price - session_open, 3)
        chg_day_pct  = round((price - session_open) / session_open * 100, 3) if session_open else 0
        ticks = []
        for i in range(max(0, len(closes) - 20), len(closes)):
            direction = "U" if i == 0 or closes[i] >= closes[i - 1] else "D"
            ticks.append({"t": datetime.utcfromtimestamp(ts[i]).strftime("%H:%M"), "p": round(closes[i], 3), "dir": direction})
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


def check_alerts(live: dict, snap: dict) -> list:
    """Return triggered alert dicts."""
    alerts = []
    if not live:
        return [{"level": "yellow", "msg": "No live data"}]
    price = live["price"]
    if snap.get("d_rsi") and snap["d_rsi"] > 75:
        alerts.append({"level": "red",    "msg": f"RSI OVERBOUGHT {snap['d_rsi']} - reversal risk"})
    if snap.get("d_rsi") and snap["d_rsi"] < 25:
        alerts.append({"level": "green",  "msg": f"RSI OVERSOLD {snap['d_rsi']} - bounce opportunity"})
    for lvl in snap.get("d_sup", []):
        if abs(price - lvl) / lvl < 0.003:
            alerts.append({"level": "green", "msg": f"Near daily Support ${lvl} - watch bounce"})
    for lvl in snap.get("d_res", []):
        if abs(price - lvl) / lvl < 0.003:
            alerts.append({"level": "red",   "msg": f"Near daily Resistance ${lvl} - watch rejection"})
    if live["vol_ratio"] >= 2.5:
        alerts.append({"level": "yellow", "msg": f"Volume SPIKE {live['vol_ratio']}x avg"})
    ema20 = snap.get("d_ema20") or 0
    ema50 = snap.get("d_ema50") or 0
    if ema20 and ema50 and abs(ema20 - ema50) / ema50 < 0.002:
        alerts.append({"level": "yellow", "msg": "EMA 20/50 near crossover - trend change incoming"})
    for lvl in snap.get("d_res", []):
        if price > lvl and live.get("chg_1m_pct", 0) > 0.1:
            alerts.append({"level": "green", "msg": f"BREAKOUT above ${lvl}!"})
    for lvl in snap.get("d_sup", []):
        if price < lvl and live.get("chg_1m_pct", 0) < -0.1:
            alerts.append({"level": "red",   "msg": f"BREAKDOWN below ${lvl}!"})
    dm = snap.get("d_macd") or 0
    dms = snap.get("d_macd_sig") or 0
    if dm and dms and abs(dm - dms) < 0.001:
        alerts.append({"level": "yellow", "msg": "MACD crossing signal line"})
    if not alerts:
        alerts.append({"level": "green", "msg": "No critical alerts - market normal"})
    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW CHART — embedded Advanced Chart widget
# ══════════════════════════════════════════════════════════════════════════════
import streamlit.components.v1 as components


def make_tradingview_chart(interval: str, support: list, resistance: list,
                           fib: dict, current_price: float, height: int = 620) -> str:
    """Return HTML for embedded TradingView Advanced Real-Time Chart widget."""
    tv_interval_map = {
        "daily": "D",
        "hourly": "60",
        "15m": "15",
        "5m": "5",
    }
    tv_interval = tv_interval_map.get(interval, 'D')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; }}
  #tv_chart_container {{ width: 100%; height: {height}px; }}
</style>
</head>
<body>
<div id="tv_chart_container"></div>
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
  new TradingView.widget({{
    autosize: true,
    symbol: "COMEX:SI1!",
    interval: "{tv_interval}",
    timezone: "Etc/UTC",
    theme: "dark",
    style: "1",
    locale: "en",
    toolbar_bg: "#161b22",
    enable_publishing: false,
    hide_top_toolbar: false,
    hide_legend: false,
    save_image: true,
    allow_symbol_change: false,
    withdateranges: true,
    container_id: "tv_chart_container",
    studies: [
      "RSI@tv-basicstudies",
      "MACD@tv-basicstudies",
      "BB@tv-basicstudies",
      "MAExp@tv-basicstudies",
      "Volume@tv-basicstudies"
    ],
    overrides: {{
      "mainSeriesProperties.candleStyle.upColor": "#3fb950",
      "mainSeriesProperties.candleStyle.downColor": "#f85149",
      "mainSeriesProperties.candleStyle.wickUpColor": "#3fb950",
      "mainSeriesProperties.candleStyle.wickDownColor": "#f85149",
      "mainSeriesProperties.candleStyle.borderUpColor": "#3fb950",
      "mainSeriesProperties.candleStyle.borderDownColor": "#f85149",
      "paneProperties.background": "#0d1117",
      "paneProperties.backgroundType": "solid",
      "paneProperties.vertGridProperties.color": "#1c2128",
      "paneProperties.horzGridProperties.color": "#1c2128",
      "scalesProperties.textColor": "#8b949e"
    }},
    studies_overrides: {{
      "volume.volume.color.0": "rgba(248, 81, 73, 0.5)",
      "volume.volume.color.1": "rgba(63, 185, 80, 0.5)",
      "moving average exponential.length": 20,
      "moving average exponential.plot.color": "#3fb950",
      "moving average exponential.plot.linewidth": 2
    }}
  }});
</script>
</body>
</html>'''
    return html

# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS — Anthropic via requests (no SDK needed)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_ai_analysis(price_key: float, api_key: str) -> str:
    snap = load_snapshot()
    fib  = snap["fibs"]

    prompt = f"""Analyze this live COMEX silver (SI=F) market snapshot and provide a complete pro-trader setup.

═══ PRICE ACTION ═══
Current : ${snap['price']}/oz  │  24h Change: {snap['change_pct']:+}%
Today   : O {snap['d_open']}  H {snap['d_high']}  L {snap['d_low']}  │  Prev Close: {snap['prev_close']}
Week    : High {snap['week_high']} / Low {snap['week_low']}
Month   : High {snap['month_high']} / Low {snap['month_low']}

═══ DAILY INDICATORS ═══
RSI(14)         : {snap['d_rsi']}
EMA 20/50/200   : {snap['d_ema20']} / {snap['d_ema50']} / {snap['d_ema200']}
MACD / Signal   : {snap['d_macd']} / {snap['d_macd_sig']}  (Hist: {snap['d_macd_hist']})
Bollinger Bands : Lower {snap['d_bb_lo']}  │  Mid {snap['d_bb_mid']}  │  Upper {snap['d_bb_up']}
ATR(14)         : {snap['d_atr']}

═══ HOURLY INDICATORS ═══
RSI(14) : {snap['h_rsi']}  │  EMA20: {snap['h_ema20']}
MACD    : {snap['h_macd']}  │  Signal: {snap['h_macd_sig']}

═══ SUPPORT & RESISTANCE ═══
Daily  Resistance : {snap['d_res']}
Daily  Support    : {snap['d_sup']}
Hourly Resistance : {snap['h_res']}
Hourly Support    : {snap['h_sup']}

═══ FIBONACCI (60-day swing: ${fib['swing_low']} → ${fib['swing_high']}) ═══
Retracements : 23.6%={fib['ret_236']}  38.2%={fib['ret_382']}  50%={fib['ret_500']}  61.8%={fib['ret_618']}  78.6%={fib['ret_786']}
Extensions   : 127.2%={fib['ext_1272']}  161.8%={fib['ext_1618']}

Respond in EXACTLY this format:

## MARKET BIAS
[BULLISH / BEARISH / NEUTRAL] — concise paragraph explaining the overall technical picture.

## BUY SETUP
Entry Zone  : $X.XXX – $X.XXX
Target 1    : $X.XXX  (+X.X%)  ← conservative
Target 2    : $X.XXX  (+X.X%)  ← swing
Target 3    : $X.XXX  (+X.X%)  ← momentum
Stop Loss   : $X.XXX  (–X.X%)
R/R Ratio   : 1 : X.X

## SELL / SHORT SETUP
Entry Zone  : $X.XXX – $X.XXX
Target 1    : $X.XXX  (–X.X%)
Target 2    : $X.XXX  (–X.X%)
Target 3    : $X.XXX  (–X.X%)
Stop Loss   : $X.XXX  (+X.X%)
R/R Ratio   : 1 : X.X

## KEY LEVELS
List 3-5 critical price levels with a one-line reason each.

## PATTERN / SIGNAL
Any candlestick patterns, divergences, or momentum signals.

## RISK RATING
[LOW / MEDIUM / HIGH] — one sentence on volatility/risk context.

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
            "max_tokens": 2000,
            "system": (
                "You are a senior COMEX silver futures trader with 20+ years of experience. "
                "Give precise, actionable trade setups with exact price levels. All prices in USD/oz."
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
    return f'<span class="level-badge {cls}">${v}</span>'

def sig_tag(label, kind):
    cls = {"bull": "tag-bull", "bear": "tag-bear", "neut": "tag-neut"}.get(kind, "tag-neut")
    return f'<span class="tag {cls}">{label}</span>'

def metric_card(label, value, sub="", sub_class="neut"):
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub {sub_class}">{sub}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Silver AI Agent")
    st.caption("COMEX Silver Futures (SI=F)")
    st.divider()

    # On Snowflake the key comes from a Secret; show input only as fallback
    snowflake_key = _get_secret()
    if snowflake_key:
        st.success("🔐 API key loaded from Snowflake Secret")
        api_key = snowflake_key
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-…",
            help="On Snowflake this is injected from a Secret automatically.",
        )

    auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)
    run_btn      = st.button("🔄 Refresh Now", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**TradingView Chart**")
        st.markdown("🟢 Green candle = Bullish")
        st.markdown("🔴 Red candle = Bearish")
        st.markdown("🟢 S1/S2/S3 = Support levels")
        st.markdown("🔴 R1/R2/R3 = Resistance levels")
        st.markdown("🔵 Fib = Fibonacci retracements")
        st.markdown("🟡 EMA overlay available in chart")
        st.divider()
        if _IN_SNOWFLAKE:
            st.caption("🏔 Running inside Snowflake")
        st.caption("**Data:** Yahoo Finance\n\n**Chart:** TradingView\n\n**AI:** Claude Opus"))
    st.divider()
    st.warning("⚠️ Educational use only.\nNot financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 style='color:#f0f6fc;margin-bottom:0'>⚡ Silver AI Trading Agent</h1>",
            unsafe_allow_html=True)
st.caption(f"Live COMEX Silver Futures (SI=F) · Updated {datetime.now().strftime('%H:%M:%S')}")
st.divider()

if run_btn:
    st.cache_data.clear()

with st.spinner("Fetching live silver prices…"):
    try:
        snap = load_snapshot()
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

p, chg = snap["price"], snap["change_pct"] or 0
chg_color = "bull" if chg >= 0 else "bear"
arrow     = "▲" if chg >= 0 else "▼"

# ── price strip ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    ("💰 SILVER / OZ",  f"${p}",                  f"{arrow} {abs(chg)}%",                    chg_color),
    ("📅 Prev Close",   f"${snap['prev_close']}",  f"Day: {snap['d_low']} – {snap['d_high']}", "neut"),
    ("📊 Week Range",   f"{snap['week_low']}",     f"↑ {snap['week_high']}",                  "neut"),
    ("📆 Month Range",  f"{snap['month_low']}",    f"↑ {snap['month_high']}",                 "neut"),
    ("📐 ATR (vol)",    f"${snap['d_atr']}/oz",    "Daily volatility",                        "neut"),
]
for col, (label, val, sub, sc) in zip([c1, c2, c3, c4, c5], cards):
    col.markdown(metric_card(label, val, sub, sc), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW CHART — Daily / Hourly tabs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📈 Live TradingView Chart — COMEX Silver Futures (SI=F)")
st.caption("Full-featured TradingView chart with RSI · MACD · Bollinger Bands · Volume · EMA · Drawing tools · Replay mode")

tab_d, tab_h = st.tabs(["📅 Daily (6 months)", "⏱ Hourly (5 days)"])

with tab_d:
    html_d = make_tradingview_chart(
        interval="daily",
        support=snap["d_sup"], resistance=snap["d_res"],
        fib=snap["fibs"], current_price=snap["price"],
        height=660,
    )
    components.html(html_d, height=665, scrolling=False)
    # Key levels annotation below chart
    sup_str = '  '.join([f'🟢 S{i+1} ${v}' for i, v in enumerate(snap['d_sup'])])
    res_str = '  '.join([f'🔴 R{i+1} ${v}' for i, v in enumerate(snap['d_res'])])
    fib_str = f"🔵 Fib 38.2% ${snap['fibs']['ret_382']}  🔵 50% ${snap['fibs']['ret_500']}  🔵 61.8% ${snap['fibs']['ret_618']}"
    st.caption(f"{sup_str}  {res_str}  {fib_str}")

with tab_h:
    html_h = make_tradingview_chart(
        interval="hourly",
        support=snap["h_sup"], resistance=snap["h_res"],
        fib=snap["fibs"], current_price=snap["price"],
        height=660,
    )
    components.html(html_h, height=665, scrolling=False)
    sup_str = '  '.join([f'🟢 S{i+1} ${v}' for i, v in enumerate(snap['h_sup'])])
    res_str = '  '.join([f'🔴 R{i+1} ${v}' for i, v in enumerate(snap['h_res'])])
    fib_str = f"🔵 Fib 38.2% ${snap['fibs']['ret_382']}  🔵 50% ${snap['fibs']['ret_500']}  🔵 61.8% ${snap['fibs']['ret_618']}"
    st.caption(f"{sup_str}  {res_str}  {fib_str}")

st.divider()
# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS + KEY LEVELS
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns(2)

with left:
    st.markdown("### 📈 Technical Indicators")
    d_rsi = snap["d_rsi"] or 50
    h_rsi = snap["h_rsi"] or 50

    rows = []
    rsi_lbl  = "OVERBOUGHT ▼" if d_rsi > 70 else "OVERSOLD ▲" if d_rsi < 30 else "NEUTRAL"
    rsi_kind = "bear" if d_rsi > 70 else "bull" if d_rsi < 30 else "neut"
    rows.append(("RSI (14)", f"Daily {d_rsi} / Hourly {h_rsi}", sig_tag(rsi_lbl, rsi_kind)))

    ema_bull = p > (snap["d_ema20"] or 0) > (snap["d_ema50"] or 0)
    ema_bear = p < (snap["d_ema20"] or 0) < (snap["d_ema50"] or 0)
    ema_kind = "bull" if ema_bull else "bear" if ema_bear else "neut"
    rows.append(("EMA 20 / 50", f"{snap['d_ema20']} / {snap['d_ema50']}",
                 sig_tag("BULLISH ▲" if ema_bull else "BEARISH ▼" if ema_bear else "MIXED", ema_kind)))

    macd_bull = (snap["d_macd"] or 0) > (snap["d_macd_sig"] or 0)
    rows.append(("MACD (12/26/9)", f"Line {snap['d_macd']} / Sig {snap['d_macd_sig']}",
                 sig_tag("BULLISH ▲" if macd_bull else "BEARISH ▼", "bull" if macd_bull else "bear")))

    if   p > (snap["d_bb_up"] or 0): bb_l, bb_k = "ABOVE UPPER ▼", "bear"
    elif p < (snap["d_bb_lo"] or 0): bb_l, bb_k = "BELOW LOWER ▲", "bull"
    else:                             bb_l, bb_k = "WITHIN BANDS", "neut"
    rows.append(("Bollinger (20,2σ)", f"{snap['d_bb_lo']} – {snap['d_bb_up']}", sig_tag(bb_l, bb_k)))

    for name, val, sig in rows:
        r1, r2, r3 = st.columns([1.1, 1.6, 1.1])
        r1.markdown(f"**{name}**")
        r2.markdown(f"`{val}`")
        r3.markdown(sig, unsafe_allow_html=True)
        st.divider()

with right:
    st.markdown("### 🔑 Key Price Levels")
    fib = snap["fibs"]

    st.markdown("**🔴 Resistance**")
    if snap["d_res"]:
        st.markdown(" ".join(badge(v, "res") for v in snap["d_res"]), unsafe_allow_html=True)
    else:
        st.caption("No daily resistance above current price in 6-month range")

    st.markdown("**🟢 Support**")
    if snap["d_sup"]:
        st.markdown(" ".join(badge(v, "sup") for v in snap["d_sup"]), unsafe_allow_html=True)
    else:
        st.caption("Price below all 6-month pivots — watch Fib extensions")

    st.markdown("**🔵 Fibonacci (60-day swing)**")
    st.caption(f"Swing Low: ${fib['swing_low']}  →  Swing High: ${fib['swing_high']}")
    fib_rows = [
        ("23.6%", fib["ret_236"]), ("38.2%", fib["ret_382"]),
        ("50.0%", fib["ret_500"]), ("61.8%", fib["ret_618"]), ("78.6%", fib["ret_786"]),
        ("Ext 127.2%", fib["ext_1272"]), ("Ext 161.8%", fib["ext_1618"]),
    ]
    for label, val in fib_rows:
        a, b = st.columns([1, 1.5])
        a.caption(label)
        b.markdown(badge(val, "fib"), unsafe_allow_html=True)

st.divider()

# ======================================================================
# LIVE DATA MONITORING SYSTEM
# ======================================================================
st.markdown("### 🟢 Live Data Monitoring System")
st.caption("Real-time 5-min tick · Smart Alerts · Session Stats · Volume Analysis · Momentum Gauges")

live = fetch_live_price()

if not live:
    st.warning("⚠️ Live tick data temporarily unavailable (market may be closed or data delayed). Try refreshing in a few minutes.")
else:
    def mon_card(label, value, sub="", color="#f0f6fc"):
        return (f'<div class="monitor-card">'
                f'<div class="monitor-label">{label}</div>'
                f'<div class="monitor-value" style="color:{color}">{value}</div>'
                f'<div class="monitor-sub">{sub}</div></div>')

    tick_color = "#3fb950" if live["chg_1m"] >= 0 else "#f85149"
    day_color  = "#3fb950" if live["chg_day"] >= 0 else "#f85149"
    tick_arrow = "▲" if live["chg_1m"] >= 0 else "▼"
    day_arrow  = "▲" if live["chg_day"] >= 0 else "▼"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.markdown(mon_card("⚡ LIVE PRICE",   f"${live['price']}",        f"{tick_arrow} {live['chg_1m']:+.3f} ({live['chg_1m_pct']:+.3f}%) 1min", tick_color), unsafe_allow_html=True)
    m2.markdown(mon_card("DAY CHANGE",  f"{day_arrow} {abs(live['chg_day_pct']):.2f}%", f"vs open ${live['session_open']}", day_color), unsafe_allow_html=True)
    m3.markdown(mon_card("SESSION HIGH", f"${live['session_high']}", f"+{round(live['session_high']-live['session_open'],3)} from open", "#3fb950"), unsafe_allow_html=True)
    m4.markdown(mon_card("SESSION LOW",  f"${live['session_low']}",  f"{round(live['session_low']-live['session_open'],3)} from open",  "#f85149"), unsafe_allow_html=True)
    m5.markdown(mon_card("SESSION RANGE", f"{live['session_range_pct']:.2f}%", f"${live['session_low']} – ${live['session_high']}", "#d29922"), unsafe_allow_html=True)
    vc = "#f85149" if live['vol_ratio'] >= 2 else "#d29922" if live['vol_ratio'] >= 1.3 else "#8b949e"
    m6.markdown(mon_card("VOL RATIO",    f"{live['vol_ratio']}x",   f"last {live['last_vol']:,} / avg {live['avg_vol']:,}", vc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_alerts, col_gauge, col_ticks = st.columns([1.4, 1, 1.2])

    with col_alerts:
        st.markdown("#### ** Alerts**")
        alerts = check_alerts(live, snap)
        dot_map = {"red": "alert-dot-red", "green": "alert-dot-green", "yellow": "alert-dot-yellow"}
        for al in alerts:
            dc = dot_map.get(al["level"], "alert-dot-yellow")
            st.markdown(
                f'<div class="alert-row"><div class="alert-dot {dc}"></div>'
                f'<span style="color:#c9d1d9">{al["msg"]}</span></div>',
                unsafe_allow_html=True,
            )

    with col_gauge:
        st.markdown("#### **Momentum Gauges**")
        span = live["session_high"] - live["session_low"]
        pos  = int(((live["price"] - live["session_low"]) / span * 100)) if span else 50
        bc   = "#3fb950" if pos > 60 else "#f85149" if pos < 40 else "#d29922"
        st.markdown(f'<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">Session Position</div><div class="gauge-bar-wrap"><div class="gauge-bar" style="width:{pos}%;background:{bc}"></div></div><div style="font-size:11px;color:#8b949e;margin-top:3px">{pos}% of range</div></div>', unsafe_allow_html=True)
        rv = snap.get("d_rsi") or 50
        rp = int(min(100, max(0, rv)))
        rc = "#f85149" if rv > 70 else "#3fb950" if rv < 30 else "#d29922"
        rl = "OVERBOUGHT" if rv > 70 else "OVERSOLD" if rv < 30 else "NEUTRAL"
        st.markdown(f'<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">RSI(14): {rv}</div><div class="gauge-bar-wrap"><div class="gauge-bar" style="width:{rp}%;background:{rc}"></div></div><div style="font-size:11px;color:#8b949e;margin-top:3px">{rl}</div></div>', unsafe_allow_html=True)
        vp = int(min(100, live["vol_ratio"] / 3 * 100))
        vgc = "#f85149" if live["vol_ratio"] >= 2 else "#d29922" if live["vol_ratio"] >= 1.3 else "#3fb950"
        vl  = "SPIKE" if live["vol_ratio"] >= 2 else "ELEVATED" if live["vol_ratio"] >= 1.3 else "NORMAL"
        st.markdown(f'<div><div style="font-size:12px;color:#8b949e;margin-bottom:4px">Volume: {live["vol_ratio"]}x avg</div><div class="gauge-bar-wrap"><div class="gauge-bar" style="width:{vp}%;background:{vgc}"></div></div><div style="font-size:11px;color:#8b949e;margin-top:3px">{vl}</div></div>', unsafe_allow_html=True)

    with col_ticks:
        st.markdown("#### **Price Tick Log (5-min)**")
        ticks = live.get("ticks", [])[-15:]
        th = '<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:6px;max-height:220px;overflow-y:auto;">'
        for tk in reversed(ticks):
            tc = "#3fb950" if tk["dir"] == "U" else "#f85149"
            ta = "▲" if tk["dir"] == "U" else "▼"
            th += f'<div class="tick-row"><span style="color:#8b949e">{tk["t"]}</span><span style="color:{tc};font-weight:700">{ta} ${tk["p"]}</span></div>'
        th += '</div>'
        st.markdown(th, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sparkline
    tick_prices = [tk["p"] for tk in live.get("ticks", [])]
    tick_times  = [tk["t"] for tk in live.get("ticks", [])]
    if len(tick_prices) >= 2:
        sp_color = "#3fb950" if tick_prices[-1] >= tick_prices[0] else "#f85149"
        fill_c   = "rgba(63,185,80,0.08)" if sp_color == "#3fb950" else "rgba(248,81,73,0.08)"
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
        st.caption(f"Last {len(tick_prices)} one-minute ticks")
        st.plotly_chart(fig_spark, use_container_width=True)

st.divider()

# ======================================================================
# AI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🤖 AI Trade Analysis & Targets")
st.caption(f"Powered by Claude {AI_MODEL} · Cached 5 min")

if not api_key:
    st.info("👈 Enter your Anthropic API key in the sidebar to unlock AI trade analysis.")
else:
    with st.spinner("Claude is analysing the market…"):
        try:
            analysis = get_ai_analysis(snap["price"], api_key)
            st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                st.error("Invalid Anthropic API key.")
            else:
                st.error(f"API error: {e}")
        except Exception as e:
            st.error(f"AI analysis failed: {e}")

st.divider()
st.caption("⚠️ Educational / research use only — not financial advice. Silver futures carry significant risk.")

if auto_refresh:
    time.sleep(300)
    st.rerun()
