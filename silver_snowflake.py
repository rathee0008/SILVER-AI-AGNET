"""
Silver AI Trading Agent — Streamlit in Snowflake (SiS)
=======================================================
Deploy with:  setup_snowflake.sql

Differences from silver_web.py:
  • yfinance        → requests to Yahoo Finance v8 API
  • anthropic SDK   → requests to Anthropic REST API
  • TradingView JS  → plotly candlestick (SiS blocks custom HTML/JS)
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


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART — candlestick + volume + S/R + Fibonacci lines
# ══════════════════════════════════════════════════════════════════════════════
def make_chart(df: pd.DataFrame, support: list, resistance: list,
               fib: dict, current_price: float) -> go.Figure:
    df = df.copy()
    # Strip timezone for Plotly x-axis
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.02,
    )

    # ── Candlestick ─────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Silver (SI=F)",
        increasing_line_color="#3fb950", increasing_fillcolor="#3fb950",
        decreasing_line_color="#f85149", decreasing_fillcolor="#f85149",
        line_width=1,
    ), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_colors = [
        "#3fb95055" if float(c) >= float(o) else "#f8514955"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # ── EMA 20 ───────────────────────────────────────────────────────────────
    ema20 = df["EMA20"].dropna()
    fig.add_trace(go.Scatter(
        x=ema20.index, y=ema20.values,
        line=dict(color="#3fb950", width=1.2),
        name="EMA 20", hovertemplate="%{y:.3f}",
    ), row=1, col=1)

    # ── EMA 50 ───────────────────────────────────────────────────────────────
    ema50 = df["EMA50"].dropna()
    fig.add_trace(go.Scatter(
        x=ema50.index, y=ema50.values,
        line=dict(color="#f85149", width=1.2),
        name="EMA 50", hovertemplate="%{y:.3f}",
    ), row=1, col=1)

    # ── Current price ─────────────────────────────────────────────────────────
    fig.add_hline(
        y=current_price, line_color="#ffffff", line_width=2,
        annotation_text=f"  ▶ ${current_price}",
        annotation_position="right", annotation_font_color="#ffffff",
        row=1, col=1,
    )

    # ── Support levels ────────────────────────────────────────────────────────
    for i, s in enumerate(support):
        fig.add_hline(
            y=s, line_color="#3fb950", line_dash="dash", line_width=1,
            annotation_text=f"  S{i + 1}  ${s}",
            annotation_position="right", annotation_font_color="#3fb950",
            row=1, col=1,
        )

    # ── Resistance levels ─────────────────────────────────────────────────────
    for i, r in enumerate(resistance):
        fig.add_hline(
            y=r, line_color="#f85149", line_dash="dash", line_width=1,
            annotation_text=f"  R{i + 1}  ${r}",
            annotation_position="right", annotation_font_color="#f85149",
            row=1, col=1,
        )

    # ── Fibonacci levels ──────────────────────────────────────────────────────
    fib_cfg = [
        ("ret_236", "#a371f7", "Fib 23.6%"),
        ("ret_382", "#a371f7", "Fib 38.2%"),
        ("ret_500", "#58a6ff", "Fib 50%"),
        ("ret_618", "#a371f7", "Fib 61.8%"),
        ("ret_786", "#a371f7", "Fib 78.6%"),
        ("ext_1272", "#d29922", "Ext 127.2%"),
        ("ext_1618", "#d29922", "Ext 161.8%"),
    ]
    for key, color, label in fib_cfg:
        if key in fib:
            fig.add_hline(
                y=fib[key], line_color=color, line_dash="dot", line_width=1,
                annotation_text=f"  {label}  ${fib[key]}",
                annotation_position="left", annotation_font_color=color,
                row=1, col=1,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,
        height=640,
        margin=dict(l=10, r=130, t=20, b=10),
        legend=dict(orientation="h", y=1.02, yanchor="bottom",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1c2128", showgrid=True)
    fig.update_yaxes(gridcolor="#1c2128", showgrid=True)

    return fig


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
            "system": [{
                "type": "text",
                "text": (
                    "You are a senior COMEX silver futures trader with 20+ years of experience. "
                    "Give precise, actionable trade setups with exact price levels. All prices in USD/oz."
                ),
                "cache_control": {"type": "ephemeral"},
            }],
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
    st.markdown("**Chart legend**")
    st.markdown("🟢 Green line = EMA 20")
    st.markdown("🔴 Red line = EMA 50")
    st.markdown("🟢 Green dashed = Support")
    st.markdown("🔴 Red dashed = Resistance")
    st.markdown("🟣 Purple dotted = Fibonacci")
    st.markdown("🟡 Gold dotted = Fib Extensions")
    st.markdown("⚪ White = Current price")
    st.divider()
    if _IN_SNOWFLAKE:
        st.caption("🏔 Running inside Snowflake")
    st.caption("**Data:** Yahoo Finance\n\n**Chart:** Plotly\n\n**AI:** Claude Opus")
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
# PLOTLY CHART — Daily / Hourly tabs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Live Chart — Candlestick with Support, Resistance & Fibonacci")
st.caption("Green dashed = Support · Red dashed = Resistance · Purple/Gold dotted = Fibonacci · EMA 20 (green) / EMA 50 (red)")

tab_d, tab_h = st.tabs(["📅 Daily  (6 months)", "⏱ Hourly  (5 days)"])

with tab_d:
    fig_d = make_chart(
        snap["daily_df"].tail(120), snap["d_sup"], snap["d_res"],
        snap["fibs"], snap["price"],
    )
    st.plotly_chart(fig_d, use_container_width=True)

with tab_h:
    fig_h = make_chart(
        snap["hourly_df"], snap["h_sup"], snap["h_res"],
        snap["fibs"], snap["price"],
    )
    st.plotly_chart(fig_h, use_container_width=True)

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

# ══════════════════════════════════════════════════════════════════════════════
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
