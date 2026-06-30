"""
Silver AI Trading Agent PRO v4 — ETF Volume Flow Edition
=========================================================
Features: Live COMEX Silver · Pro Technical Indicators · AI Analysis
5-Model Price Forecasting · Risk Analytics · Correlation
Gold/Silver Ratio · DXY Overlay · News Sentiment · Enhanced UI
ETF Volume Flow: SLV · SIVR · PSLV · AGQ · ZSL
"""

import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

try:
    import _snowflake
    _IN_SNOWFLAKE = True
except ImportError:
    _IN_SNOWFLAKE = False

def _get_secret():
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

st.set_page_config(
    page_title="Silver AI Trading Agent PRO v4",
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
    white-space:pre-wrap; font-family:"Segoe UI",sans-serif;
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
.pred-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d; border-radius: 10px;
    padding: 16px; margin: 6px 0; text-align: center;
}
.gauge-container {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:16px; text-align:center;
}
.news-card {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:12px 16px; margin:6px 0;
}
.ratio-card {
    background: linear-gradient(135deg, #1a1f2e, #161b22);
    border: 1px solid #3d4f7c; border-radius:10px; padding:16px; text-align:center;
}
.etf-vflow-card {
    background: linear-gradient(135deg, #1a2333, #161b22);
    border: 1px solid #2d4a6b; border-radius:10px; padding:16px; margin:6px 0;
}
.etf-stat {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:10px 14px; text-align:center;
}
</style>
""", unsafe_allow_html=True)

TICKER = "SI=F"
GOLD_TICKER = "GC=F"
DXY_TICKER = "DX-Y.NYB"
AI_MODEL = "claude-opus-4-5"
AI_MODEL_FAST = "claude-haiku-4-5"
YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
YF_BASE2 = "https://query2.finance.yahoo.com/v8/finance/chart"
YF_HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}
ETF_TICKERS = {
    "SLV": "iShares Silver Trust",
    "SIVR": "abrdn Physical Silver",
    "PSLV": "Sprott Physical Silver",
    "AGQ": "ProShares Ultra Silver (2x)",
    "ZSL": "ProShares UltraShort Silver (Bear)",
}

# ── Data Fetching ────────────────────────────────────────────
def _yf_fetch(ticker, interval, range_, _retry=3):
    urls = [YF_BASE, YF_BASE2]
    last_exc = None
    for attempt in range(_retry):
        base = urls[attempt % len(urls)]
        try:
            r = requests.get(
                f"{base}/{ticker}",
                params={"interval": interval, "range": range_,
                        "includePrePost": "false", "corsDomain": "finance.yahoo.com"},
                headers=YF_HDR, timeout=20,
            )
            r.raise_for_status()
            js = r.json()
            result = js["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            df = pd.DataFrame(
                {"Open": q["open"], "High": q["high"], "Low": q["low"],
                 "Close": q["close"], "Volume": q.get("volume", [0]*len(ts))},
                index=pd.to_datetime(ts, unit="s", utc=True),
            )
            df = df.dropna(subset=["Close"])
            if df.empty:
                raise ValueError("Empty DataFrame")
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < _retry - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"_yf_fetch failed: {last_exc}")

# ── Pro Indicators ──────────────────────────────────────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def _rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, float("nan")))

def _stoch_rsi(s, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3):
    rsi = _rsi(s, rsi_len)
    lo = rsi.rolling(stoch_len).min()
    hi = rsi.rolling(stoch_len).max()
    raw = (rsi - lo) / (hi - lo + 1e-10) * 100
    k = raw.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def _williams_r(df, n=14):
    hi = df["High"].rolling(n).max()
    lo = df["Low"].rolling(n).min()
    return (hi - df["Close"]) / (hi - lo + 1e-10) * -100

def _cci(df, n=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)

def _adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff(); dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr_n = tr.ewm(com=n-1, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=n-1,adjust=False).mean() / atr_n.replace(0,float("nan"))
    minus_di = 100 * minus_dm.ewm(com=n-1,adjust=False).mean() / atr_n.replace(0,float("nan"))
    dx = (plus_di-minus_di).abs() / (plus_di+minus_di+1e-10) * 100
    return dx.ewm(com=n-1,adjust=False).mean(), plus_di, minus_di

def _macd(s):
    line = _ema(s,12) - _ema(s,26); sig = _ema(line,9)
    return line, sig, line-sig

def _bollinger(s, n=20, k=2.0):
    mid = s.rolling(n).mean(); std = s.rolling(n).std()
    return mid+k*std, mid, mid-k*std

def _atr(df, n=14):
    h,l,c = df["High"],df["Low"],df["Close"]
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(com=n-1,adjust=False).mean()

def _obv(df):
    direction = df["Close"].diff().apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    return (df["Volume"]*direction).cumsum()

def _vwap(df):
    tp = (df["High"]+df["Low"]+df["Close"])/3
    return (tp*df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0,float("nan"))

def _ichimoku(df, t=9, k=26, s=52):
    tenkan = (df["High"].rolling(t).max()+df["Low"].rolling(t).min())/2
    kijun = (df["High"].rolling(k).max()+df["Low"].rolling(k).min())/2
    senkou_a = ((tenkan+kijun)/2).shift(k)
    senkou_b = ((df["High"].rolling(s).max()+df["Low"].rolling(s).min())/2).shift(k)
    chikou = df["Close"].shift(-k)
    return tenkan, kijun, senkou_a, senkou_b, chikou

def _supertrend(df, n=10, mult=3.0):
    atr_v = _atr(df,n).values
    hl2 = ((df["High"]+df["Low"])/2).values
    close = df["Close"].values
    ub = hl2 + mult*atr_v
    lb = hl2 - mult*atr_v
    supertrend = np.full(len(df), np.nan)
    direction = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        ub[i] = min(ub[i],ub[i-1]) if close[i-1] <= ub[i-1] else ub[i]
        lb[i] = max(lb[i],lb[i-1]) if close[i-1] >= lb[i-1] else lb[i]
        if i == 1: direction[i] = -1
        elif supertrend[i-1] == ub[i-1]: direction[i] = -1 if close[i]>ub[i] else 1
        else: direction[i] = 1 if close[i]<lb[i] else -1
        supertrend[i] = lb[i] if direction[i]==-1 else ub[i]
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def _detect_divergence(price, indicator, lookback=20):
    try:
        p = price.tail(lookback).values
        ind = indicator.tail(lookback).values
        price_hi = p[-1] > p[:-1].max()*0.995
        price_lo = p[-1] < p[:-1].min()*1.005
        ind_hi = ind[-1] > ind[:-1].max()*0.995
        ind_lo = ind[-1] < ind[:-1].min()*1.005
        if price_hi and not ind_hi: return "bearish"
        if price_lo and not ind_lo: return "bullish"
    except Exception: pass
    return "none"

def enrich(df):
    df = df.copy(); c = df["Close"]
    df["EMA20"],df["EMA50"],df["EMA200"] = _ema(c,20),_ema(c,50),_ema(c,200)
    df["RSI"] = _rsi(c)
    df["StochRSI_K"],df["StochRSI_D"] = _stoch_rsi(c)
    df["WilliamsR"] = _williams_r(df)
    df["CCI"] = _cci(df)
    df["ADX"],df["Plus_DI"],df["Minus_DI"] = _adx(df)
    df["MACD"],df["MACD_Sig"],df["MACD_Hist"] = _macd(c)
    df["BB_Up"],df["BB_Mid"],df["BB_Lo"] = _bollinger(c)
    df["ATR"] = _atr(df)
    df["OBV"] = _obv(df)
    df["VWAP"] = _vwap(df)
    (df["Ichi_Tenkan"],df["Ichi_Kijun"],
     df["Ichi_SenkouA"],df["Ichi_SenkouB"],df["Ichi_Chikou"]) = _ichimoku(df)
    df["Supertrend"],df["ST_Dir"] = _supertrend(df)
    return df

# ── Signal Scoring ──────────────────────────────────────────
def signal_score(snap):
    points = 0; max_pts = 0; breakdown = []
    def add(name, val, bull_cond, bear_cond, weight=1):
        nonlocal points, max_pts
        max_pts += weight
        if bull_cond:
            points += weight
            breakdown.append((name,"BULL","#3fb950",val))
        elif bear_cond:
            breakdown.append((name,"BEAR","#f85149",val))
        else:
            breakdown.append((name,"NEUT","#d29922",val))
    p = snap.get("price") or 0
    rsi = snap.get("d_rsi") or 50
    add("RSI(14)", str(rsi), rsi<50, rsi>50)
    sk = snap.get("d_stochrsi_k") or 50; sd_v = snap.get("d_stochrsi_d") or 50
    add("StochRSI K/D", str(sk)+"/"+str(sd_v),
        sk<20 or (sk>sd_v and sk<80), sk>80 or (sk<sd_v and sk>20))
    wr = snap.get("d_wr") or -50
    add("Williams %R", str(round(wr,1)), wr<-80, wr>-20)
    cci = snap.get("d_cci") or 0
    add("CCI(20)", str(round(cci,1)), cci<-100, cci>100)
    adx=snap.get("d_adx") or 0; pdi=snap.get("d_pdi") or 0; mdi=snap.get("d_mdi") or 0
    add("ADX/DMI","ADX="+str(round(adx,1)),adx>25 and pdi>mdi,adx>25 and mdi>pdi)
    macd=snap.get("d_macd") or 0; msig=snap.get("d_macd_sig") or 0
    add("MACD",str(round(macd,4)),macd>msig,macd<msig)
    e20=snap.get("d_ema20") or 0; e50=snap.get("d_ema50") or 0; e200=snap.get("d_ema200") or 0
    add("EMA Stack","Bull" if p>e20>e50>e200 else "Bear",
        p>e20>e50>e200, p<e20<e50<e200, weight=2)
    bb_up=snap.get("d_bb_up") or 0; bb_lo=snap.get("d_bb_lo") or 0
    add("Bollinger","Lo="+str(bb_lo), p<bb_lo, p>bb_up)
    vwap=snap.get("d_vwap") or 0
    if vwap: add("VWAP","$"+str(round(vwap,3)), p>vwap, p<vwap)
    sa=snap.get("d_ichi_sa") or 0; sb=snap.get("d_ichi_sb") or 0
    if sa and sb:
        cloud_top=max(sa,sb); cloud_bot=min(sa,sb)
        add("Ichimoku Cloud","A="+str(round(sa,3)), p>cloud_top, p<cloud_bot)
    st_dir=snap.get("d_st_dir")
    if st_dir is not None:
        add("Supertrend","BUY" if st_dir==-1 else "SELL", st_dir==-1, st_dir==1, weight=2)
    rsi_div=snap.get("d_rsi_div") or "none"
    macd_div=snap.get("d_macd_div") or "none"
    if rsi_div!="none": add("RSI Div",rsi_div.upper(), rsi_div=="bullish", rsi_div=="bearish")
    if macd_div!="none": add("MACD Div",macd_div.upper(), macd_div=="bullish", macd_div=="bearish")
    gs_ratio = snap.get("gs_ratio") or 0
    if gs_ratio > 0:
        if gs_ratio > 80: add("G/S Ratio","Ratio="+str(round(gs_ratio,1)), True, False)
        elif gs_ratio < 60: add("G/S Ratio","Ratio="+str(round(gs_ratio,1)), False, True)
    etf_bias = snap.get("etf_flow_bias") or "neutral"
    if etf_bias == "inflow": add("ETF Flow","Inflow", True, False, weight=2)
    elif etf_bias == "outflow": add("ETF Flow","Outflow", False, True, weight=2)
    score = round(points/max_pts*100) if max_pts else 50
    if score>=60: label,color="BULLISH","#3fb950"
    elif score<=40: label,color="BEARISH","#f85149"
    else: label,color="NEUTRAL","#d29922"
    return {"score":score,"label":label,"color":color,"breakdown":breakdown}

# ── Support/Resistance + Fibonacci ─────────────────────────────────
def pivot_levels(df, n=5, top=3):
    h,l = df["High"].values, df["Low"].values
    res,sup = [],[]
    for i in range(n, len(df)-n):
        if h[i]==max(h[i-n:i+n+1]): res.append(round(h[i],3))
        if l[i]==min(l[i-n:i+n+1]): sup.append(round(l[i],3))
    p = df["Close"].iloc[-1]
    return (sorted(set(s for s in sup if s<p),reverse=True)[:top],
            sorted(set(r for r in res if r>p))[:top])

def fibonacci(df):
    hi=df["High"].tail(60).max(); lo=df["Low"].tail(60).min(); d=hi-lo
    return {"swing_high":round(hi,3),"swing_low":round(lo,3),
            "ret_236":round(hi-0.236*d,3),"ret_382":round(hi-0.382*d,3),
            "ret_500":round(hi-0.500*d,3),"ret_618":round(hi-0.618*d,3),
            "ret_786":round(hi-0.786*d,3),
            "ext_1272":round(lo+1.272*d,3),"ext_1618":round(lo+1.618*d,3)}

# ── Gold/Silver Ratio & DXY ──────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_gs_ratio_and_dxy():
    result = {"gold_price": None, "dxy": None, "gs_ratio": None,
              "gold_daily": None, "dxy_daily": None}
    try:
        gold_df = _yf_fetch(GOLD_TICKER, "1d", "6mo")
        result["gold_price"] = float(gold_df["Close"].iloc[-1])
        result["gold_daily"] = gold_df
    except Exception:
        pass
    try:
        dxy_df = _yf_fetch(DXY_TICKER, "1d", "6mo")
        result["dxy"] = float(dxy_df["Close"].iloc[-1])
        result["dxy_daily"] = dxy_df
    except Exception:
        pass
    return result

# ── ETF Volume Flow ────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_etf_vflow():
    """Fetch daily OHLCV for silver ETFs and compute volume flow metrics."""
    results = {}
    for sym in ETF_TICKERS:
        try:
            df = _yf_fetch(sym, "1d", "3mo")
            df = df.copy()
            df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
            df["DollarVol"] = df["TP"] * df["Volume"]
            df["OBV"] = _obv(df)
            df["FlowSign"] = (df["Close"] >= df["Open"]).astype(int) * 2 - 1
            df["DollarFlow"] = df["DollarVol"] * df["FlowSign"]
            df["CumFlow"] = df["DollarFlow"].cumsum()
            results[sym] = df
        except Exception:
            pass
    if not results:
        return "neutral", {}, {}
    summary = {}
    bear_etfs = {"ZSL"}
    total_inflow = 0; total_outflow = 0
    for sym, df in results.items():
        last5 = df.tail(5)
        net_flow_5d = last5["DollarFlow"].sum()
        last_price = float(df["Close"].iloc[-1])
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].tail(20).mean())
        vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 0
        pct_chg = float((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6] * 100) if len(df) >= 6 else 0
        is_bear = sym in bear_etfs
        if net_flow_5d > 0 and not is_bear:
            total_inflow += abs(net_flow_5d)
        elif net_flow_5d > 0 and is_bear:
            total_outflow += abs(net_flow_5d)
        elif net_flow_5d < 0 and not is_bear:
            total_outflow += abs(net_flow_5d)
        elif net_flow_5d < 0 and is_bear:
            total_inflow += abs(net_flow_5d)
        summary[sym] = {
            "price": round(last_price, 2),
            "pct_chg_5d": round(pct_chg, 2),
            "net_flow_5d": int(net_flow_5d),
            "vol_ratio": vol_ratio,
            "is_bear": is_bear,
            "name": ETF_TICKERS[sym],
        }
    total = total_inflow + total_outflow
    bias = "neutral"
    if total > 0:
        inflow_pct = total_inflow / total * 100
        if inflow_pct >= 60: bias = "inflow"
        elif inflow_pct <= 40: bias = "outflow"
    return bias, summary, results

# ── ETF VFlow Charts ──────────────────────────────────────────────
def make_etf_vflow_chart(etf_dfs, selected_etfs=None):
    """3-panel chart: Cumulative Flow, Daily Dollar Volume, ETF Price."""
    if not etf_dfs:
        return None
    if selected_etfs is None:
        selected_etfs = [s for s in ETF_TICKERS if s in etf_dfs and s != "ZSL"]
    colors = {"SLV":"#c0c0c0","SIVR":"#58a6ff","PSLV":"#3fb950","AGQ":"#ffa726","ZSL":"#f85149"}
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.35, 0.25], vertical_spacing=0.04,
                        subplot_titles=("Cumulative Dollar Flow (3mo)",
                                        "Daily Dollar Volume (USD)",
                                        "ETF Price"))
    for sym in selected_etfs:
        if sym not in etf_dfs:
            continue
        df = etf_dfs[sym].copy()
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        color = colors.get(sym, "#c0c0c0")
        dash = "dash" if sym == "ZSL" else "solid"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["CumFlow"].values,
            name=sym, line=dict(color=color, width=2, dash=dash),
            hovertemplate=sym + " CumFlow: $%{y:,.0f}<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Bar(
            x=df.index, y=df["DollarVol"].values,
            name=sym+" Vol", marker_color=color, opacity=0.65, showlegend=False,
            hovertemplate=sym + ": $%{y:,.0f}<extra></extra>"),
            row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"].values,
            name=sym+" Price", line=dict(color=color, width=1.5, dash=dash),
            showlegend=False,
            hovertemplate=sym + ": $%{y:.2f}<extra></extra>"),
            row=3, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=640, margin=dict(l=10, r=60, t=40, b=10),
        legend=dict(orientation="h", y=1.03, font=dict(size=11, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.85)", bordercolor="#30363d", borderwidth=1),
        hovermode="x unified", barmode="stack")
    fig.update_xaxes(gridcolor="#1c2128", showgrid=True)
    fig.update_yaxes(gridcolor="#1c2128", showgrid=True)
    fig.update_yaxes(title_text="Cum Flow ($)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Vol ($)", tickformat="$,.0f", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", tickformat="$,.2f", row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, row=1, col=1)
    return fig

def make_etf_obv_chart(etf_dfs, selected_etfs=None):
    """Normalized OBV comparison across ETFs."""
    if not etf_dfs:
        return None
    if selected_etfs is None:
        selected_etfs = [s for s in ETF_TICKERS if s in etf_dfs]
    colors = {"SLV":"#c0c0c0","SIVR":"#58a6ff","PSLV":"#3fb950","AGQ":"#ffa726","ZSL":"#f85149"}
    fig = go.Figure()
    for sym in selected_etfs:
        if sym not in etf_dfs:
            continue
        df = etf_dfs[sym].copy()
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        obv = df["OBV"]
        obv_min, obv_max = obv.min(), obv.max()
        if obv_max > obv_min:
            obv_norm = (obv - obv_min) / (obv_max - obv_min) * 100
        else:
            obv_norm = obv * 0 + 50
        color = colors.get(sym, "#c0c0c0")
        dash = "dash" if sym == "ZSL" else "solid"
        fig.add_trace(go.Scatter(
            x=df.index, y=obv_norm.values,
            name=sym, line=dict(color=color, width=2, dash=dash),
            hovertemplate=sym + " OBV (norm): %{y:.1f}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=320, title="Normalized OBV Comparison (0=min, 100=max per ETF)",
        margin=dict(l=10, r=60, t=50, b=10),
        legend=dict(orientation="h", font=dict(size=11, color="#c9d1d9")),
        yaxis=dict(title="OBV (normalized)", gridcolor="#1c2128", range=[0,100]),
        xaxis=dict(gridcolor="#1c2128"),
        hovermode="x unified")
    return fig

# ── News Sentiment ────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_sentiment():
    headlines = []
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SI=F&region=US&lang=en-US",
        "https://www.kitco.com/rss/silver.xml",
    ]
    for url in feeds:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                import re
                titles = re.findall(r'<title><![CDATA[(.*?)]]></title>', r.text)
                if not titles:
                    titles = re.findall(r'<title>(.*?)</title>', r.text)
                for t in titles[1:6]:
                    clean = re.sub(r'<[^>]+>', '', t).strip()
                    if clean and len(clean) > 10:
                        headlines.append(clean)
        except Exception:
            pass
    bullish_kw = ["surge","rally","rise","gain","bull","up","high","strong","buy","support","breakout","gold"]
    bearish_kw = ["fall","drop","decline","bear","low","weak","sell","pressure","breakdown","sell-off","crash"]
    scores = []
    for h in headlines[:8]:
        hl = h.lower()
        bull = sum(1 for k in bullish_kw if k in hl)
        bear = sum(1 for k in bearish_kw if k in hl)
        if bull > bear: scores.append(1)
        elif bear > bull: scores.append(-1)
        else: scores.append(0)
    avg = sum(scores)/len(scores) if scores else 0
    return {"headlines": headlines[:8], "sentiment_score": round(avg,2), "count": len(headlines)}

# ── Snapshot (cached 5 min) ────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot():
    daily = enrich(_yf_fetch(TICKER,"1d","6mo"))
    hourly = enrich(_yf_fetch(TICKER,"1h","5d"))
    m15 = _yf_fetch(TICKER,"15m","2d")
    try: m5 = _yf_fetch(TICKER,"5m","1d")
    except Exception: m5 = pd.DataFrame()
    _src = next((f for f in [m5,m15,hourly,daily]
                 if not f.empty and not pd.isna(f["Close"].iloc[-1])), daily)
    price = float(_src["Close"].iloc[-1])
    d,d1 = daily.iloc[-1], daily.iloc[-2]
    h = hourly.iloc[-1]
    d_sup,d_res = pivot_levels(daily)
    h_sup,h_res = pivot_levels(hourly,n=3)
    fibs = fibonacci(daily)
    rsi_div = _detect_divergence(daily["Close"], daily["RSI"])
    macd_div = _detect_divergence(daily["Close"], daily["MACD"])
    gs_data = fetch_gs_ratio_and_dxy()
    gs_ratio = None
    if gs_data["gold_price"] and price > 0:
        gs_ratio = round(gs_data["gold_price"] / price, 2)
    # ETF flow bias
    try:
        etf_bias, _, _ = fetch_etf_vflow()
    except Exception:
        etf_bias = "neutral"
    def f(v,n=3):
        try: return round(float(v),n) if v is not None and not pd.isna(v) else None
        except: return None
    return dict(
        price=f(price), change_pct=f((price-d1["Close"])/d1["Close"]*100,2),
        d_close=f(d["Close"]), d_open=f(d["Open"]), d_high=f(d["High"]),
        d_low=f(d["Low"]), prev_close=f(d1["Close"]),
        week_high=f(daily["High"].tail(5).max()), week_low=f(daily["Low"].tail(5).min()),
        month_high=f(daily["High"].tail(21).max()), month_low=f(daily["Low"].tail(21).min()),
        d_rsi=f(d["RSI"],2), d_ema20=f(d["EMA20"]), d_ema50=f(d["EMA50"]), d_ema200=f(d["EMA200"]),
        d_macd=f(d["MACD"],4), d_macd_sig=f(d["MACD_Sig"],4), d_macd_hist=f(d["MACD_Hist"],4),
        d_bb_up=f(d["BB_Up"]), d_bb_mid=f(d["BB_Mid"]), d_bb_lo=f(d["BB_Lo"]), d_atr=f(d["ATR"]),
        d_stochrsi_k=f(d["StochRSI_K"],2), d_stochrsi_d=f(d["StochRSI_D"],2),
        d_wr=f(d["WilliamsR"],2), d_cci=f(d["CCI"],2),
        d_adx=f(d["ADX"],2), d_pdi=f(d["Plus_DI"],2), d_mdi=f(d["Minus_DI"],2),
        d_obv=f(d["OBV"],0), d_vwap=f(d["VWAP"],3),
        d_ichi_tenkan=f(d["Ichi_Tenkan"]), d_ichi_kijun=f(d["Ichi_Kijun"]),
        d_ichi_sa=f(d["Ichi_SenkouA"]), d_ichi_sb=f(d["Ichi_SenkouB"]),
        d_supertrend=f(d["Supertrend"]), d_st_dir=int(d["ST_Dir"]) if not pd.isna(d["ST_Dir"]) else None,
        d_rsi_div=rsi_div, d_macd_div=macd_div,
        h_rsi=f(h["RSI"],2), h_ema20=f(h["EMA20"]),
        h_macd=f(h["MACD"],4), h_macd_sig=f(h["MACD_Sig"],4),
        h_stochrsi_k=f(h["StochRSI_K"],2), h_adx=f(h["ADX"],2),
        h_wr=f(h["WilliamsR"],2), h_cci=f(h["CCI"],2), h_vwap=f(h["VWAP"],3),
        d_sup=d_sup, d_res=d_res, h_sup=h_sup, h_res=h_res, fibs=fibs,
        daily_df=daily, hourly_df=hourly,
        gold_price=gs_data.get("gold_price"), dxy=gs_data.get("dxy"),
        gs_ratio=gs_ratio, gold_daily=gs_data.get("gold_daily"),
        dxy_daily=gs_data.get("dxy_daily"),
        etf_flow_bias=etf_bias,
    )

# ── Price Predictor Functions ──────────────────────────────────────────
def run_forecasts(daily_df, forecast_days=7):
    close = daily_df["Close"].values
    n = len(close)
    x = np.arange(n)
    x_future = np.arange(n, n + forecast_days)
    future_dates = pd.date_range(
        daily_df.index[-1].tz_localize(None) + timedelta(days=1),
        periods=forecast_days, freq="B")
    lin_coef = np.polyfit(x, close, 1)
    lin_pred = np.polyval(lin_coef, x_future)
    poly_coef = np.polyfit(x, close, 3)
    poly_pred = np.polyval(poly_coef, x_future)
    ma20 = daily_df["BB_Mid"].values if "BB_Mid" in daily_df.columns else pd.Series(close).rolling(20).mean().values
    ma_slope = (ma20[-1] - ma20[-21]) / 20 if n > 21 else 0
    ma_pred = np.array([ma20[-1] + ma_slope*(i+1) for i in range(forecast_days)])
    alpha, beta = 0.3, 0.1
    level, trend_val = close[0], close[1]-close[0]
    for v in close[1:]:
        prev_l = level
        level = alpha*v + (1-alpha)*(level+trend_val)
        trend_val = beta*(level-prev_l) + (1-beta)*trend_val
    exp_pred = np.array([level + trend_val*(i+1) for i in range(forecast_days)])
    mom_period = min(10, n//4)
    momentum = (close[-1] - close[-mom_period]) / mom_period
    mom_pred = np.array([close[-1] + momentum*(i+1) for i in range(forecast_days)])
    ensemble = (lin_pred + poly_pred + ma_pred + exp_pred + mom_pred) / 5
    returns = pd.Series(close).pct_change().dropna()
    vol_daily = returns.std()
    sigma = vol_daily * close[-1]
    ci_upper = ensemble + 1.645 * sigma * np.sqrt(np.arange(1, forecast_days+1))
    ci_lower = ensemble - 1.645 * sigma * np.sqrt(np.arange(1, forecast_days+1))
    return {"future_dates":future_dates,"lin_pred":lin_pred,"poly_pred":poly_pred,
            "ma_pred":ma_pred,"exp_pred":exp_pred,"mom_pred":mom_pred,
            "ensemble":ensemble,"ci_upper":ci_upper,"ci_lower":ci_lower,
            "vol_daily":vol_daily,"close":close}

def get_correlation_data(period="1y"):
    tickers = {"Gold":"GC=F","USD Index":"DX-Y.NYB",
               "S&P 500":"%5EGSPC","Copper":"HG=F","Oil":"CL=F"}
    data = {}
    for name, sym in tickers.items():
        try:
            r = requests.get(
                f"{YF_BASE}/{sym}",
                params={"interval":"1d","range":period,"includePrePost":"false"},
                headers=YF_HDR, timeout=15)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            closes = res["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if closes:
                data[name] = pd.Series(closes).pct_change().dropna()
        except Exception:
            pass
    return data

# ── Live Monitoring ──────────────────────────────────────────────────
def fetch_live_price():
    for interval, range_ in [("5m","1d"),("15m","5d")]:
        try:
            r = requests.get(f"{YF_BASE}/{TICKER}",
                params={"interval":interval,"range":range_,"includePrePost":"false"},
                headers=YF_HDR, timeout=10)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            ts = res["timestamp"]
            closes = [c for c in q["close"] if c is not None]
            opens_l = [o for o in q["open"] if o is not None]
            highs = [h for h in q["high"] if h is not None]
            lows = [lv for lv in q["low"] if lv is not None]
            volumes = [v for v in q.get("volume",[0]*len(ts)) if v is not None]
            if not closes: continue
            price = closes[-1]
            session_high = max(highs) if highs else price
            session_low = min(lows) if lows else price
            session_open = opens_l[0] if opens_l else price
            avg_vol = sum(volumes)/len(volumes) if volumes else 1
            last_vol = volumes[-1] if volumes else 0
            prev_close = closes[-2] if len(closes)>=2 else closes[0]
            chg_1m = round(price-prev_close,3)
            chg_1m_pct = round((price-prev_close)/prev_close*100,3) if prev_close else 0
            chg_day = round(price-session_open,3)
            chg_day_pct = round((price-session_open)/session_open*100,3) if session_open else 0
            ticks = []
            for i in range(max(0,len(closes)-20), len(closes)):
                direction = "U" if i==0 or closes[i]>=closes[i-1] else "D"
                ticks.append({"t":datetime.utcfromtimestamp(ts[i]).strftime("%H:%M"),
                              "p":round(closes[i],3),"dir":direction})
            return {
                "price":round(price,3),"session_high":round(session_high,3),
                "session_low":round(session_low,3),"session_open":round(session_open,3),
                "chg_1m":chg_1m,"chg_1m_pct":chg_1m_pct,
                "chg_day":chg_day,"chg_day_pct":chg_day_pct,
                "last_vol":int(last_vol),"avg_vol":int(avg_vol),
                "vol_ratio":round(last_vol/avg_vol,2) if avg_vol else 0,
                "ticks":ticks,
                "session_range_pct":round((session_high-session_low)/session_low*100,2) if session_low else 0,
            }
        except Exception: pass
    return {}

def check_alerts(live, snap):
    alerts = []
    if not live: return [{"level":"yellow","msg":"No live data"}]
    price = live["price"]
    rsi = snap.get("d_rsi") or 50
    if rsi>75: alerts.append({"level":"red","msg":"RSI OVERBOUGHT "+str(rsi)+" - reversal risk"})
    if rsi<25: alerts.append({"level":"green","msg":"RSI OVERSOLD "+str(rsi)+" - bounce opportunity"})
    sk = snap.get("d_stochrsi_k") or 50
    if sk>85: alerts.append({"level":"red","msg":"StochRSI OVERBOUGHT "+str(sk)})
    if sk<15: alerts.append({"level":"green","msg":"StochRSI OVERSOLD "+str(sk)+" - reversal setup"})
    wr = snap.get("d_wr") or -50
    if wr>-10: alerts.append({"level":"red","msg":"Williams %R overbought ("+str(wr)+")"})
    if wr<-90: alerts.append({"level":"green","msg":"Williams %R oversold ("+str(wr)+")"})
    adx=snap.get("d_adx") or 0; pdi=snap.get("d_pdi") or 0; mdi=snap.get("d_mdi") or 0
    if adx>30 and pdi>mdi: alerts.append({"level":"green","msg":"Strong BULLISH trend ADX "+str(round(adx,1))})
    if adx>30 and mdi>pdi: alerts.append({"level":"red","msg":"Strong BEARISH trend ADX "+str(round(adx,1))})
    if adx<20: alerts.append({"level":"yellow","msg":"Weak trend (ADX "+str(round(adx,1))+") ranging market"})
    rsi_div=snap.get("d_rsi_div") or "none"
    if rsi_div=="bullish": alerts.append({"level":"green","msg":"BULLISH RSI Divergence detected"})
    if rsi_div=="bearish": alerts.append({"level":"red","msg":"BEARISH RSI Divergence detected"})
    macd_div=snap.get("d_macd_div") or "none"
    if macd_div=="bullish": alerts.append({"level":"green","msg":"BULLISH MACD Divergence"})
    if macd_div=="bearish": alerts.append({"level":"red","msg":"BEARISH MACD Divergence"})
    st_dir=snap.get("d_st_dir")
    if st_dir is not None:
        if st_dir==-1: alerts.append({"level":"green","msg":"Supertrend: BULLISH"})
        else: alerts.append({"level":"red","msg":"Supertrend: BEARISH"})
    sa=snap.get("d_ichi_sa") or 0; sb_v=snap.get("d_ichi_sb") or 0
    if sa and sb_v:
        ct=max(sa,sb_v); cb=min(sa,sb_v)
        if price>ct: alerts.append({"level":"green","msg":"Price ABOVE Ichimoku Cloud"})
        elif price<cb: alerts.append({"level":"red","msg":"Price BELOW Ichimoku Cloud"})
        else: alerts.append({"level":"yellow","msg":"Price INSIDE Ichimoku Cloud"})
    gs_ratio = snap.get("gs_ratio") or 0
    if gs_ratio > 85: alerts.append({"level":"green","msg":f"Gold/Silver Ratio HIGH ({gs_ratio:.1f}) — Silver historically CHEAP"})
    elif gs_ratio > 0 and gs_ratio < 55: alerts.append({"level":"yellow","msg":f"Gold/Silver Ratio LOW ({gs_ratio:.1f}) — Silver historically EXPENSIVE"})
    # ETF flow alerts
    etf_bias = snap.get("etf_flow_bias") or "neutral"
    if etf_bias == "inflow":
        alerts.append({"level":"green","msg":"ETF Volume Flow: NET INFLOW — Bullish accumulation signal"})
    elif etf_bias == "outflow":
        alerts.append({"level":"red","msg":"ETF Volume Flow: NET OUTFLOW — Bearish distribution signal"})
    for lvl in snap.get("d_sup",[]):
        if abs(price-lvl)/lvl<0.003: alerts.append({"level":"green","msg":"Near Support $"+str(lvl)})
    for lvl in snap.get("d_res",[]):
        if abs(price-lvl)/lvl<0.003: alerts.append({"level":"red","msg":"Near Resistance $"+str(lvl)})
    if live.get("vol_ratio",0)>=2.5: alerts.append({"level":"yellow","msg":"Volume SPIKE "+str(live["vol_ratio"])+"x avg"})
    if not alerts: alerts.append({"level":"green","msg":"No critical alerts — market normal"})
    return alerts

# ── Chart ────────────────────────────────────────────────────────────────────────────
def make_chart(df, support, resistance, fib, current_price, show_ichimoku=True, timeframe="Daily",
               gold_df=None, dxy_df=None, show_gold=False, show_dxy=False):
    df = df.copy()
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    rows = 4
    row_heights = [0.55,0.12,0.18,0.15]
    subplot_titles = ("","Volume","RSI / StochRSI","MACD")
    fig = make_subplots(rows=rows,cols=1,shared_xaxes=True,
                        row_heights=row_heights,vertical_spacing=0.02,
                        subplot_titles=subplot_titles)
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Silver (SI=F)",
        increasing_line_color="#26a69a",increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350",decreasing_fillcolor="#ef5350",line_width=1),row=1,col=1)
    if show_ichimoku and "Ichi_SenkouA" in df.columns:
        sa=df["Ichi_SenkouA"].dropna(); sb_v=df["Ichi_SenkouB"].dropna()
        cidx=sa.index.intersection(sb_v.index)
        if len(cidx)>0:
            sa_c=sa.loc[cidx]; sb_c=sb_v.loc[cidx]
            bull=(sa_c>=sb_c).mean()>0.5
            fill_c="rgba(38,166,154,0.10)" if bull else "rgba(239,83,80,0.10)"
            fig.add_trace(go.Scatter(x=sa_c.index,y=sa_c.values,name="Senkou A",
                line=dict(color="rgba(38,166,154,0.5)",width=1)),row=1,col=1)
            fig.add_trace(go.Scatter(x=sb_c.index,y=sb_c.values,name="Senkou B",
                line=dict(color="rgba(239,83,80,0.5)",width=1),fill="tonexty",fillcolor=fill_c),row=1,col=1)
    if "Supertrend" in df.columns and "ST_Dir" in df.columns:
        st_bull=df[df["ST_Dir"]==-1]["Supertrend"]
        st_bear=df[df["ST_Dir"]==1]["Supertrend"]
        if not st_bull.empty:
            fig.add_trace(go.Scatter(x=st_bull.index,y=st_bull.values,mode="markers",
                marker=dict(color="#26a69a",size=4,symbol="circle"),name="Supertrend BUY"),row=1,col=1)
        if not st_bear.empty:
            fig.add_trace(go.Scatter(x=st_bear.index,y=st_bear.values,mode="markers",
                marker=dict(color="#ef5350",size=4,symbol="circle"),name="Supertrend SELL"),row=1,col=1)
    for col_,color,name,dash in [("EMA20","#26a69a","EMA 20","solid"),
                                  ("EMA50","#ef5350","EMA 50","solid"),("EMA200","#ffa726","EMA 200","dash")]:
        if col_ in df.columns:
            s=df[col_].dropna()
            fig.add_trace(go.Scatter(x=s.index,y=s.values,
                line=dict(color=color,width=1.5,dash=dash),name=name),row=1,col=1)
    if "VWAP" in df.columns:
        vwap_s=df["VWAP"].dropna()
        fig.add_trace(go.Scatter(x=vwap_s.index,y=vwap_s.values,
            line=dict(color="#42a5f5",width=1.5,dash="dot"),name="VWAP"),row=1,col=1)
    fig.add_hline(y=current_price,line_color="rgba(255,255,255,0.85)",line_width=1.5,
        annotation_text=" Price $"+str(current_price),annotation_position="right",
        annotation_font_color="#ffffff",annotation_font_size=11,row=1,col=1)
    for i,s in enumerate(support):
        fig.add_hline(y=s,line_color="#26a69a",line_dash="dash",line_width=1,
            annotation_text=" S"+str(i+1)+" $"+str(s),annotation_position="right",
            annotation_font_color="#26a69a",annotation_font_size=10,row=1,col=1)
    for i,r in enumerate(resistance):
        fig.add_hline(y=r,line_color="#ef5350",line_dash="dash",line_width=1,
            annotation_text=" R"+str(i+1)+" $"+str(r),annotation_position="right",
            annotation_font_color="#ef5350",annotation_font_size=10,row=1,col=1)
    fib_cfg=[("ret_236","#ab47bc","Fib 23.6%"),("ret_382","#ab47bc","Fib 38.2%"),
             ("ret_500","#42a5f5","Fib 50%"),("ret_618","#ab47bc","Fib 61.8%"),
             ("ret_786","#ab47bc","Fib 78.6%"),
             ("ext_1272","#ffa726","Ext 127.2%"),("ext_1618","#ffa726","Ext 161.8%")]
    for key,color,label in fib_cfg:
        if key in fib:
            fig.add_hline(y=fib[key],line_color=color,line_dash="dot",line_width=0.8,
                annotation_text=" "+label+" $"+str(fib[key]),annotation_position="left",
                annotation_font_color=color,annotation_font_size=9,row=1,col=1)
    vol_colors=["rgba(38,166,154,0.5)" if float(c)>=float(o) else "rgba(239,83,80,0.5)"
                for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",
        marker_color=vol_colors,showlegend=False),row=2,col=1)
    vol_ma=df["Volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=vol_ma.index,y=vol_ma.values,
        line=dict(color="#ffa726",width=1,dash="dot"),name="Vol MA20",showlegend=False),row=2,col=1)
    if "RSI" in df.columns:
        rsi_s=df["RSI"].dropna()
        fig.add_trace(go.Scatter(x=rsi_s.index,y=rsi_s.values,
            line=dict(color="#42a5f5",width=1.5),name="RSI(14)"),row=3,col=1)
    if "StochRSI_K" in df.columns:
        sk=df["StochRSI_K"].dropna(); sd_v=df["StochRSI_D"].dropna()
        fig.add_trace(go.Scatter(x=sk.index,y=sk.values,
            line=dict(color="#ffa726",width=1),name="StochRSI K"),row=3,col=1)
        fig.add_trace(go.Scatter(x=sd_v.index,y=sd_v.values,
            line=dict(color="#ef5350",width=1,dash="dot"),name="StochRSI D"),row=3,col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(239,83,80,0.07)",line_width=0,row=3,col=1)
    fig.add_hrect(y0=0,y1=30,fillcolor="rgba(38,166,154,0.07)",line_width=0,row=3,col=1)
    fig.add_hline(y=70,line_color="rgba(239,83,80,0.5)",line_dash="dot",line_width=1,row=3,col=1)
    fig.add_hline(y=30,line_color="rgba(38,166,154,0.5)",line_dash="dot",line_width=1,row=3,col=1)
    if "MACD" in df.columns:
        macd_s=df["MACD"].dropna(); sig_s=df["MACD_Sig"].dropna(); hist_s=df["MACD_Hist"].dropna()
        hist_colors=["rgba(38,166,154,0.6)" if v>=0 else "rgba(239,83,80,0.6)" for v in hist_s.values]
        fig.add_trace(go.Bar(x=hist_s.index,y=hist_s.values,name="MACD Hist",
            marker_color=hist_colors,showlegend=False),row=4,col=1)
        fig.add_trace(go.Scatter(x=macd_s.index,y=macd_s.values,
            line=dict(color="#42a5f5",width=1.5),name="MACD"),row=4,col=1)
        fig.add_trace(go.Scatter(x=sig_s.index,y=sig_s.values,
            line=dict(color="#ffa726",width=1,dash="dot"),name="Signal"),row=4,col=1)
        fig.add_hline(y=0,line_color="rgba(255,255,255,0.2)",line_width=1,row=4,col=1)
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,height=840,margin=dict(l=10,r=150,t=25,b=10),
        legend=dict(orientation="h",y=1.02,yanchor="bottom",font=dict(size=10,color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.85)",bordercolor="#30363d",borderwidth=1),
        hovermode="x unified",hoverlabel=dict(bgcolor="#161b22",bordercolor="#30363d",font_color="#c9d1d9"),
        spikedistance=-1)
    fig.update_xaxes(gridcolor="#1c2128",showgrid=True,zeroline=False,
                     showspikes=True,spikecolor="#42a5f5",spikethickness=1,spikedash="dot")
    fig.update_yaxes(gridcolor="#1c2128",showgrid=True,zeroline=False)
    fig.update_yaxes(title_text="USD/oz",tickformat="$,.2f",row=1,col=1)
    fig.update_yaxes(title_text="Vol",row=2,col=1)
    fig.update_yaxes(title_text="RSI",range=[0,100],row=3,col=1)
    fig.update_yaxes(title_text="MACD",row=4,col=1)
    return fig

def make_gs_ratio_chart(silver_df, gold_df):
    silver_df = silver_df.copy()
    gold_df = gold_df.copy()
    if silver_df.index.tzinfo is not None:
        silver_df.index = silver_df.index.tz_convert("UTC").tz_localize(None)
    if gold_df.index.tzinfo is not None:
        gold_df.index = gold_df.index.tz_convert("UTC").tz_localize(None)
    common = silver_df.index.intersection(gold_df.index)
    if len(common) < 5:
        return None
    ratio = gold_df.loc[common, "Close"] / silver_df.loc[common, "Close"]
    ratio_ma20 = ratio.rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values,
        line=dict(color="#c0c0c0", width=2), name="G/S Ratio"))
    fig.add_trace(go.Scatter(x=ratio_ma20.index, y=ratio_ma20.values,
        line=dict(color="#ffa726", width=1.5, dash="dot"), name="MA20"))
    fig.add_hrect(y0=80, y1=120, fillcolor="rgba(63,185,80,0.08)", line_width=0,
        annotation_text="Silver CHEAP zone (>80)", annotation_font_color="#3fb950",
        annotation_font_size=10)
    fig.add_hrect(y0=0, y1=55, fillcolor="rgba(248,81,73,0.08)", line_width=0,
        annotation_text="Silver EXPENSIVE zone (<55)", annotation_font_color="#f85149",
        annotation_font_size=10)
    fig.add_hline(y=80, line_color="#3fb950", line_dash="dash", line_width=1)
    fig.add_hline(y=55, line_color="#f85149", line_dash="dash", line_width=1)
    current_ratio = ratio.iloc[-1]
    fig.add_hline(y=current_ratio, line_color="white", line_width=1.5,
        annotation_text=f" Current: {current_ratio:.1f}", annotation_font_color="white")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, title="Gold/Silver Ratio (6 months)",
        legend=dict(font=dict(size=11, color="#c9d1d9")),
        yaxis_gridcolor="#1c2128", hovermode="x unified",
        margin=dict(l=10, r=80, t=40, b=10))
    return fig

def make_dxy_overlay_chart(silver_df, dxy_df):
    silver_df = silver_df.copy()
    dxy_df = dxy_df.copy()
    if silver_df.index.tzinfo is not None:
        silver_df.index = silver_df.index.tz_convert("UTC").tz_localize(None)
    if dxy_df.index.tzinfo is not None:
        dxy_df.index = dxy_df.index.tz_convert("UTC").tz_localize(None)
    common = silver_df.index.intersection(dxy_df.index)
    if len(common) < 5:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=common, y=silver_df.loc[common, "Close"].values,
        line=dict(color="#c0c0c0", width=2), name="Silver (SI=F)"),
        secondary_y=False)
    fig.add_trace(go.Scatter(x=common, y=dxy_df.loc[common, "Close"].values,
        line=dict(color="#ffa726", width=1.5, dash="dot"), name="DXY (USD Index)"),
        secondary_y=True)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, title="Silver vs USD Index (Inverse Correlation)",
        legend=dict(font=dict(size=11, color="#c9d1d9")),
        hovermode="x unified", margin=dict(l=10, r=80, t=40, b=10))
    fig.update_yaxes(title_text="Silver (USD/oz)", gridcolor="#1c2128", secondary_y=False)
    fig.update_yaxes(title_text="DXY", secondary_y=True)
    return fig

def tradingview_widget(symbol="OANDA:XAGUSD",theme="dark",height=580,interval="60"):
    studies=["RSI@tv-basicstudies","MASimple@tv-basicstudies","MACD@tv-basicstudies",
             "IchimokuCloud@tv-basicstudies","SuperTrend@tv-basicstudies"]
    studies_json="["+",".join(chr(34)+s+chr(34) for s in studies)+"]"
    overrides={
        "mainSeriesProperties.candleStyle.upColor":"#26a69a",
        "mainSeriesProperties.candleStyle.downColor":"#ef5350",
        "mainSeriesProperties.candleStyle.borderUpColor":"#26a69a",
        "mainSeriesProperties.candleStyle.borderDownColor":"#ef5350",
        "paneProperties.background":"#0d1117",
        "paneProperties.backgroundType":"solid",
        "paneProperties.vertGridProperties.color":"#1c2128",
        "paneProperties.horzGridProperties.color":"#1c2128",
        "scalesProperties.textColor":"#c9d1d9",
    }
    ov_json = json.dumps(overrides)
    cid="tv_silver_chart"
    return f"""<div class="tradingview-widget-container" style="height:{height}px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #30363d">
<div id="{cid}" style="height:100%;width:100%"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
new TradingView.widget({{
"width":"100%","height":{height},"symbol":"{symbol}","interval":"{interval}",
"timezone":"Etc/UTC","theme":"{theme}","style":"1","locale":"en",
"toolbar_bg":"#0d1117","enable_publishing":false,"hide_top_toolbar":false,
"hide_legend":false,"hide_side_toolbar":true,"allow_symbol_change":false,
"save_image":true,"container_id":"{cid}",
"studies":{studies_json},"overrides":{ov_json}
}});
</script></div>"""

def tradingview_ticker_tape():
    symbols=[{"proName":"TVC:SILVER","title":"Silver Futures"},
             {"proName":"OANDA:XAGUSD","title":"XAG/USD"},
             {"proName":"TVC:GOLD","title":"Gold"},
             {"proName":"TVC:DXY","title":"DXY"},
             {"proName":"TVC:US10Y","title":"10Y Yield"}]
    config=json.dumps({"symbols":symbols,"showSymbolLogo":True,
                       "isTransparent":True,"displayMode":"adaptive","colorTheme":"dark","locale":"en"})
    return f"""<div class="tradingview-widget-container" style="margin-bottom:12px">
<div class="tradingview-widget-container__widget"></div>
<script type="text/javascript"
src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
{config}
</script></div>"""

@st.cache_data(ttl=300, show_spinner=False)
def get_ai_analysis(price_key, api_key, gs_ratio=None, dxy=None, sentiment_score=None, etf_bias=None):
    snap = load_snapshot()
    fib = snap["fibs"]
    extra_context = ""
    if gs_ratio:
        extra_context += f"\nGold/Silver Ratio: {gs_ratio:.1f} ({'Silver historically CHEAP - contrarian BUY signal' if gs_ratio>80 else 'Normal range'})"
    if dxy:
        extra_context += f"\nDXY (USD Index): {dxy:.2f} (Silver has inverse correlation with USD)"
    if sentiment_score is not None:
        sent_label = "Bullish" if sentiment_score > 0.2 else ("Bearish" if sentiment_score < -0.2 else "Neutral")
        extra_context += f"\nNews Sentiment Score: {sentiment_score:.2f} ({sent_label})"
    if etf_bias:
        etf_label = {"inflow":"Bullish accumulation","outflow":"Bearish distribution","neutral":"Mixed/neutral"}.get(etf_bias,"Mixed/neutral")
        extra_context += f"\nETF Volume Flow Bias: {etf_bias.upper()} ({etf_label})"
    prompt = ("Analyze this live COMEX silver (SI=F) market snapshot and provide a complete pro-trader setup.\n\n"
              "=== PRICE ACTION ===\n"
              "Current: $"+str(snap["price"])+"/oz | 24h Change: "+str(snap["change_pct"])+"%\n"
              "Today: O "+str(snap["d_open"])+" H "+str(snap["d_high"])+" L "+str(snap["d_low"])+" | Prev: "+str(snap["prev_close"])+"\n"
              "Week: High "+str(snap["week_high"])+" / Low "+str(snap["week_low"])+"\n"
              "Month: High "+str(snap["month_high"])+" / Low "+str(snap["month_low"])+"\n\n"
              "=== DAILY INDICATORS ===\n"
              "RSI(14): "+str(snap["d_rsi"])+"\n"
              "StochRSI K/D: "+str(snap["d_stochrsi_k"])+" / "+str(snap["d_stochrsi_d"])+"\n"
              "Williams %R: "+str(snap["d_wr"])+"\n"
              "CCI(20): "+str(snap["d_cci"])+"\n"
              "ADX/+DI/-DI: "+str(snap["d_adx"])+" / "+str(snap["d_pdi"])+" / "+str(snap["d_mdi"])+"\n"
              "EMA 20/50/200: "+str(snap["d_ema20"])+" / "+str(snap["d_ema50"])+" / "+str(snap["d_ema200"])+"\n"
              "MACD/Signal: "+str(snap["d_macd"])+" / "+str(snap["d_macd_sig"])+"\n"
              "Bollinger: "+str(snap["d_bb_lo"])+" - "+str(snap["d_bb_up"])+"\n"
              "ATR(14): "+str(snap["d_atr"])+" | VWAP: "+str(snap["d_vwap"])+"\n"
              "Supertrend: "+str(snap["d_supertrend"])+" ("+("BUY" if snap["d_st_dir"]==-1 else "SELL")+")\n"
              "RSI Div: "+str(snap["d_rsi_div"])+" | MACD Div: "+str(snap["d_macd_div"])+"\n\n"
              "=== FIBONACCI ===\n"
              "Swing: $"+str(fib["swing_low"])+" to $"+str(fib["swing_high"])+"\n"
              "23.6%="+str(fib["ret_236"])+" 38.2%="+str(fib["ret_382"])+" 50%="+str(fib["ret_500"])+" 61.8%="+str(fib["ret_618"])+"\n\n"
              + ("=== MACRO CONTEXT ===" + extra_context + "\n\n" if extra_context else "") +
              "Respond with: MARKET BIAS, BUY SETUP (entry/targets/SL/RR), SELL SETUP, KEY LEVELS, PATTERN/SIGNAL, RISK RATING, TRADER NOTE.")
    resp = requests.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
        json={"model":AI_MODEL,"max_tokens":2500,
              "system":"You are a senior COMEX silver futures trader. Give precise actionable setups. All prices USD/oz.",
              "messages":[{"role":"user","content":prompt}]},
        timeout=60)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(v, kind="fib"):
    cls={"sup":"sup-badge","res":"res-badge","fib":"fib-badge"}.get(kind,"fib-badge")
    return '<span class="level-badge '+cls+'">$'+str(v)+'</span>'

def sig_tag(label, kind):
    cls={"bull":"tag-bull","bear":"tag-bear","neut":"tag-neut"}.get(kind,"tag-neut")
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

def gauge_html(value, max_val, label, color, suffix=""):
    pct = min(100, max(0, int(value/max_val*100)))
    return (f'<div style="margin-bottom:12px">'
            f'<div style="font-size:12px;color:#8b949e;margin-bottom:4px">{label}: {value}{suffix}</div>'
            f'<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{color}"></div></div>'
            f'</div>')

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Silver AI Agent PRO v4")
    st.caption("COMEX Silver Futures (SI=F)")
    st.divider()
    snowflake_key = _get_secret()
    if snowflake_key:
        st.success("U0001f510 API key loaded from Snowflake Secret")
        api_key = snowflake_key
    else:
        api_key = st.text_input("Anthropic API Key", type="password",
                                placeholder="sk-ant-…",
                                help="Enter your API key to unlock AI analysis.")
    auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)
    run_btn = st.button("U0001f504 Refresh Now", use_container_width=True, type="primary")
    st.divider()
    forecast_days = st.slider("U0001f4c5 Forecast Days", 1, 30, 7)
    st.divider()
    st.markdown("**Chart Overlays**")
    show_gold_overlay = st.checkbox("Show Gold/Silver Ratio Chart", value=True)
    show_dxy_overlay = st.checkbox("Show DXY vs Silver Chart", value=True)
    show_news = st.checkbox("Show News Sentiment", value=True)
    show_etf_vflow = st.checkbox("Show ETF Volume Flow", value=True)
    st.divider()
    st.markdown("**Indicators**")
    st.markdown("EMA 20/50/200 · RSI · StochRSI")
    st.markdown("Williams %R · CCI · ADX/DMI")
    st.markdown("MACD · Bollinger Bands · ATR")
    st.markdown("OBV · VWAP · Ichimoku Cloud")
    st.markdown("Supertrend · Divergence Detection")
    st.markdown("Signal Score (composite 0-100)")
    st.markdown("Gold/Silver Ratio · DXY Overlay")
    st.markdown("ETF Volume Flow (SLV/SIVR/PSLV/AGQ/ZSL)")
    st.divider()
    st.markdown("**Prediction Models**")
    st.markdown("Linear · Polynomial · MA · Exp Smoothing · Momentum → Ensemble")
    st.divider()
    if _IN_SNOWFLAKE:
        st.caption("U0001f3d4 Running inside Snowflake")
    st.caption("Data: Yahoo Finance | AI: Claude")
    st.divider()
    st.warning("⚠️ Educational use only. Not financial advice.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 style=color:#f0f6fc;margin-bottom:0>⚡ Silver AI Trading Agent PRO v4</h1>",
            unsafe_allow_html=True)
st.caption("Live COMEX Silver Futures (SI=F) · Updated "+datetime.now().strftime("%H:%M:%S")
           +" · New: ETF Volume Flow · Gold/Silver Ratio · DXY Overlay · News Sentiment")
st.divider()

if run_btn:
    st.cache_data.clear()

with st.spinner("Fetching live silver prices…"):
    try:
        snap = load_snapshot()
    except Exception as e:
        st.error("Data fetch failed: "+str(e))
        st.stop()

if not snap.get("price") or snap["price"] <= 0:
    st.error("⚠️ Price data unavailable. Please refresh.")
    st.stop()

p = snap["price"]
chg = float(snap["change_pct"]) if snap.get("change_pct") is not None else 0.0
chg_color = "bull" if chg >= 0 else "bear"
arrow = "▲" if chg >= 0 else "▼"

# ── Top Metric Cards ──────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
gs_ratio = snap.get("gs_ratio")
gold_price = snap.get("gold_price")
dxy_val = snap.get("dxy")

cards = [
    ("U0001f4b0 SILVER / OZ","$"+str(p), arrow+" "+str(abs(chg))+"%", chg_color),
    ("U0001f4c5 Prev Close","$"+str(snap["prev_close"]),"Day: "+str(snap["d_low"])+" – "+str(snap["d_high"]),"neut"),
    ("U0001f4ca Week Range",str(snap["week_low"]),"↑ "+str(snap["week_high"]),"neut"),
    ("U0001f4c6 Month Range",str(snap["month_low"]),"↑ "+str(snap["month_high"]),"neut"),
    ("U0001f947 Gold Price","$"+str(round(gold_price,2)) if gold_price else "—",
     "G/S Ratio: "+str(gs_ratio) if gs_ratio else "Loading…","neut"),
    ("U0001f4b5 DXY (USD)","$"+str(round(dxy_val,2)) if dxy_val else "—",
     "Inverse corr w/ Silver","neut"),
]
for col,(label,val,sub,sc) in zip([c1,c2,c3,c4,c5,c6],cards):
    col.markdown(metric_card(label,val,sub,sc), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gold/Silver Ratio Card ─────────────────────────────────────────────────────────────────────────────
if gs_ratio:
    gs_color = "#3fb950" if gs_ratio > 80 else "#f85149" if gs_ratio < 55 else "#d29922"
    gs_label = "Silver CHEAP — Contrarian BUY Zone" if gs_ratio > 80 else ("Silver EXPENSIVE" if gs_ratio < 55 else "Normal Range")
    st.markdown(f'''<div class="ratio-card" style="margin-bottom:16px">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
<div>
<div style="font-size:13px;color:#8b949e">⚖️ Gold / Silver Ratio</div>
<div style="font-size:36px;font-weight:800;color:{gs_color}">{gs_ratio:.1f}</div>
<div style="font-size:13px;color:{gs_color}">{gs_label}</div>
</div>
<div style="font-size:13px;color:#8b949e;max-width:400px;line-height:1.6">
Historically, ratio above 80 means silver is undervalued vs gold — a classic contrarian entry signal.
Ratio below 55 suggests silver is expensive relative to gold.
<br><b style="color:#c9d1d9">Current: {gs_ratio:.1f}x</b>
</div>
</div>
</div>''', unsafe_allow_html=True)

# ── News Sentiment Display ────────────────────────────────────────────────────────────────
news_data = {}
if show_news:
    with st.spinner("Fetching news sentiment…"):
        news_data = fetch_news_sentiment()
    if news_data.get("headlines"):
        sent = news_data["sentiment_score"]
        sent_color = "#3fb950" if sent > 0.2 else "#f85149" if sent < -0.2 else "#d29922"
        sent_label = "BULLISH" if sent > 0.2 else "BEARISH" if sent < -0.2 else "NEUTRAL"
        with st.expander(f"U0001f4f0 News Sentiment: {sent_label} ({sent:+.2f}) — {news_data['count']} headlines", expanded=False):
            st.markdown(f'<div style="margin-bottom:8px"><span style="color:{sent_color};font-weight:700;font-size:16px">{sent_label}</span>'
                        f'<span style="color:#8b949e;font-size:12px;margin-left:8px">Sentiment Score: {sent:+.2f}</span></div>',
                        unsafe_allow_html=True)
            for h in news_data.get("headlines",[]):
                h_lower = h.lower()
                bullish_kw = ["surge","rally","rise","gain","bull","up","high","strong","buy","support","breakout"]
                bearish_kw = ["fall","drop","decline","bear","low","weak","sell","pressure","breakdown"]
                bull = sum(1 for k in bullish_kw if k in h_lower)
                bear_c = sum(1 for k in bearish_kw if k in h_lower)
                dot = "U0001f7e2" if bull > bear_c else "U0001f534" if bear_c > bull else "⚪"
                st.markdown(f'<div class="news-card">{dot} {h}</div>', unsafe_allow_html=True)

# ── Signal Score ───────────────────────────────────────────────────────────────────────────────
st.markdown("### U0001f3af Signal Score — Composite Indicator Consensus")
scoring = signal_score(snap)
sc1, sc2 = st.columns([1, 2])
with sc1:
    score_html = ('<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;text-align:center">'
                  '<div style="font-size:13px;color:#8b949e;margin-bottom:8px">OVERALL SIGNAL</div>'
                  '<div style="font-size:52px;font-weight:800;color:'+scoring["color"]+'">'
                  +str(scoring["score"])+'</div>'
                  '<div style="font-size:18px;font-weight:700;color:'+scoring["color"]+'">'
                  +scoring["label"]+'</div></div>')
    st.markdown(score_html, unsafe_allow_html=True)
    st.markdown(score_bar_html(scoring["score"],scoring["color"]), unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=scoring["score"],
        domain={"x":[0,1],"y":[0,1]},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#8b949e"},
            "bar":{"color":scoring["color"]},
            "bgcolor":"#21262d",
            "bordercolor":"#30363d",
            "steps":[
                {"range":[0,40],"color":"rgba(248,81,73,0.2)"},
                {"range":[40,60],"color":"rgba(210,153,34,0.2)"},
                {"range":[60,100],"color":"rgba(63,185,80,0.2)"},
            ],
            "threshold":{"line":{"color":scoring["color"],"width":3},"thickness":0.75,"value":scoring["score"]}
        },
        number={"font":{"color":scoring["color"],"size":28},"suffix":"/100"},
    ))
    fig_gauge.update_layout(
        template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        height=200,margin=dict(l=20,r=20,t=20,b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)
with sc2:
    st.markdown("**Indicator Breakdown**")
    breakdown = scoring["breakdown"]
    for i in range(0, len(breakdown), 2):
        row_items = breakdown[i:i+2]
        row_cols = st.columns(2)
        for col,(name,lbl,color,val) in zip(row_cols,row_items):
            col.markdown('<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 10px;margin-bottom:4px">'
                         '<div style="font-size:11px;color:#8b949e">'+name+'</div>'
                         '<div style="font-size:13px;color:'+color+';font-weight:700">'+lbl+'</div>'
                         '<div style="font-size:11px;color:#6e7681">'+str(val)+'</div></div>',
                         unsafe_allow_html=True)
st.divider()
# ── Chart Section ──────────────────────────────────────────────────────────────────────────────
st.markdown("### U0001f4ca Live Chart — Candlestick + Ichimoku + Supertrend + EMAs + MACD + RSI")
st.caption("Green/Red fill = Ichimoku Cloud · Dots = Supertrend · EMA 20/50/200 · VWAP · S/R · Fibonacci")
tab_d,tab_h,tab_tv = st.tabs(["U0001f4c5 Daily (6 months)","⏱ Hourly (5 days)","U0001f4e1 TradingView Live"])
with tab_d:
    fig_d = make_chart(snap["daily_df"].tail(120),snap["d_sup"],snap["d_res"],
                       snap["fibs"],snap["price"],timeframe="Daily")
    st.plotly_chart(fig_d, use_container_width=True)
with tab_h:
    fig_h = make_chart(snap["hourly_df"],snap["h_sup"],snap["h_res"],
                       snap["fibs"],snap["price"],show_ichimoku=False,timeframe="Hourly")
    st.plotly_chart(fig_h, use_container_width=True)
with tab_tv:
    st.markdown("⁠#### U0001f4e1 TradingView Live Silver Chart")
    st.caption("Live data from TradingView · COMEX:SI1! · Hourly candles")
    components.html(tradingview_ticker_tape(), height=60, scrolling=False)
    tv_col1,tv_col2 = st.columns([3,1])
    with tv_col1:
        tv_interval = st.selectbox("Interval",["5","15","30","60","240","D","W"],index=3,
            format_func=lambda x: {"5":"5 min","15":"15 min","30":"30 min","60":"1 Hour",
                                   "240":"4 Hour","D":"Daily","W":"Weekly"}.get(x,x),
            key="tv_interval")
    with tv_col2:
        tv_height = st.slider("Chart Height",400,800,580,50,key="tv_height")
    tv_html = tradingview_widget(symbol="OANDA:XAGUSD",theme="dark",height=tv_height,interval=tv_interval)
    components.html(tv_html, height=tv_height+30, scrolling=False)
st.divider()
# ── Gold/Silver Ratio & DXY Charts ───────────────────────────────────────────────────────────
if show_gold_overlay or show_dxy_overlay:
    st.markdown("### ⚖️ Macro Context — Gold/Silver Ratio & USD Index")
    col_gs, col_dxy = st.columns(2)
    with col_gs:
        if show_gold_overlay and snap.get("gold_daily") is not None:
            fig_gs = make_gs_ratio_chart(snap["daily_df"], snap["gold_daily"])
            if fig_gs:
                st.plotly_chart(fig_gs, use_container_width=True)
            else:
                st.caption("Gold/Silver ratio data unavailable")
        else:
            st.caption("Enable 'Show Gold/Silver Ratio Chart' in sidebar")
    with col_dxy:
        if show_dxy_overlay and snap.get("dxy_daily") is not None:
            fig_dxy = make_dxy_overlay_chart(snap["daily_df"], snap["dxy_daily"])
            if fig_dxy:
                st.plotly_chart(fig_dxy, use_container_width=True)
            else:
                st.caption("DXY overlay data unavailable")
        else:
            st.caption("Enable 'Show DXY vs Silver Chart' in sidebar")
    st.divider()

# ── ETF Volume Flow Section ───────────────────────────────────────────────────────────────
if show_etf_vflow:
    st.markdown("### U0001f4ca Silver ETF Volume Flow — SLV · SIVR · PSLV · AGQ · ZSL")
    st.caption("Cumulative dollar flow, daily volume, OBV comparison · 3-month window · ZSL = inverse/bear ETF")
    with st.spinner("Fetching ETF volume flow data…"):
        try:
            etf_bias, etf_summary, etf_dfs = fetch_etf_vflow()
        except Exception as e:
            st.warning(f"ETF flow data unavailable: {e}")
            etf_bias, etf_summary, etf_dfs = "neutral", {}, {}
    if etf_summary:
        bias_colors = {"inflow":"#3fb950","outflow":"#f85149","neutral":"#d29922"}
        bias_color = bias_colors.get(etf_bias,"#d29922")
        bias_icons = {"inflow":"U0001f7e2","outflow":"U0001f534","neutral":"⚪"}
        bias_icon = bias_icons.get(etf_bias,"⚪")
        bias_labels = {"inflow":"NET INFLOW — Bullish Accumulation","outflow":"NET OUTFLOW — Bearish Distribution","neutral":"MIXED / NEUTRAL"}
        bias_label = bias_labels.get(etf_bias,"NEUTRAL")
        st.markdown(
            '<div class="etf-vflow-card"><div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
            '<div><div style="font-size:12px;color:#8b949e">ETF FLOW BIAS (5-day)</div>'
            f'<div style="font-size:24px;font-weight:800;color:{bias_color}">{bias_icon} {bias_label}</div></div>'
            '</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)        etf_cols = st.columns(len(etf_summary))
        for col_el, (sym, info) in zip(etf_cols, etf_summary.items()):
            flow = info["net_flow_5d"]
            flow_str = f"+${abs(flow):,.0f}" if flow > 0 else f"-${abs(flow):,.0f}"
            flow_col_map = {}
            if (flow > 0 and info["is_bear"]) or (flow < 0 and not info["is_bear"]):
                flow_color = "#f85149"
            elif flow != 0:
                flow_color = "#3fb950"
            else:
                flow_color = "#d29922"
            pct_str = f"{info['pct_chg_5d']:+.1f}%"
            pct_color = "#f85149" if info["pct_chg_5d"] < 0 else "#3fb950"
            bear_tag = ' <span style="color:#f85149;font-size:10px">[BEAR]</span>' if info["is_bear"] else ""
            col_el.markdown(
                '<div class="etf-stat">'
                f'<div style="font-size:13px;font-weight:700;color:#f0f6fc">{sym}{bear_tag}</div>'
                f'<div style="font-size:11px;color:#8b949e;margin-bottom:6px">{info["name"][:20]}</div>'
                f'<div style="font-size:18px;font-weight:700;color:#f0f6fc">${info["price"]}</div>'
                f'<div style="font-size:12px;color:{pct_color}">{pct_str} 5d</div>'
                f'<div style="font-size:11px;color:{flow_color};margin-top:4px">Flow: {flow_str}</div>'
                f'<div style="font-size:11px;color:#8b949e">Vol: {info["vol_ratio"]}x avg</div>'
                '</div>',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        etf_tab1, etf_tab2 = st.tabs(["U0001f4ca Volume Flow Chart", "U0001f4c8 OBV Comparison"])
        with etf_tab1:
            etf_select = st.multiselect(
                "Select ETFs to display:",
                options=list(etf_dfs.keys()),
                default=[s for s in ["SLV","SIVR","PSLV"] if s in etf_dfs],
                key="etf_vflow_select")
            if etf_select:
                fig_etf = make_etf_vflow_chart(etf_dfs, selected_etfs=etf_select)
                if fig_etf:
                    st.plotly_chart(fig_etf, use_container_width=True)
            else:
                st.caption("Select at least one ETF above.")
        with etf_tab2:
            etf_obv_select = st.multiselect(
                "Select ETFs for OBV:",
                options=list(etf_dfs.keys()),
                default=list(etf_dfs.keys()),
                key="etf_obv_select")
            if etf_obv_select:
                fig_obv = make_etf_obv_chart(etf_dfs, selected_etfs=etf_obv_select)
                if fig_obv:
                    st.plotly_chart(fig_obv, use_container_width=True)
    else:
        st.warning("ETF volume flow data unavailable. Check internet connection.")
    st.divider()
# ── Pro Indicators Table ───────────────────────────────────────────────────────────────────────────
st.markdown("### U0001f4c8 Pro Technical Indicators — Multi-Timeframe")
left,right = st.columns(2)
def ind_row(name,val_str,sig_label,sig_kind):
    r1,r2,r3 = st.columns([1.2,1.6,1.0])
    r1.markdown("**"+name+"**")
    r2.markdown("`"+val_str+"`")
    r3.markdown(sig_tag(sig_label,sig_kind), unsafe_allow_html=True)
    st.divider()
with left:
    st.markdown("**Daily Oscillators**")
    d_rsi=snap["d_rsi"] or 50; h_rsi=snap["h_rsi"] or 50
    rsi_lbl="OVERBOUGHT" if d_rsi>70 else "OVERSOLD" if d_rsi<30 else "NEUTRAL"
    rsi_kind="bear" if d_rsi>70 else "bull" if d_rsi<30 else "neut"
    ind_row("RSI(14)","D:"+str(d_rsi)+" H:"+str(h_rsi),rsi_lbl,rsi_kind)
    sk=snap.get("d_stochrsi_k") or 50; sd_v=snap.get("d_stochrsi_d") or 50
    stk_lbl="OVERBOUGHT" if sk>80 else "OVERSOLD" if sk<20 else ("K>D BULL" if sk>sd_v else "K<D BEAR")
    stk_kind="bear" if sk>80 else "bull" if sk<20 else ("bull" if sk>sd_v else "bear")
    ind_row("StochRSI K/D","K:"+str(sk)+" D:"+str(sd_v),stk_lbl,stk_kind)
    wr=snap.get("d_wr") or -50
    wr_lbl="OVERBOUGHT" if wr>-20 else "OVERSOLD" if wr<-80 else "NEUTRAL"
    wr_kind="bear" if wr>-20 else "bull" if wr<-80 else "neut"
    ind_row("Williams %R",str(wr),wr_lbl,wr_kind)
    cci=snap.get("d_cci") or 0
    cci_lbl="OVERBOUGHT" if cci>100 else "OVERSOLD" if cci<-100 else "NEUTRAL"
    cci_kind="bear" if cci>100 else "bull" if cci<-100 else "neut"
    ind_row("CCI(20)",str(round(cci,1)),cci_lbl,cci_kind)
    adx=snap.get("d_adx") or 0; pdi=snap.get("d_pdi") or 0; mdi=snap.get("d_mdi") or 0
    adx_lbl=("STRONG BUY" if adx>25 and pdi>mdi else "STRONG SELL" if adx>25 and mdi>pdi else "WEAK TREND")
    adx_kind="bull" if adx>25 and pdi>mdi else "bear" if adx>25 and mdi>pdi else "neut"
    ind_row("ADX/+DI/-DI","ADX:"+str(round(adx,1))+" +"+str(round(pdi,1))+"/-"+str(round(mdi,1)),adx_lbl,adx_kind)
    macd_bull=(snap.get("d_macd") or 0)>(snap.get("d_macd_sig") or 0)
    ind_row("MACD(12/26/9)","Line:"+str(snap["d_macd"])+" Sig:"+str(snap["d_macd_sig"]),
            "BULLISH" if macd_bull else "BEARISH","bull" if macd_bull else "bear")
with right:
    st.markdown("**Trend & Price Structure**")
    e20=snap.get("d_ema20") or 0; e50=snap.get("d_ema50") or 0; e200=snap.get("d_ema200") or 0
    ema_bull=p>e20>e50>e200; ema_bear=p<e20<e50<e200
    ema_kind="bull" if ema_bull else "bear" if ema_bear else "neut"
    ind_row("EMA 20/50/200",str(e20)+"/"+str(e50)+"/"+str(e200),
            "BULL STACK" if ema_bull else "BEAR STACK" if ema_bear else "MIXED",ema_kind)
    bb_up=snap.get("d_bb_up") or 0; bb_lo=snap.get("d_bb_lo") or 0
    if p>bb_up: bb_l,bb_k="ABOVE UPPER","bear"
    elif p<bb_lo: bb_l,bb_k="BELOW LOWER","bull"
    else: bb_l,bb_k="WITHIN BANDS","neut"
    ind_row("Bollinger(20,2σ)",str(bb_lo)+" – "+str(bb_up),bb_l,bb_k)
    vwap=snap.get("d_vwap") or 0
    vwap_kind="bull" if p>vwap else "bear"
    ind_row("VWAP","$"+str(vwap),"ABOVE VWAP" if p>vwap else "BELOW VWAP",vwap_kind)
    sa=snap.get("d_ichi_sa") or 0; sb_v=snap.get("d_ichi_sb") or 0
    if sa and sb_v:
        cloud_top=max(sa,sb_v); cloud_bot=min(sa,sb_v)
        if p>cloud_top: ichi_l,ichi_k="ABOVE CLOUD","bull"
        elif p<cloud_bot: ichi_l,ichi_k="BELOW CLOUD","bear"
        else: ichi_l,ichi_k="IN CLOUD","neut"
        ind_row("Ichimoku Cloud","A:"+str(sa)+" B:"+str(sb_v),ichi_l,ichi_k)
    st_dir=snap.get("d_st_dir"); st_val=snap.get("d_supertrend") or 0
    if st_dir is not None:
        st_lbl="BUY SIGNAL" if st_dir==-1 else "SELL SIGNAL"
        st_kind="bull" if st_dir==-1 else "bear"
        ind_row("Supertrend","$"+str(st_val),st_lbl,st_kind)
    rsi_div=snap.get("d_rsi_div") or "none"
    macd_div=snap.get("d_macd_div") or "none"
    div_text=[]
    if rsi_div!="none": div_text.append("RSI: "+rsi_div.upper())
    if macd_div!="none": div_text.append("MACD: "+macd_div.upper())
    div_display=" | ".join(div_text) if div_text else "none detected"
    div_kind="bull" if "bullish" in (rsi_div+macd_div) else "bear" if "bearish" in (rsi_div+macd_div) else "neut"
    ind_row("Divergence",div_display,"DETECTED" if div_text else "NONE",div_kind)
    if gs_ratio:
        gs_lbl = "SILVER CHEAP" if gs_ratio > 80 else ("SILVER EXPENSIVE" if gs_ratio < 55 else "NORMAL")
        gs_kind = "bull" if gs_ratio > 80 else "bear" if gs_ratio < 55 else "neut"
        ind_row("G/S Ratio","Ratio: "+str(gs_ratio),gs_lbl,gs_kind)
    etf_bias_snap = snap.get("etf_flow_bias") or "neutral"
    etf_bias_lbl = {"inflow":"NET INFLOW","outflow":"NET OUTFLOW","neutral":"NEUTRAL"}.get(etf_bias_snap,"NEUTRAL")
    etf_bias_kind = {"inflow":"bull","outflow":"bear","neutral":"neut"}.get(etf_bias_snap,"neut")
    ind_row("ETF Volume Flow",etf_bias_snap.upper(),etf_bias_lbl,etf_bias_kind)
st.divider()
# ── Multi-Timeframe Confluence Table ───────────────────────────────────────────────────────────
st.markdown("### U0001f500 Multi-Timeframe Confluence Table")
def tf_signal(rsi,stochrsi_k,wr,macd,macd_sig,adx,pdi,mdi):
    bulls=0; bears=0
    if rsi and rsi<50: bulls+=1
    if rsi and rsi>50: bears+=1
    if stochrsi_k and stochrsi_k<50: bulls+=1
    if stochrsi_k and stochrsi_k>50: bears+=1
    if wr and wr<-50: bulls+=1
    if wr and wr>-50: bears+=1
    if macd and macd_sig and macd>macd_sig: bulls+=1
    if macd and macd_sig and macd<macd_sig: bears+=1
    if adx and pdi and mdi and adx>20 and pdi>mdi: bulls+=1
    if adx and pdi and mdi and adx>20 and mdi>pdi: bears+=1
    total=bulls+bears
    if total==0: return "NEUTRAL","#d29922"
    ratio=bulls/total
    if ratio>=0.7: return "BULLISH","#3fb950"
    if ratio<=0.3: return "BEARISH","#f85149"
    return "NEUTRAL","#d29922"
d_sig,d_col=tf_signal(snap.get("d_rsi"),snap.get("d_stochrsi_k"),snap.get("d_wr"),
                       snap.get("d_macd"),snap.get("d_macd_sig"),snap.get("d_adx"),snap.get("d_pdi"),snap.get("d_mdi"))
h_sig,h_col=tf_signal(snap.get("h_rsi"),snap.get("h_stochrsi_k"),snap.get("h_wr"),
                       snap.get("h_macd"),snap.get("h_macd_sig"),snap.get("h_adx"),None,None)
d_macd_arrow="▲" if (snap.get("d_macd") or 0)>(snap.get("d_macd_sig") or 0) else "▼"
h_macd_arrow="▲" if (snap.get("h_macd") or 0)>(snap.get("h_macd_sig") or 0) else "▼"
table_html=('<table class="conf-table">'
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
# ── Key Levels + Ichimoku ──────────────────────────────────────────────────────────────────────
left2,right2 = st.columns(2)
with left2:
    st.markdown("### U0001f511 Key Price Levels")
    fib=snap["fibs"]
    st.markdown("**U0001f534 Resistance**")
    if snap["d_res"]:
        res_badges = ' '.join(['<span class="level-badge res-badge">$'+str(v)+'</span>' for v in snap["d_res"]])
        st.markdown(res_badges, unsafe_allow_html=True)
    else:
        st.caption("No daily resistance above price")
    st.markdown("**U0001f7e2 Support**")
    if snap["d_sup"]:
        sup_badges = ' '.join(['<span class="level-badge sup-badge">$'+str(v)+'</span>' for v in snap["d_sup"]])
        st.markdown(sup_badges, unsafe_allow_html=True)
    else:
        st.caption("Price below all pivots — watch Fib extensions")
    st.markdown("**U0001f535 Fibonacci (60-day swing)**")
    st.caption("Swing Low: $"+str(fib["swing_low"])+" → Swing High: $"+str(fib["swing_high"]))
    for label_,val in [("23.6%",fib["ret_236"]),("38.2%",fib["ret_382"]),
                       ("50.0%",fib["ret_500"]),("61.8%",fib["ret_618"]),("78.6%",fib["ret_786"]),
                       ("Ext 127.2%",fib["ext_1272"]),("Ext 161.8%",fib["ext_1618"])]:
        a,b = st.columns([1,1.5])
        a.caption(label_)
        b.markdown(badge(val,"fib"), unsafe_allow_html=True)
with right2:
    st.markdown("### U0001f310 Ichimoku Cloud Levels")
    st.markdown("**Tenkan-sen (9)**")
    st.markdown(badge(snap.get("d_ichi_tenkan") or "—","fib"), unsafe_allow_html=True)
    st.markdown("**Kijun-sen (26)**")
    st.markdown(badge(snap.get("d_ichi_kijun") or "—","fib"), unsafe_allow_html=True)
    st.markdown("**Senkou A**")
    st.markdown(badge(snap.get("d_ichi_sa") or "—","sup"), unsafe_allow_html=True)
    st.markdown("**Senkou B**")
    st.markdown(badge(snap.get("d_ichi_sb") or "—","res"), unsafe_allow_html=True)
    st.markdown("**Supertrend Level**")
    st_badge_kind="sup" if snap.get("d_st_dir")==-1 else "res"
    st.markdown(badge(snap.get("d_supertrend") or "—",st_badge_kind), unsafe_allow_html=True)
st.divider()
# ── Live Data Monitoring ────────────────────────────────────────────────────────────────────
st.markdown("### U0001f7e2 Live Data Monitoring System")
st.caption("Real-time 5-min tick · Smart Alerts · Session Stats · Volume Analysis · Momentum Gauges")
live = fetch_live_price()
if not live:
    st.warning("⚠️ Live tick data temporarily unavailable — displaying snapshot price $"+str(snap["price"]))
else:
    tick_color="#3fb950" if live["chg_1m"]>=0 else "#f85149"
    day_color="#3fb950" if live["chg_day"]>=0 else "#f85149"
    tick_arrow="▲" if live["chg_1m"]>=0 else "▼"
    day_arrow="▲" if live["chg_day"]>=0 else "▼"
    def mon_card(label,value,sub="",color="#f0f6fc"):
        return ('<div class="metric-card">'
                '<div class="metric-label">'+label+'</div>'
                '<div class="metric-value" style="color:'+color+'">'+str(value)+'</div>'
                '<div class="metric-sub">'+sub+'</div></div>')
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.markdown(mon_card("⚡ LIVE PRICE","$"+str(live["price"]),
        tick_arrow+" "+str(live["chg_1m"])+" ("+str(live["chg_1m_pct"])+"%) 1min",tick_color), unsafe_allow_html=True)
    m2.markdown(mon_card("DAY CHANGE",day_arrow+" "+str(abs(live["chg_day_pct"]))+"%",
        "vs open $"+str(live["session_open"]),day_color), unsafe_allow_html=True)
    m3.markdown(mon_card("SESSION HIGH","$"+str(live["session_high"]),
        "+"+str(round(live["session_high"]-live["session_open"],3))+" from open","#3fb950"), unsafe_allow_html=True)
    m4.markdown(mon_card("SESSION LOW","$"+str(live["session_low"]),
        str(round(live["session_low"]-live["session_open"],3))+" from open","#f85149"), unsafe_allow_html=True)
    m5.markdown(mon_card("SESSION RANGE",str(live["session_range_pct"])+"%",
        "$"+str(live["session_low"])+" – $"+str(live["session_high"]),"#d29922"), unsafe_allow_html=True)
    vol_ratio = live.get("vol_ratio", 0)
    vol_display = str(vol_ratio)+"x" if vol_ratio > 0 else "Calculating…"
    vc="#f85149" if vol_ratio>=2 else "#d29922" if vol_ratio>=1.3 else "#8b949e"
    m6.markdown(mon_card("VOL RATIO",vol_display,
        "last "+str(live["last_vol"])+" / avg "+str(live["avg_vol"]),vc), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_alerts,col_gauge,col_ticks = st.columns([1.4,1,1.2])
    with col_alerts:
        st.markdown("#### ⚡ Alerts")
        alerts = check_alerts(live,snap)
        dot_map={"red":"U0001f534","green":"U0001f7e2","yellow":"U0001f7e1"}
        for al in alerts:
            dot=dot_map.get(al["level"],"U0001f7e1")
            st.markdown(dot+" "+al["msg"])    with col_gauge:
        st.markdown("#### **Momentum Gauges**")
        span=live["session_high"]-live["session_low"]
        pos=int(((live["price"]-live["session_low"])/span*100)) if span else 50
        bc="#3fb950" if pos>60 else "#f85149" if pos<40 else "#d29922"
        st.markdown('<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">Session Position</div>'
            '<div class="score-bar-wrap"><div class="score-bar" style="width:'+str(pos)+'%;background:'+bc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+str(pos)+'% of range</div></div>',
            unsafe_allow_html=True)
        rv=snap.get("d_rsi") or 50
        rp=int(min(100,max(0,rv)))
        rc="#f85149" if rv>70 else "#3fb950" if rv<30 else "#d29922"
        rl="OVERBOUGHT" if rv>70 else "OVERSOLD" if rv<30 else "NEUTRAL"
        st.markdown('<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">RSI(14): '+str(rv)+'</div>'
            '<div class="score-bar-wrap"><div class="score-bar" style="width:'+str(rp)+'%;background:'+rc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+rl+'</div></div>',
            unsafe_allow_html=True)
        sk_live=snap.get("d_stochrsi_k") or 50
        skp=int(min(100,max(0,sk_live)))
        skc="#f85149" if sk_live>80 else "#3fb950" if sk_live<20 else "#d29922"
        skl="OVERBOUGHT" if sk_live>80 else "OVERSOLD" if sk_live<20 else "NEUTRAL"
        st.markdown('<div style="margin-bottom:12px"><div style="font-size:12px;color:#8b949e;margin-bottom:4px">StochRSI K: '+str(sk_live)+'</div>'
            '<div class="score-bar-wrap"><div class="score-bar" style="width:'+str(skp)+'%;background:'+skc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+skl+'</div></div>',
            unsafe_allow_html=True)
        vp=int(min(100,vol_ratio/3*100))
        vgc="#f85149" if vol_ratio>=2 else "#d29922" if vol_ratio>=1.3 else "#3fb950"
        vl="SPIKE" if vol_ratio>=2 else "ELEVATED" if vol_ratio>=1.3 else "NORMAL"
        st.markdown('<div><div style="font-size:12px;color:#8b949e;margin-bottom:4px">Volume: '+str(vol_ratio)+'x avg</div>'
            '<div class="score-bar-wrap"><div class="score-bar" style="width:'+str(vp)+'%;background:'+vgc+'"></div></div>'
            '<div style="font-size:11px;color:#8b949e;margin-top:3px">'+vl+'</div></div>',
            unsafe_allow_html=True)
    with col_ticks:
        st.markdown("#### **Price Tick Log**")
        ticks=live.get("ticks",[])[-15:]
        th='<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:6px;max-height:220px;overflow-y:auto;">'
        for tk in reversed(ticks):
            tc="#3fb950" if tk["dir"]=="U" else "#f85149"
            ta="▲" if tk["dir"]=="U" else "▼"
            th+='<div style="display:flex;justify-content:space-between;padding:2px 4px;">'
            th+='<span style="color:#8b949e">'+tk["t"]+'</span>'
            th+='<span style="color:'+tc+';font-weight:700">'+ta+' $'+str(tk["p"])+'</span></div>'
        th+='</div>'
        st.markdown(th, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        tick_prices=[tk["p"] for tk in live.get("ticks",[])]
        tick_times=[tk["t"] for tk in live.get("ticks",[])]
        if len(tick_prices)>=2:
            sp_color="#3fb950" if tick_prices[-1]>=tick_prices[0] else "#f85149"
            fill_c="rgba(63,185,80,0.08)" if sp_color=="#3fb950" else "rgba(248,81,73,0.08)"
            fig_spark=go.Figure(go.Scatter(
                x=tick_times,y=tick_prices,mode="lines+markers",
                line=dict(color=sp_color,width=2),marker=dict(size=4,color=sp_color),
                fill="tozeroy",fillcolor=fill_c,
                hovertemplate="<b>%{x}</b><br>$%{y:.3f}<extra></extra>"))
            fig_spark.update_layout(
                template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
                height=140,margin=dict(l=0,r=0,t=10,b=0),showlegend=False,
                xaxis=dict(showgrid=False,tickfont=dict(size=9)),
                yaxis=dict(showgrid=True,gridcolor="#1c2128",tickformat="$.3f",tickfont=dict(size=9)))
            st.caption("Last "+str(len(tick_prices))+" ticks")
            st.plotly_chart(fig_spark, use_container_width=True)
st.divider()
# ── Price Predictor Section ──────────────────────────────────────────────────────────────────
st.markdown('### U0001f52e Price Predictor — 5-Model Ensemble Forecast')
st.caption(f'Forecasting {forecast_days} trading days ahead · Linear · Polynomial · MA · Exponential Smoothing · Momentum')
with st.spinner('Running forecast models…'):
    fc = run_forecasts(snap['daily_df'], forecast_days)
current = snap['price']
future_dates = fc['future_dates']
ensemble = fc['ensemble']
lin_pred = fc['lin_pred']
poly_pred = fc['poly_pred']
ma_pred = fc['ma_pred']
exp_pred = fc['exp_pred']
mom_pred = fc['mom_pred']
ci_upper = fc['ci_upper']
ci_lower = fc['ci_lower']
close_arr = fc['close']
ens_target = ensemble[-1]
ens_chg_pct = ((ens_target - current) / current) * 100
ens_color = '#3fb950' if ens_target > current else '#f85149'
ens_arrow = '▲' if ens_target > current else '▼'
pa, pb, pc = st.columns(3)
pa.markdown(
    f'<div class="pred-card"><div style="font-size:0.7rem;color:#8b949e">Ensemble Target ({forecast_days}d)</div>'
    f'<div style="font-size:2rem;font-weight:700;color:{ens_color}">{ens_arrow} ${ens_target:.2f}</div>'
    f'<div style="font-size:0.85rem;color:#8b949e">{ens_chg_pct:+.2f}% from current</div></div>',
    unsafe_allow_html=True)
pb.markdown(
    f'<div class="pred-card"><div style="font-size:0.7rem;color:#8b949e">90% Confidence Range</div>'
    f'<div style="font-size:1.4rem;font-weight:700;color:#f0f6fc">${ci_lower[-1]:.2f} – ${ci_upper[-1]:.2f}</div>'
    f'<div style="font-size:0.85rem;color:#8b949e">±{((ci_upper[-1]-ci_lower[-1])/2/current*100):.1f}% uncertainty</div></div>',
    unsafe_allow_html=True)
bulls_count = sum(1 for v in [lin_pred[-1],poly_pred[-1],ma_pred[-1],exp_pred[-1],mom_pred[-1]] if v>current)
pc.markdown(
    f'<div class="pred-card"><div style="font-size:0.7rem;color:#8b949e">Model Consensus</div>'
    f'<div style="font-size:1.4rem;font-weight:700;color:#f0f6fc">{bulls_count}/5 Bullish</div>'
    f'<div style="font-size:0.85rem;color:#8b949e">models above current price</div></div>',
    unsafe_allow_html=True)pf_tab1, pf_tab2, pf_tab3 = st.tabs(['U0001f4c8 Candlestick + Forecast', 'U0001f500 All Models Comparison', 'U0001f4ca Technical Oscillators'])

with pf_tab1:
    df_tail = snap['daily_df'].tail(60).copy()
    if df_tail.index.tzinfo is not None:
        df_tail.index = df_tail.index.tz_convert('UTC').tz_localize(None)
    fig_pf = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25],vertical_spacing=0.05)
    fig_pf.add_trace(go.Candlestick(
        x=df_tail.index,open=df_tail['Open'],high=df_tail['High'],
        low=df_tail['Low'],close=df_tail['Close'],
        increasing_line_color='#26a69a',increasing_fillcolor='#26a69a',
        decreasing_line_color='#ef5350',decreasing_fillcolor='#ef5350',
        name='Silver',line_width=1),row=1,col=1)
    for col_,color,name in [('EMA20','#26a69a','EMA 20'),('EMA50','#ef5350','EMA 50'),
                             ('BB_Up','rgba(150,150,150,0.4)','BB Upper'),('BB_Lo','rgba(150,150,150,0.4)','BB Lower')]:
        if col_ in df_tail.columns:
            s=df_tail[col_].dropna()
            fig_pf.add_trace(go.Scatter(x=s.index,y=s.values,line=dict(color=color,width=1.2),name=name),row=1,col=1)
    fig_pf.add_trace(go.Scatter(x=list(future_dates),y=ensemble,
        line=dict(color='#c0c0c0',width=2.5,dash='dash'),name='Ensemble Forecast'),row=1,col=1)
    fig_pf.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates[::-1]),
        y=list(ci_upper)+list(ci_lower[::-1]),
        fill='toself',fillcolor='rgba(192,192,192,0.12)',
        line=dict(color='rgba(255,255,255,0)'),name='90% CI'),row=1,col=1)
    vol_colors=['rgba(38,166,154,0.5)' if df_tail['Close'].iloc[i]>=df_tail['Open'].iloc[i] else 'rgba(239,83,80,0.5)'
                for i in range(len(df_tail))]
    fig_pf.add_trace(go.Bar(x=df_tail.index,y=df_tail['Volume'],
        marker_color=vol_colors,name='Volume',opacity=0.7,showlegend=False),row=2,col=1)
    fig_pf.add_hline(y=current,line_color='rgba(255,255,255,0.6)',line_dash='dot',
        annotation_text=f' Now ${current:.2f}',annotation_font_color='#ffffff',row=1,col=1)
    fig_pf.update_layout(template='plotly_dark',paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',
        height=560,xaxis_rangeslider_visible=False,hovermode='x unified')
    fig_pf.update_yaxes(gridcolor='#1c2128')
    st.plotly_chart(fig_pf, use_container_width=True)

with pf_tab2:
    fig_models = go.Figure()
    for pred,name,color,dash in [
        (lin_pred,'Linear Regression','#ffa726','dot'),
        (poly_pred,'Polynomial (deg 3)','#ab47bc','dot'),
        (ma_pred,'MA Extrapolation','#42a5f5','dot'),
        (exp_pred,'Exponential Smoothing','#26a69a','dot'),
        (mom_pred,'Momentum','#ff7043','dot'),
        (ensemble,'Ensemble (Average)','#c0c0c0','solid')]:
        fig_models.add_trace(go.Scatter(x=list(future_dates),y=pred,
            line=dict(color=color,width=2.5 if name.startswith('Ensemble') else 1.5,dash=dash),name=name))
    fig_models.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates[::-1]),
        y=list(ci_upper)+list(ci_lower[::-1]),
        fill='toself',fillcolor='rgba(192,192,192,0.08)',
        line=dict(color='rgba(0,0,0,0)'),name='90% CI'))
    fig_models.add_hline(y=current,line_dash='dash',line_color='#8b949e',
        annotation_text=f' Current: ${current:.2f}',annotation_font_color='#8b949e')
    fig_models.update_layout(template='plotly_dark',paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',
        height=420,title='All Forecast Models Comparison',
        legend=dict(font=dict(size=11,color='#c9d1d9'),bgcolor='rgba(13,17,23,0.85)',bordercolor='#30363d',borderwidth=1),
        yaxis_gridcolor='#1c2128',hovermode='x unified')
    st.plotly_chart(fig_models, use_container_width=True)

with pf_tab3:
    fig_osc = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.06,
                            subplot_titles=('RSI (14)','MACD','ATR (14)'))
    df_osc = snap['daily_df'].copy()
    if df_osc.index.tzinfo is not None:
        df_osc.index = df_osc.index.tz_convert('UTC').tz_localize(None)
    if 'RSI' in df_osc.columns:
        fig_osc.add_trace(go.Scatter(x=df_osc.index,y=df_osc['RSI'].dropna(),
            line=dict(color='#42a5f5',width=1.5),name='RSI'),row=1,col=1)
        fig_osc.add_hrect(y0=70,y1=100,fillcolor='rgba(239,83,80,0.07)',line_width=0,row=1,col=1)
        fig_osc.add_hrect(y0=0,y1=30,fillcolor='rgba(38,166,154,0.07)',line_width=0,row=1,col=1)
        fig_osc.add_hline(y=70,line_dash='dash',line_color='#ef5350',row=1,col=1)
        fig_osc.add_hline(y=30,line_dash='dash',line_color='#26a69a',row=1,col=1)
    if 'MACD' in df_osc.columns:
        hist_s=df_osc['MACD_Hist'].dropna()
        hist_colors=['rgba(38,166,154,0.6)' if v>=0 else 'rgba(239,83,80,0.6)' for v in hist_s.values]
        fig_osc.add_trace(go.Bar(x=hist_s.index,y=hist_s.values,
            marker_color=hist_colors,name='Hist',showlegend=False),row=2,col=1)
        fig_osc.add_trace(go.Scatter(x=df_osc.index,y=df_osc['MACD'].dropna(),
            line=dict(color='#42a5f5',width=1.2),name='MACD'),row=2,col=1)
        fig_osc.add_trace(go.Scatter(x=df_osc.index,y=df_osc['MACD_Sig'].dropna(),
            line=dict(color='#ffa726',width=1.2,dash='dot'),name='Signal'),row=2,col=1)
    if 'ATR' in df_osc.columns:
        fig_osc.add_trace(go.Scatter(x=df_osc.index,y=df_osc['ATR'].dropna(),
            line=dict(color='#c0c0c0',width=1.5),name='ATR'),row=3,col=1)
    fig_osc.update_layout(template='plotly_dark',paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',height=520)
    fig_osc.update_yaxes(gridcolor='#1c2128')
    st.plotly_chart(fig_osc, use_container_width=True)
# Day-by-Day Forecast Table
st.markdown('### U0001f5d3️ Day-by-Day Price Forecast')
pred_df = pd.DataFrame({
    'Date': future_dates.strftime('%a %b %d, %Y'),
    'Linear ($)': [f'${v:.2f}' for v in lin_pred],
    'Polynomial ($)': [f'${v:.2f}' for v in poly_pred],
    'MA Extrap ($)': [f'${v:.2f}' for v in ma_pred],
    'Exp Smooth ($)': [f'${v:.2f}' for v in exp_pred],
    'Momentum ($)': [f'${v:.2f}' for v in mom_pred],
    'Ensemble ($)': [f'${v:.2f}' for v in ensemble],
    'Upper CI ($)': [f'${v:.2f}' for v in ci_upper],
    'Lower CI ($)': [f'${v:.2f}' for v in ci_lower],
    'Change from Now': [f'{((v-current)/current)*100:+.2f}%' for v in ensemble],
})
st.dataframe(pred_df.set_index('Date'), use_container_width=True)

# Forecast Signal Dashboard
st.markdown('### U0001f3af Forecast Signal Dashboard')
rsi_now=snap.get('d_rsi') or 50
macd_now=snap.get('d_macd') or 0; sig_now=snap.get('d_macd_sig') or 0
ma5_val=snap['daily_df']['EMA20'].iloc[-1] if 'EMA20' in snap['daily_df'].columns else current
ma50_val=snap['daily_df']['EMA50'].iloc[-1] if 'EMA50' in snap['daily_df'].columns else current

def signal_card(col, title, signal, value):
    col.markdown(
        f'<div class="pred-card">'
        f'<div style="font-size:0.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">{title}</div>'
        f'<div style="font-size:0.9rem;font-weight:600;color:#f0f6fc;margin:8px 0;">{signal}</div>'
        f'<div style="font-size:0.75rem;color:#6e7681;">{value}</div>'
        f'</div>', unsafe_allow_html=True)

fs1,fs2,fs3,fs4,fs5 = st.columns(5)
signal_card(fs1,'RSI Signal',
    'Oversold — Bullish U0001f7e2' if rsi_now<35 else ('Overbought — Bearish U0001f534' if rsi_now>65 else 'Neutral ⚪'),
    f'RSI: {rsi_now:.1f}')
signal_card(fs2,'MACD Cross',
    'Bullish Cross U0001f7e2' if macd_now>sig_now else 'Bearish Cross U0001f534',
    f'MACD {macd_now:.3f}')
signal_card(fs3,'EMA Trend',
    'Uptrend U0001f7e2' if current>ma5_val>ma50_val else ('Downtrend U0001f534' if current<ma5_val<ma50_val else 'Mixed ⚪'),
    f'EMA20 ${ma5_val:.2f}')
signal_card(fs4,'BB Position',
    'Near Upper U0001f534' if current>(snap.get('d_bb_up') or current)*0.98 else
    ('Near Lower U0001f7e2' if current<(snap.get('d_bb_lo') or current)*1.02 else 'Mid Band ⚪'),
    f'Mid ${snap.get("d_bb_mid") or 0:.2f}')
signal_card(fs5,'Forecast Bias',
    'U0001f4c8 Bullish' if ensemble[-1]>current else 'U0001f4c9 Bearish',
    f'Target ${ensemble[-1]:.2f}')
st.divider()
# Correlation with Related Assets
st.markdown('### U0001f4e1 Silver Correlation with Related Assets')
with st.spinner('Fetching correlation data…'):
    try:
        corr_data = get_correlation_data('1y')
        silver_ret = snap['daily_df']['Close'].pct_change().dropna()
        corr_results = {}
        for name,ret in corr_data.items():
            aligned = pd.concat([silver_ret,ret],axis=1,join='inner')
            if len(aligned)>10:
                c_val = aligned.corr().iloc[0,1]
                corr_results[name] = round(c_val,3)
        if corr_results:
            corr_df = pd.DataFrame(list(corr_results.items()),columns=['Asset','Correlation with Silver'])
            corr_df = corr_df.sort_values('Correlation with Silver',ascending=False)
            fig_corr = px.bar(corr_df,x='Asset',y='Correlation with Silver',
                color='Correlation with Silver',color_continuous_scale='RdYlGn',
                title='Silver 1-Year Correlation with Key Assets',range_color=[-1,1])
            fig_corr.update_layout(template='plotly_dark',height=320,
                paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',
                yaxis_gridcolor='#1c2128')
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception: st.caption('Correlation data unavailable')

# Returns Distribution & Risk Metrics
st.markdown('### U0001f4ca Returns Analysis & Risk Metrics')
col_a,col_b = st.columns(2)
with col_a:
    ret_series = snap['daily_df']['Close'].pct_change().dropna()*100
    fig_hist = px.histogram(ret_series,nbins=60,
        title='Daily Returns Distribution (%)',color_discrete_sequence=['#c0c0c0'])
    fig_hist.add_vline(x=0,line_dash='dash',line_color='white')
    fig_hist.update_layout(template='plotly_dark',height=280,
        paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',
        showlegend=False,xaxis_title='Return (%)',yaxis_title='Frequency')
    st.plotly_chart(fig_hist, use_container_width=True)
with col_b:
    ret_clean = snap['daily_df']['Close'].pct_change().dropna()
    sharpe_r = (ret_clean.mean()/ret_clean.std())*np.sqrt(252)
    var_95 = np.percentile(ret_clean,5)*100
    max_dd = ((snap['daily_df']['Close']/snap['daily_df']['Close'].cummax())-1).min()*100
    avg_ret_ann = ret_clean.mean()*252*100
    vol_ann_pct = ret_clean.std()*np.sqrt(252)*100
    risk_df = pd.DataFrame({
        'Metric':['Annualized Return','Sharpe Ratio','VaR 95% (Daily)','Max Drawdown','Annual Volatility'],
        'Value':[f'{avg_ret_ann:.1f}%',f'{sharpe_r:.2f}',f'{var_95:.2f}%',f'{max_dd:.1f}%',f'{vol_ann_pct:.1f}%']
    })
    st.dataframe(risk_df.set_index('Metric'), use_container_width=True, height=230)
st.divider()
# AI Analysis Section
st.markdown('### U0001f916 AI Trade Analysis & Targets')
st.caption('Powered by Claude '+AI_MODEL+' · All pro indicators included · Gold/Silver Ratio · DXY · ETF Flow · Cached 5 min')
if not api_key:
    st.info('U0001f448 Enter your Anthropic API key in the sidebar to unlock AI trade analysis.')
    st.markdown('<div class="analysis-box"><b>What AI Analysis Provides:</b>\n\n'
                '• MARKET BIAS with conviction level\n'
                '• BUY SETUP — Entry zone, targets (T1/T2/T3), stop loss, risk:reward\n'
                '• SELL SETUP — Entry zone, targets, stop loss, risk:reward\n'
                '• KEY LEVELS to watch\n'
                '• PATTERN/SIGNAL identification\n'
                '• RISK RATING (1-10)\n'
                '• TRADER NOTE with macro context (G/S Ratio, DXY, ETF Flow, Sentiment)\n'
                '</div>', unsafe_allow_html=True)
else:
    with st.spinner('Claude is analysing the market with all pro indicators…'):
        try:
            etf_bias_for_ai = snap.get('etf_flow_bias') if show_etf_vflow else None
            sent_score_for_ai = news_data.get('sentiment_score') if show_news and news_data else None
            analysis = get_ai_analysis(snap['price'], api_key,
                gs_ratio=snap.get('gs_ratio'),
                dxy=snap.get('dxy'),
                sentiment_score=sent_score_for_ai,
                etf_bias=etf_bias_for_ai)
            st.markdown('<div class="analysis-box">'+analysis+'</div>', unsafe_allow_html=True)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code==401:
                st.error('Invalid Anthropic API key.')
            else:
                st.error('API error: '+str(e))
        except Exception as e:
            st.error('AI analysis failed: '+str(e))

st.divider()
st.markdown(
    '<div style="text-align:center;color:#8b949e;font-size:0.78rem;">'
    '⚡ Silver AI Trading Agent PRO v4 · '
    'New: ETF Volume Flow (SLV/SIVR/PSLV/AGQ/ZSL) · '
    'Gold/Silver Ratio · DXY Overlay · News Sentiment · Gauge Charts · '
    'Data: Yahoo Finance · AI: Claude · '
    'Forecasting: Linear · Polynomial · MA · Exp Smoothing · Momentum Ensemble · '
    '⚠️ Educational use only — not financial advice.</div>',
    unsafe_allow_html=True)

if auto_refresh:
    time.sleep(300)
    st.rerun()
