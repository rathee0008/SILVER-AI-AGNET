import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title='Silver Price Predictor',
    page_icon='🥈',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown('''
<style>
.metric-card {
    background: linear-gradient(135deg, #1e2130, #2a2f45);
    border: 1px solid #3a3f55;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #c0c0c0; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.up { color: #00d26a !important; }
.down { color: #ff4d6d !important; }
.pred-card {
    background: linear-gradient(135deg, #1a2035, #252b42);
    border: 1px solid #4a5080;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    text-align: center;
}
</style>
''', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('## 🥈 Silver Predictor')
period = st.sidebar.selectbox('Historical Period', ['6mo', '1y', '2y', '5y'], index=1)
interval = st.sidebar.selectbox('Interval', ['1d', '1wk'], index=0)
forecast_days = st.sidebar.slider('Forecast Days', 1, 30, 7)
st.sidebar.markdown('---')
st.sidebar.markdown('**Models Used**')
st.sidebar.markdown('- Linear Regression\n- Polynomial Regression\n- Moving Average\n- Exponential Smoothing\n- Momentum Forecast')
st.sidebar.markdown('---')
st.sidebar.markdown('⚠️ *For educational purposes only. Not financial advice.*')

# Header
st.markdown('# 🥈 Silver Price Predictor')
st.markdown('**AI-powered analysis with multiple forecasting models · Live COMEX Silver (SI=F)**')
st.markdown('---')

# Data Fetching
@st.cache_data(ttl=300)
def fetch_silver_data(period, interval):
    ticker = yf.Ticker('SI=F')
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        ticker = yf.Ticker('SLV')
        df = ticker.history(period=period, interval=interval)
    df = df.dropna()
    return df

@st.cache_data(ttl=60)
def fetch_live_price():
    try:
        t = yf.Ticker('SI=F')
        info = t.fast_info
        return float(info.last_price), float(info.previous_close)
    except:
        return None, None

with st.spinner('Fetching silver market data...'):
    df = fetch_silver_data(period, interval)
    live_price, prev_close = fetch_live_price()

if df.empty:
    st.error('⚠️ Could not fetch silver data. Please try again.')
    st.stop()

# Feature Engineering
df = df.copy()
df['Returns'] = df['Close'].pct_change()
df['MA5']  = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['EMA12'] = df['Close'].ewm(span=12).mean()
df['EMA26'] = df['Close'].ewm(span=26).mean()
df['MACD']  = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9).mean()
df['BB_mid'] = df['Close'].rolling(20).mean()
df['BB_std'] = df['Close'].rolling(20).std()
df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df['ATR'] = pd.concat([
    df['High'] - df['Low'],
    (df['High'] - df['Close'].shift()).abs(),
    (df['Low']  - df['Close'].shift()).abs()
], axis=1).max(axis=1).rolling(14).mean()

df = df.dropna()

# Live Price Metrics
current = live_price if live_price else df['Close'].iloc[-1]
prev    = prev_close if prev_close else df['Close'].iloc[-2]
chg     = current - prev
pct     = (chg / prev) * 100
hi52    = df['Close'].max()
lo52    = df['Close'].min()
vol_ann = df['Volatility'].iloc[-1] * 100

c1, c2, c3, c4, c5 = st.columns(5)
def metric_card(col, label, value, sub='', cls=''):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value {cls}'>{value}</div>
        <div class='metric-label'>{label}</div>
        {'<div style=font-size:0.8rem;color:#888>'+sub+'</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

metric_card(c1, 'Silver / oz', f'${current:.3f}')
metric_card(c2, 'Change', f"{'▲' if chg>=0 else '▼'} {abs(chg):.3f}", f'{pct:+.2f}%', 'up' if chg>=0 else 'down')
metric_card(c3, '52-Week High', f'${hi52:.2f}')
metric_card(c4, '52-Week Low',  f'${lo52:.2f}')
metric_card(c5, 'Annual Volatility', f'{vol_ann:.1f}%')

# Forecasting Models
close = df['Close'].values
n = len(close)
x = np.arange(n)
x_future = np.arange(n, n + forecast_days)
future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')

# 1. Linear Regression
lin_coef = np.polyfit(x, close, 1)
lin_pred  = np.polyval(lin_coef, x_future)

# 2. Polynomial Regression (degree 3)
poly_coef = np.polyfit(x, close, 3)
poly_pred = np.polyval(poly_coef, x_future)

# 3. Moving Average Extrapolation
ma_slope = (df['MA20'].iloc[-1] - df['MA20'].iloc[-21]) / 20 if n > 21 else 0
ma_pred  = np.array([df['MA20'].iloc[-1] + ma_slope * (i+1) for i in range(forecast_days)])

# 4. Exponential Smoothing (Holt's)
alpha, beta = 0.3, 0.1
level, trend_val = close[0], close[1] - close[0]
for v in close[1:]:
    prev_l = level
    level = alpha * v + (1 - alpha) * (level + trend_val)
    trend_val = beta * (level - prev_l) + (1 - beta) * trend_val
exp_pred = np.array([level + trend_val * (i+1) for i in range(forecast_days)])

# 5. Momentum Forecast
mom_period = min(10, n//4)
momentum = (close[-1] - close[-mom_period]) / mom_period
mom_pred  = np.array([close[-1] + momentum*(i+1) for i in range(forecast_days)])

# Ensemble
ensemble = (lin_pred + poly_pred + ma_pred + exp_pred + mom_pred) / 5

# Confidence interval
vol_daily = df['Returns'].std()
sigma = vol_daily * current
ci_upper = ensemble + 1.645 * sigma * np.sqrt(np.arange(1, forecast_days+1))
ci_lower = ensemble - 1.645 * sigma * np.sqrt(np.arange(1, forecast_days+1))

# Main Price Chart
st.markdown('### 📊 Price Chart with Technical Indicators')
tab1, tab2, tab3 = st.tabs(['📈 Candlestick + Predictions', '📉 Technical Indicators', '🔮 Forecast Detail'])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#00d26a', decreasing_line_color='#ff4d6d',
        name='Silver'), row=1, col=1)
    for ma, color, name in [('MA20','#f5a623','MA 20'),('MA50','#7b68ee','MA 50'),('EMA12','#00ced1','EMA 12')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], line=dict(color=color,width=1.2),name=name), row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='rgba(192,192,192,0.3)',width=1),name='BB Upper'), row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='rgba(192,192,192,0.3)',width=1),fill='tonexty',fillcolor='rgba(192,192,192,0.05)',name='BB Lower'), row=1,col=1)
    fig.add_trace(go.Scatter(
        x=list(future_dates), y=ensemble,
        line=dict(color='#c0c0c0', width=2.5, dash='dash'),
        name='Ensemble Forecast'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates[::-1]),
        y=list(ci_upper)+list(ci_lower[::-1]),
        fill='toself', fillcolor='rgba(192,192,192,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% CI'), row=1, col=1)
    colors_vol = ['#00d26a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4d6d' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Volume', opacity=0.7), row=2, col=1)
    fig.update_layout(
        template='plotly_dark', height=600, showlegend=True,
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10))
    )
    fig.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=('RSI (14)', 'MACD', 'ATR (14)'))
    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#7b68ee',width=1.5), name='RSI'), row=1,col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor='rgba(255,77,109,0.1)', line_width=0, row=1,col=1)
    fig2.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,210,106,0.1)', line_width=0, row=1,col=1)
    fig2.add_hline(y=70, line_dash='dash', line_color='#ff4d6d', row=1,col=1)
    fig2.add_hline(y=30, line_dash='dash', line_color='#00d26a', row=1,col=1)
    macd_colors = ['#00d26a' if v >= 0 else '#ff4d6d' for v in df['MACD'] - df['Signal']]
    fig2.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal'], marker_color=macd_colors, name='Histogram'), row=2,col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'],   line=dict(color='#00ced1',width=1.2), name='MACD'),   row=2,col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#f5a623',width=1.2), name='Signal'), row=2,col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df['ATR'], line=dict(color='#c0c0c0',width=1.5), name='ATR'), row=3,col=1)
    fig2.update_layout(template='plotly_dark', height=550, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig2.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = go.Figure()
    for pred, name, color, dash in [
        (lin_pred,  'Linear Regression',      '#f5a623', 'dot'),
        (poly_pred, 'Polynomial (deg 3)',      '#7b68ee', 'dot'),
        (ma_pred,   'MA Extrapolation',        '#00ced1', 'dot'),
        (exp_pred,  'Exponential Smoothing',   '#00d26a', 'dot'),
        (mom_pred,  'Momentum',                '#ff9f40', 'dot'),
        (ensemble,  'Ensemble (Average)',      '#c0c0c0', 'solid'),
    ]:
        fig3.add_trace(go.Scatter(
            x=list(future_dates), y=pred,
            line=dict(color=color, width=2.5 if name.startswith('Ensemble') else 1.5, dash=dash),
            name=name))
    fig3.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates[::-1]),
        y=list(ci_upper)+list(ci_lower[::-1]),
        fill='toself', fillcolor='rgba(192,192,192,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='90% CI'))
    fig3.add_hline(y=current, line_dash='dash', line_color='#888',
                   annotation_text=f'Current: ${current:.2f}')
    fig3.update_layout(
        template='plotly_dark', height=450,
        title='All Forecast Models Comparison',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        yaxis_gridcolor='#1e2130')
    st.plotly_chart(fig3, use_container_width=True)

# Prediction Table
st.markdown('### 🗓️ Day-by-Day Price Forecast')
pred_df = pd.DataFrame({
    'Date':                future_dates.strftime('%a %b %d, %Y'),
    'Linear ($)':          [f'${v:.2f}' for v in lin_pred],
    'Polynomial ($)':      [f'${v:.2f}' for v in poly_pred],
    'MA Extrap ($)':       [f'${v:.2f}' for v in ma_pred],
    'Exp Smooth ($)':      [f'${v:.2f}' for v in exp_pred],
    'Momentum ($)':        [f'${v:.2f}' for v in mom_pred],
    'Ensemble ($)':        [f'${v:.2f}' for v in ensemble],
    'Upper CI ($)':        [f'${v:.2f}' for v in ci_upper],
    'Lower CI ($)':        [f'${v:.2f}' for v in ci_lower],
    'Change from Now (%)': [f'{((v-current)/current)*100:+.2f}%' for v in ensemble],
})
st.dataframe(pred_df.set_index('Date'), use_container_width=True)

# Signal Analysis
st.markdown('### 🎯 Technical Signal Dashboard')
rsi_now  = df['RSI'].iloc[-1]
macd_now = df['MACD'].iloc[-1]
sig_now  = df['Signal'].iloc[-1]
ma5_now  = df['MA5'].iloc[-1]
ma20_now = df['MA20'].iloc[-1]
ma50_now = df['MA50'].iloc[-1]

signals = []
signals.append(('RSI Signal',    'Oversold — Bullish 🟢' if rsi_now < 35 else ('Overbought — Bearish 🔴' if rsi_now > 65 else 'Neutral ⚪'), f'{rsi_now:.1f}'))
signals.append(('MACD Cross',    'Bullish Cross 🟢' if macd_now > sig_now else 'Bearish Cross 🔴', f'MACD {macd_now:.3f}'))
signals.append(('MA Trend',      'Uptrend 🟢' if ma5_now > ma20_now > ma50_now else ('Downtrend 🔴' if ma5_now < ma20_now < ma50_now else 'Mixed ⚪'), f"MA5 {'>' if ma5_now>ma20_now else '<'} MA20"))
signals.append(('BB Position',   'Near Upper Band 🔴' if current > df['BB_upper'].iloc[-1]*0.98 else ('Near Lower Band 🟢' if current < df['BB_lower'].iloc[-1]*1.02 else 'Mid Band ⚪'), f'${df["BB_mid"].iloc[-1]:.2f} mid'))
signals.append(('Forecast Bias', '📈 Bullish' if ensemble[-1] > current else '📉 Bearish', f'Target ${ensemble[-1]:.2f}'))

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
for col, (name, signal, value) in zip([sc1,sc2,sc3,sc4,sc5], signals):
    col.markdown(f"""
    <div class='pred-card'>
        <div style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:1px;'>{name}</div>
        <div style='font-size:0.95rem;font-weight:600;color:#c0c0c0;margin:8px 0;'>{signal}</div>
        <div style='font-size:0.75rem;color:#666;'>{value}</div>
    </div>""", unsafe_allow_html=True)

# Correlation with Related Assets
st.markdown('### 📡 Correlation with Related Assets')

@st.cache_data(ttl=600)
def get_correlation_data(period):
    tickers = {'Gold (GC=F)':'GC=F','USD Index':'DX-Y.NYB',
               'S&P 500':'%5EGSPC','Copper (HG=F)':'HG=F','Oil (CL=F)':'CL=F'}
    data = {}
    for name, sym in tickers.items():
        try:
            tmp = yf.Ticker(sym).history(period=period)
            if not tmp.empty:
                data[name] = tmp['Close'].pct_change().dropna()
        except:
            pass
    return data

corr_data = get_correlation_data(period)
silver_ret = df['Returns'].dropna()
corr_results = {}
for name, ret in corr_data.items():
    aligned = pd.concat([silver_ret, ret], axis=1, join='inner')
    if len(aligned) > 10:
        c = aligned.corr().iloc[0, 1]
        corr_results[name] = round(c, 3)

if corr_results:
    corr_df = pd.DataFrame(list(corr_results.items()), columns=['Asset', 'Correlation with Silver'])
    corr_df = corr_df.sort_values('Correlation with Silver', ascending=False)
    fig_corr = px.bar(corr_df, x='Asset', y='Correlation with Silver',
                      color='Correlation with Silver', color_continuous_scale='RdYlGn',
                      title='Silver Correlation with Key Assets', range_color=[-1, 1])
    fig_corr.update_layout(template='plotly_dark', height=350,
                           paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                           yaxis_gridcolor='#1e2130')
    st.plotly_chart(fig_corr, use_container_width=True)

# Returns Distribution
st.markdown('### 📊 Returns Analysis & Risk Metrics')
col_a, col_b = st.columns(2)
with col_a:
    fig_hist = px.histogram(df['Returns'].dropna()*100, nbins=60,
                            title='Daily Returns Distribution (%)',
                            color_discrete_sequence=['#c0c0c0'])
    fig_hist.add_vline(x=0, line_dash='dash', line_color='white')
    fig_hist.update_layout(template='plotly_dark', height=300,
                           paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                           showlegend=False, xaxis_title='Return (%)', yaxis_title='Frequency')
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    ret_clean = df['Returns'].dropna()
    sharpe = (ret_clean.mean() / ret_clean.std()) * np.sqrt(252)
    var_95  = np.percentile(ret_clean, 5) * 100
    max_dd  = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
    avg_ret = ret_clean.mean() * 252 * 100
    risk_df = pd.DataFrame({
        'Metric': ['Annualized Return','Sharpe Ratio','VaR 95% (Daily)','Max Drawdown','Annual Volatility'],
        'Value':  [f'{avg_ret:.1f}%', f'{sharpe:.2f}', f'{var_95:.2f}%', f'{max_dd:.1f}%', f'{vol_ann:.1f}%']
    })
    st.dataframe(risk_df.set_index('Metric'), use_container_width=True, height=230)

# Footer
st.markdown('---')
st.markdown(
    '<div style=text-align:center;color:#555;font-size:0.75rem;>🥈 Silver Price Predictor · Data: Yahoo Finance · '
    'Forecasting: Linear, Polynomial, MA, Exponential Smoothing, Momentum Ensemble · '
    '⚠️ Not financial advice. Past performance does not equal future results.</div>',
    unsafe_allow_html=True
)
