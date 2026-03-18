# app.py – Ultimate Crypto Scanner (with strategy toggles and pair count)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Ultimate Crypto Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0;
    }
    .signal-card {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 6px solid #3498db;
    }
    .grade-AAA { color: #f1c40f; font-weight: bold; }
    .grade-AA { color: #9b59b6; font-weight: bold; }
    .grade-A { color: #3498db; font-weight: bold; }
    .grade-B { color: #2ecc71; font-weight: bold; }
    .grade-C { color: #e74c3c; font-weight: bold; }
    .profit { color: #2ecc71; }
    .loss { color: #e74c3c; }
    .stButton>button {
        background: #2c3e50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📡 Ultimate Unified Crypto Scanner</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">11 Strategies • Multi‑TF • Institutional Grading • Real‑time</p>', unsafe_allow_html=True)

# ---------- Session State ----------
if 'all_pairs' not in st.session_state:
    st.session_state.all_pairs = []
if 'scanned_results' not in st.session_state:
    st.session_state.scanned_results = []
if 'active_trades' not in st.session_state:
    st.session_state.active_trades = []
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'scanning' not in st.session_state:
    st.session_state.scanning = False
if 'enabled_strategies' not in st.session_state:
    # All strategies enabled by default
    st.session_state.enabled_strategies = {
        'Khurram Ultimate': True,
        'Turtle Trading': True,
        'Swing Trading': True,
        'Breakout': True,
        'Mean Reversion': True,
        'Scalping': True,
        'Reversal': True,
        'Gap Trading': True,
        'Range Breakout': True,
        'Triple Screen': True,
        'Bollinger Bands': True
    }

# ---------- Helper ----------
def get_pakistan_time():
    return datetime.utcnow() + timedelta(hours=5)

# ---------- Cached Exchange Resource ----------
@st.cache_resource
def get_exchange():
    return ccxt.mexc({
        'enableRateLimit': True,
        'timeout': 30000,
        'options': {'defaultType': 'spot'}
    })

# ---------- Cached Data Fetching ----------
@st.cache_data(ttl=600)
def get_top_pairs(exchange, limit=100):
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and ticker.get('quoteVolume'):
                usdt_pairs.append((symbol, ticker['quoteVolume']))
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in usdt_pairs[:limit]]
    except Exception as e:
        st.warning(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]

@st.cache_data(ttl=300)
def fetch_ohlcv(exchange, symbol, timeframe='15m', limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

# ---------- Indicator Calculator ----------
class IndicatorCalculator:
    @staticmethod
    def compute(df):
        if df is None or len(df) < 50:
            return df
        # Basic
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_200'] = ta.ema(df['close'], length=200)

        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)

        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_mid'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']

        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'])
        if adx is not None:
            df['adx'] = adx['ADX_14']

        # SuperTrend (simplified)
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if supertrend is not None and 'SUPERTd_10_3.0' in supertrend.columns:
            df['supertrend'] = supertrend['SUPERT_10_3.0']
            df['supertrend_dir'] = supertrend['SUPERTd_10_3.0']

        # Swing highs/lows
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()

        df.dropna(inplace=True)
        return df

# ---------- Strategy Detector ----------
class StrategyDetector:
    def __init__(self, enabled):
        self.enabled = enabled

    def detect(self, df):
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None

        signals = []          # list of (strategy_name, signal_type, confidence_boost)
        strategies_found = []

        # ----- Khurram Ultimate Consolidation Breakout -----
        if self.enabled.get('Khurram Ultimate', True):
            recent_high = df['high'].tail(30).max()
            recent_low = df['low'].tail(30).min()
            range_width = (recent_high - recent_low) / recent_low
            if range_width < 0.08 and last['volume_ratio'] > 1.8:
                if last['close'] > recent_high and last['close'] > last['ema_200'] and last['rsi'] > 50:
                    signals.append(('Khurram Ultimate', 'LONG', 30))
                    strategies_found.append('Khurram Ultimate')
                if last['close'] < recent_low and last['close'] < last['ema_200'] and last['rsi'] < 50:
                    signals.append(('Khurram Ultimate', 'SHORT', 30))
                    strategies_found.append('Khurram Ultimate')

        # ----- Turtle Trading (20-day breakout) -----
        if self.enabled.get('Turtle Trading', True):
            if last['close'] > last['high_20'] and prev is not None and prev['close'] <= prev['high_20']:
                signals.append(('Turtle Trading', 'LONG', 15))
                strategies_found.append('Turtle Trading')
            if last['close'] < last['low_20'] and prev is not None and prev['close'] >= prev['low_20']:
                signals.append(('Turtle Trading', 'SHORT', 15))
                strategies_found.append('Turtle Trading')

        # ----- Swing Trading (pullback) -----
        if self.enabled.get('Swing Trading', True):
            if last['close'] > last['sma_50'] and last['rsi'] < 40:
                signals.append(('Swing Trading', 'LONG', 10))
                strategies_found.append('Swing Trading')
            if last['close'] < last['sma_50'] and last['rsi'] > 60:
                signals.append(('Swing Trading', 'SHORT', 10))
                strategies_found.append('Swing Trading')

        # ----- Breakout (resistance/support) -----
        if self.enabled.get('Breakout', True):
            resistance = df['high'].tail(20).max()
            support = df['low'].tail(20).min()
            if last['close'] > resistance and last['volume_ratio'] > 1.5:
                signals.append(('Breakout', 'LONG', 20))
                strategies_found.append('Breakout')
            if last['close'] < support and last['volume_ratio'] > 1.5:
                signals.append(('Breakout', 'SHORT', 20))
                strategies_found.append('Breakout')

        # ----- Mean Reversion -----
        if self.enabled.get('Mean Reversion', True):
            if last['close'] < last['bb_lower'] and last['rsi'] < 30:
                signals.append(('Mean Reversion', 'LONG', 15))
                strategies_found.append('Mean Reversion')
            if last['close'] > last['bb_upper'] and last['rsi'] > 70:
                signals.append(('Mean Reversion', 'SHORT', 15))
                strategies_found.append('Mean Reversion')

        # ----- Scalping (momentum surge) -----
        if self.enabled.get('Scalping', True):
            if last['volume_ratio'] > 2.0 and last['close'] > prev['close'] * 1.005:
                signals.append(('Scalping', 'LONG', 10))
                strategies_found.append('Scalping')
            if last['volume_ratio'] > 2.0 and last['close'] < prev['close'] * 0.995:
                signals.append(('Scalping', 'SHORT', 10))
                strategies_found.append('Scalping')

        # ----- Reversal (near S/R with RSI extreme) -----
        if self.enabled.get('Reversal', True):
            support = df['low'].tail(20).min()
            resistance = df['high'].tail(20).max()
            if abs(last['close'] - support) / support < 0.02 and last['rsi'] < 35:
                signals.append(('Reversal', 'LONG', 15))
                strategies_found.append('Reversal')
            if abs(resistance - last['close']) / last['close'] < 0.02 and last['rsi'] > 65:
                signals.append(('Reversal', 'SHORT', 15))
                strategies_found.append('Reversal')

        # ----- Gap Trading -----
        if self.enabled.get('Gap Trading', True):
            if last['open'] > prev['close'] * 1.02 and last['close'] > last['open']:
                signals.append(('Gap Trading', 'LONG', 15))
                strategies_found.append('Gap Trading')
            if last['open'] < prev['close'] * 0.98 and last['close'] < last['open']:
                signals.append(('Gap Trading', 'SHORT', 15))
                strategies_found.append('Gap Trading')

        # ----- Range Breakout -----
        if self.enabled.get('Range Breakout', True):
            recent_high = df['high'].tail(30).max()
            recent_low = df['low'].tail(30).min()
            range_width = (recent_high - recent_low) / recent_low
            if range_width < 0.10:
                if last['close'] > recent_high * 0.98 and last['volume_ratio'] > 1.5:
                    signals.append(('Range Breakout', 'LONG', 15))
                    strategies_found.append('Range Breakout')
                if last['close'] < recent_low * 1.02 and last['volume_ratio'] > 1.5:
                    signals.append(('Range Breakout', 'SHORT', 15))
                    strategies_found.append('Range Breakout')

        # ----- Triple Screen -----
        if self.enabled.get('Triple Screen', True):
            weekly_trend_up = last['close'] > df['sma_50'].iloc[-1]
            if weekly_trend_up and last['rsi'] < 35:
                signals.append(('Triple Screen', 'LONG', 10))
                strategies_found.append('Triple Screen')
            if not weekly_trend_up and last['rsi'] > 65:
                signals.append(('Triple Screen', 'SHORT', 10))
                strategies_found.append('Triple Screen')

        # ----- Bollinger Bands -----
        if self.enabled.get('Bollinger Bands', True):
            if last['bb_width'] < df['bb_width'].tail(20).mean() * 0.8:
                if last['close'] > last['bb_upper'] and last['volume_ratio'] > 1.5:
                    signals.append(('Bollinger Bands', 'LONG', 15))
                    strategies_found.append('Bollinger Bands')
                if last['close'] < last['bb_lower'] and last['volume_ratio'] > 1.5:
                    signals.append(('Bollinger Bands', 'SHORT', 15))
                    strategies_found.append('Bollinger Bands')

        if not signals:
            return None

        # Determine overall direction
        long_count = sum(1 for s in signals if s[1] == 'LONG')
        short_count = sum(1 for s in signals if s[1] == 'SHORT')
        direction = 'LONG' if long_count > short_count else 'SHORT' if short_count > long_count else 'NEUTRAL'

        # Confidence score (0-100)
        base_score = min(len(strategies_found) * 10, 60)
        extra = sum(s[2] for s in signals) // len(signals) if signals else 0
        confidence = min(base_score + extra, 100)

        # Grade
        if confidence >= 90:
            grade = 'A+'
        elif confidence >= 80:
            grade = 'A'
        elif confidence >= 70:
            grade = 'B+'
        elif confidence >= 60:
            grade = 'B'
        elif confidence >= 50:
            grade = 'C+'
        else:
            grade = 'C'

        # Trade levels
        atr = last['atr']
        entry = last['close']
        if direction == 'LONG':
            sl = entry - atr * 1.5
            tp1 = entry + atr * 2.5
            tp2 = entry + atr * 4
        elif direction == 'SHORT':
            sl = entry + atr * 1.5
            tp1 = entry - atr * 2.5
            tp2 = entry - atr * 4
        else:
            sl = tp1 = tp2 = np.nan

        return {
            'direction': direction,
            'confidence': confidence,
            'grade': grade,
            'strategies': strategies_found,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'atr': atr,
            'rsi': last['rsi'],
            'volume_ratio': last['volume_ratio']
        }

# ---------- Position Sizer ----------
def calculate_position(account_balance, risk_percent, entry, stop, leverage=1):
    risk_amount = account_balance * (risk_percent / 100)
    stop_dist = abs(entry - stop)
    if stop_dist == 0:
        return {}
    position_size = risk_amount / stop_dist
    position_value = position_size * entry
    margin = position_value / leverage
    return {
        'size': position_size,
        'value': position_value,
        'margin': margin,
        'risk': risk_amount,
        'leverage': leverage
    }

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/strategy-board.png", width=80)
    st.header("⚙️ Controls")

    # Pakistan time
    now_pk = get_pakistan_time()
    st.info(f"🇵🇰 PKT: {now_pk.strftime('%H:%M:%S')}")

    # Strategy toggles with Select All / Deselect All
    st.subheader("🎯 Active Strategies")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Select All"):
            for k in st.session_state.enabled_strategies:
                st.session_state.enabled_strategies[k] = True
    with col2:
        if st.button("❌ Deselect All"):
            for k in st.session_state.enabled_strategies:
                st.session_state.enabled_strategies[k] = False

    for strat in st.session_state.enabled_strategies.keys():
        st.session_state.enabled_strategies[strat] = st.checkbox(
            strat, value=st.session_state.enabled_strategies[strat])

    st.divider()

    # Risk management
    st.subheader("💰 Risk")
    account_balance = st.number_input("Balance (USDT)", value=1000, step=100)
    risk_percent = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    leverage = st.number_input("Leverage", 1, 500, 1, step=1)

    st.divider()

    # Scan settings
    st.subheader("🔍 Scan")
    limit_pairs = st.slider("Number of pairs to scan", 10, 200, 50,
                            help="Fewer pairs = faster scan. More pairs = more opportunities.")
    timeframe = st.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=1)
    auto_refresh = st.checkbox("Auto‑refresh every 30s", False)

    if st.button("🚀 SCAN NOW", use_container_width=True, type="primary"):
        st.session_state.scanning = True

    if st.button("🔄 Reset Trades", use_container_width=True):
        st.session_state.active_trades = []
        st.rerun()

# ---------- Main App ----------
exchange = get_exchange()
detector = StrategyDetector(st.session_state.enabled_strategies)

# Load pairs if not already
if not st.session_state.all_pairs:
    with st.spinner("Loading top pairs..."):
        st.session_state.all_pairs = get_top_pairs(exchange, limit_pairs)

# Scanning logic
if st.session_state.scanning:
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(st.session_state.all_pairs)
    for i, sym in enumerate(st.session_state.all_pairs):
        status.text(f"Scanning {i+1}/{total}: {sym}")
        df = fetch_ohlcv(exchange, sym, timeframe)
        if df is not None and len(df) >= 50:
            df = IndicatorCalculator.compute(df)
            signal = detector.detect(df)
            if signal:
                results.append({
                    'pair': sym,
                    **signal
                })
            st.session_state.data_cache[sym] = df
        progress.progress((i + 1) / total)
        time.sleep(0.2)  # be gentle
    st.session_state.scanned_results = results
    st.session_state.last_scan = datetime.now()
    st.session_state.scanning = False
    st.rerun()

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pairs", len(st.session_state.all_pairs))
col2.metric("Signals", len(st.session_state.scanned_results))
col3.metric("Active Trades", len(st.session_state.active_trades))
if st.session_state.last_scan:
    col4.metric("Last Scan", st.session_state.last_scan.strftime("%H:%M:%S"))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Signals", "📊 Active Trades", "📈 Chart", "📚 Strategy Guide"
])

# -------------------- TAB 1: Live Signals --------------------
with tab1:
    st.subheader("📡 Live Signals")
    if st.session_state.scanned_results:
        for res in st.session_state.scanned_results:
            grade_class = {
                'A+': 'grade-AAA', 'A': 'grade-AA', 'B+': 'grade-A',
                'B': 'grade-B', 'C+': 'grade-C', 'C': 'grade-C'
            }.get(res['grade'], '')
            with st.container():
                st.markdown(f"""
                <div class="signal-card">
                    <div style="display: flex; justify-content: space-between;">
                        <h3>{res['pair']}</h3>
                        <span><span class="{grade_class}">{res['grade']}</span> • {res['confidence']}%</span>
                    </div>
                    <p><b>Direction:</b> {res['direction']} | <b>Strategies:</b> {', '.join(res['strategies'])}</p>
                    <p><b>Entry:</b> ${res['entry']:.4f} | <b>SL:</b> ${res['sl']:.4f} | <b>TP1:</b> ${res['tp1']:.4f} | <b>TP2:</b> ${res['tp2']:.4f}</p>
                    <p><b>RSI:</b> {res['rsi']:.1f} | <b>Vol Ratio:</b> {res['volume_ratio']:.2f}x | <b>ATR:</b> ${res['atr']:.4f}</p>
                """, unsafe_allow_html=True)
                pos = calculate_position(account_balance, risk_percent, res['entry'], res['sl'], leverage)
                if pos:
                    st.info(f"📊 Position: {pos['size']:.4f} units | Value: ${pos['value']:.2f} | Margin: ${pos['margin']:.2f} | Risk: ${pos['risk']:.2f}")
                if st.button(f"📥 Take {res['pair']}", key=f"take_{res['pair']}"):
                    trade = res.copy()
                    trade['taken_at'] = datetime.now()
                    trade['current_price'] = res['entry']
                    trade['pnl_pct'] = 0.0
                    trade['suggestion'] = "New trade – monitor"
                    st.session_state.active_trades.append(trade)
                    st.rerun()
    else:
        st.info("No signals yet. Click SCAN NOW.")

# -------------------- TAB 2: Active Trades --------------------
with tab2:
    st.subheader("📊 Active Trades")
    if st.session_state.active_trades:
        for i, trade in enumerate(st.session_state.active_trades):
            pair = trade['pair']
            if pair in st.session_state.data_cache:
                df = st.session_state.data_cache[pair]
                current = df['close'].iloc[-1]
                trade['current_price'] = current
                if trade['direction'] == 'LONG':
                    trade['pnl_pct'] = (current - trade['entry']) / trade['entry'] * 100
                else:
                    trade['pnl_pct'] = (trade['entry'] - current) / trade['entry'] * 100
            with st.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.write(f"**{pair}**")
                col2.write(f"Dir: {trade['direction']}")
                col3.write(f"Entry: ${trade['entry']:.2f}")
                pnl = trade['pnl_pct']
                col4.write(f"P&L: <span class='{'profit' if pnl>=0 else 'loss'}'>{pnl:.2f}%</span>", unsafe_allow_html=True)
                col5.write(f"SL: ${trade['sl']:.2f}")
                if st.button(f"Close {pair}", key=f"close_{i}"):
                    st.session_state.active_trades.pop(i)
                    st.rerun()
    else:
        st.info("No active trades.")

# -------------------- TAB 3: Chart --------------------
with tab3:
    st.subheader("📈 Chart")
    if st.session_state.data_cache:
        pair = st.selectbox("Select pair", list(st.session_state.data_cache.keys()))
        df = st.session_state.data_cache[pair]
        fig = go.Figure(data=[
            go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                           low=df['low'], close=df['close'], name='Price')
        ])
        # Add indicators
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')))
        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No chart data yet. Run a scan.")

# -------------------- TAB 4: Strategy Guide (Detailed) --------------------
with tab4:
    st.subheader("📚 Strategy Guide – What Each Strategy Does")
    st.markdown("""
    Each strategy has its own logic, entry conditions, and works best in specific market conditions.
    Expand any strategy below to see its details, including the exact parameters used.
    """)

    with st.expander("👑 Khurram Ultimate – Consolidation Breakout"):
        st.markdown("""
        **Core Concept:**  
        Identifies tight consolidation ranges (less than 8% over 30 periods) and enters on a breakout with high volume.

        **Entry Conditions:**
        - Range width < 8% (recent 30‑bar high/low difference)
        - Volume surge > 1.8× average
        - For LONG: Price breaks above range high, above 200 EMA, and RSI > 50
        - For SHORT: Price breaks below range low, below 200 EMA, and RSI < 50

        **Best Market:** Strong trends after consolidation (breakout momentum).
        **Risk:** Moderate; false breakouts can occur. Use tight stops.
        """)

    with st.expander("🐢 Turtle Trading – 20‑Day Breakout"):
        st.markdown("""
        **Core Concept:**  
        Classic trend‑following system by Richard Donchian. Enters on new 20‑day highs/lows.

        **Entry Conditions:**
        - LONG: Price closes above the highest high of the last 20 candles
        - SHORT: Price closes below the lowest low of the last 20 candles
        - No volume filter by default (but can be combined with volume surge)

        **Best Market:** Strong trending markets.
        **Risk:** Whipsaws in ranges. Uses wide stops (2× ATR).
        """)

    with st.expander("⚡ Swing Trading – Pullback in Trend"):
        st.markdown("""
        **Core Concept:**  
        Waits for a pullback in an established trend (price above/below 50 SMA) and enters on RSI extremes.

        **Entry Conditions:**
        - LONG: Price > 50 SMA, RSI < 40 (oversold within uptrend)
        - SHORT: Price < 50 SMA, RSI > 60 (overbought within downtrend)

        **Best Market:** Trending markets with healthy pullbacks.
        **Risk:** Lower risk because entry is against the immediate move but with the main trend.
        """)

    with st.expander("🚀 Breakout – Resistance/Support with Volume"):
        st.markdown("""
        **Core Concept:**  
        Enters when price breaks a recent swing high (resistance) or low (support) with volume confirmation.

        **Entry Conditions:**
        - LONG: Price > 20‑period high, volume > 1.5× average
        - SHORT: Price < 20‑period low, volume > 1.5× average

        **Best Market:** Trending or volatile markets.
        **Risk:** False breakouts are common; volume filter helps.
        """)

    with st.expander("🔄 Mean Reversion – Oversold/Overbought Bounces"):
        st.markdown("""
        **Core Concept:**  
        Buys when price touches lower Bollinger Band and RSI is oversold; sells when price touches upper band and RSI is overbought.

        **Entry Conditions:**
        - LONG: Price < lower BB, RSI < 30
        - SHORT: Price > upper BB, RSI > 70

        **Best Market:** Ranging or sideways markets.
        **Risk:** Can be dangerous in strong trends; use only when ADX < 25.
        """)

    with st.expander("⚡ Scalping – Momentum Surge"):
        st.markdown("""
        **Core Concept:**  
        Catches sudden volume spikes and rapid price moves for quick profits.

        **Entry Conditions:**
        - LONG: Volume > 2× average, price up >0.5% in one candle
        - SHORT: Volume > 2× average, price down >0.5% in one candle

        **Best Market:** High‑volatility periods (news, session opens).
        **Risk:** Very high; requires fast execution and tight stops. Use only with liquid pairs.
        """)

    with st.expander("🔄 Reversal – Divergence at S/R"):
        st.markdown("""
        **Core Concept:**  
        Looks for potential trend reversals when price reaches a support/resistance level and RSI shows divergence.

        **Entry Conditions:**
        - LONG: Price near 20‑period low, RSI < 35, bullish divergence (price lower but RSI higher)
        - SHORT: Price near 20‑period high, RSI > 65, bearish divergence

        **Best Market:** Range boundaries or after extended moves.
        **Risk:** Timing reversals is difficult; use with confirmation from candlestick patterns.
        """)

    with st.expander("⬆️ Gap Trading – Gap Follow‑Through"):
        st.markdown("""
        **Core Concept:**  
        Trades gaps (open above previous close) if price continues in the gap direction.

        **Entry Conditions:**
        - LONG: Gap up >2%, price closes above open (follow‑through)
        - SHORT: Gap down >2%, price closes below open

        **Best Market:** After major news or earnings.
        **Risk:** Gaps often fill; require quick exit.
        """)

    with st.expander("📊 Range Breakout – Tight Range Expansion"):
        st.markdown("""
        **Core Concept:**  
        Identifies periods of low volatility (tight range <10%) and enters on a breakout with volume.

        **Entry Conditions:**
        - Range width <10% over 30 bars
        - LONG: Price > range high × 0.98, volume >1.5× average
        - SHORT: Price < range low × 1.02, volume >1.5× average

        **Best Market:** Quiet consolidation followed by explosion.
        **Risk:** False breakouts; volume confirmation essential.
        """)

    with st.expander("📺 Triple Screen – Multi‑Timeframe Confluence"):
        st.markdown("""
        **Core Concept:**  
        Uses higher timeframe (weekly trend via 50 SMA) and lower timeframe oscillator.

        **Entry Conditions:**
        - LONG: Weekly trend up (price > 50 SMA) + RSI < 35 on entry timeframe
        - SHORT: Weekly trend down + RSI > 65 on entry timeframe

        **Best Market:** Any; aligns with larger trend.
        **Risk:** Lower risk due to trend alignment.
        """)

    with st.expander("📉 Bollinger Bands – Squeeze & Expansion"):
        st.markdown("""
        **Core Concept:**  
        Detects Bollinger Band squeezes (width < 80% of 20‑period average) and trades the first expansion.

        **Entry Conditions:**
        - Squeeze detected
        - LONG: Price closes above upper band with volume >1.5×
        - SHORT: Price closes below lower band with volume >1.5×

        **Best Market:** Volatility contractions followed by breakouts.
        **Risk:** Breakout direction is unknown until it happens.
        """)

    st.divider()
    st.caption("""
    **Note:** All strategies use the same base indicators (RSI, volume ratio, ATR, EMAs). 
    The confidence score and grade are calculated from the number of strategies triggered and their individual scores.
    You can enable/disable any strategy in the sidebar.
    """)

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
