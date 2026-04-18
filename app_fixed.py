import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
# FIX #13: Removed unused imports `make_subplots` and `io`

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Bot Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    .main { background-color: #0d1117; }
    .block-container { padding: 1.5rem 2rem; }

    .metric-box {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-box .label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .metric-box .value { font-size: 26px; font-weight: 700; color: #58a6ff; margin-top: 4px; }

    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #1c2333 50%, #0d1117 100%);
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }
    .hero h1 { font-size: 30px; font-weight: 700; color: #f0f6fc; margin: 0 0 6px 0; }
    .hero p  { font-size: 14px; color: #8b949e; margin: 0; }

    .stPlotlyChart { background: transparent !important; }

    h1,h2,h3 { color: #f0f6fc !important; }
    .stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #1c2333 !important; color: #58a6ff !important; }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label { color: #8b949e !important; font-size: 13px; }

    .stButton>button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 8px;
        padding: 8px 22px; font-weight: 600; font-size: 14px;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }

    .info-box {
        background: #12261e;
        border-left: 3px solid #3fb950;
        border-radius: 6px;
        padding: 12px 16px;
        color: #adbac7;
        font-size: 13px;
        margin-bottom: 16px;
    }
    .warn-box {
        background: #2d1b00;
        border-left: 3px solid #e3b341;
        border-radius: 6px;
        padding: 12px 16px;
        color: #adbac7;
        font-size: 13px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#c9d1d9"),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ═════════════════════════════════════════════════════════════════════════════
# BOT 1 — DELTA-GAMMA RISK SURFACE
# ═════════════════════════════════════════════════════════════════════════════
def _bs_d1d2(S, K, T, r, sigma):
    """Shared d1/d2 computation. Raises ValueError on degenerate inputs."""
    denom = sigma * np.sqrt(T)
    if denom == 0:
        raise ValueError(f"Black-Scholes undefined: sigma={sigma}, T={T} (denom=0)")
    if S <= 0 or K <= 0:
        raise ValueError(f"Black-Scholes undefined: S={S}, K={K} must be > 0")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    return d1, d2

def bs_price(S, K, T, r, sigma, option_type='call'):
    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type='call'):
    d1, _ = _bs_d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    d1, _ = _bs_d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# FIX #16: Added bs_vega so PnL surface can account for volatility changes
def bs_vega(S, K, T, r, sigma):
    d1, _ = _bs_d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def portfolio_greeks(options, S, sigma, T, r):
    td, tg, tv = 0.0, 0.0, 0.0
    for o in options:
        try:
            td += bs_delta(S, o['K'], T, r, sigma, o['type']) * o['qty']
            tg += bs_gamma(S, o['K'], T, r, sigma) * o['qty']
            tv += bs_vega(S, o['K'], T, r, sigma) * o['qty']
        except (ValueError, ZeroDivisionError):
            # Degenerate point in surface sweep — skip rather than crash
            pass
    return td, tg, tv

def risk_surface(options, S_range, sig_range, T, r):
    D = np.zeros((len(S_range), len(sig_range)))
    G = np.zeros_like(D)
    for i, S in enumerate(S_range):
        for j, s in enumerate(sig_range):
            D[i, j], G[i, j], _ = portfolio_greeks(options, S, s, T, r)
    return D, G

# FIX #16: PnL now includes the vega term so sigma dimension is meaningful
def pnl_surface(options, S_range, sig_range, T, r, S0, sig0):
    bd, bg, bv = portfolio_greeks(options, S0, sig0, T, r)
    P = np.zeros((len(S_range), len(sig_range)))
    for i, S in enumerate(S_range):
        for j, s in enumerate(sig_range):
            dS     = S - S0
            dSigma = s - sig0
            # Full Taylor: PnL ≈ Δ·dS + ½·Γ·dS² + Vega·dσ
            P[i, j] = bd * dS + 0.5 * bg * dS**2 + bv * dSigma
    return P

def render_delta_gamma_bot():
    st.markdown(
        '<div class="hero"><h1>📐 Delta-Gamma Risk Surface Bot</h1>'
        '<p>Interactive options portfolio greeks and P&L surface visualizer powered by Black-Scholes</p></div>',
        unsafe_allow_html=True
    )

    col_ctrl, col_chart = st.columns([1, 2.5], gap="large")

    with col_ctrl:
        st.markdown("### ⚙️ Market Parameters")
        S0   = st.number_input("Spot Price (S₀)", 50.0, 300.0, 100.0, 1.0)
        sig0 = st.slider("Base Implied Vol (σ₀)", 0.05, 0.60, 0.20, 0.01, format="%.2f")
        T    = st.slider("Time to Expiry (years)", 0.05, 2.0, 0.5, 0.05, format="%.2f")
        r    = st.number_input("Risk-Free Rate", 0.0, 0.15, 0.01, 0.005, format="%.3f")

        st.markdown("### 📋 Portfolio")
        st.markdown(
            '<div class="info-box">Add up to 5 options. Negative qty = short position.</div>',
            unsafe_allow_html=True
        )

        n_opts = st.number_input("Number of Legs", min_value=1, max_value=5, value=3, step=1)
        # BUG-13 FIX: number_input can return float or None on first run in older Streamlit.
        # Clamp with max(1, ...) to prevent range(0) producing no legs silently.
        n_opts = max(1, int(n_opts or 1))
        default_strikes = [100.0, 95.0, 105.0, 110.0, 90.0]
        default_qtys    = [2, 1, -1, 1, -1]
        options = []
        for i in range(n_opts):
            with st.expander(f"Leg {i+1}", expanded=(i < 3)):
                c1, c2, c3 = st.columns(3)
                otype = c1.selectbox("Type", ["call", "put"], key=f"t{i}")
                # FIX #9 (minor): removed dead `else 100.0`; i is always < 5
                K   = c2.number_input("Strike K", 50.0, 300.0, default_strikes[i], 1.0, key=f"k{i}")
                qty = c3.number_input("Qty", -10, 10, default_qtys[i], 1, key=f"q{i}")
                options.append({'type': otype, 'K': K, 'qty': qty})

        st.markdown("### 🔭 Surface Grid")
        c1, c2 = st.columns(2)

        # BUG-3 FIX: Ensure S_lo < S_hi with a guaranteed gap of at least 2 points
        # s_lo_max must be strictly less than s_hi_min (= S0+1), so cap at S0-1 but floor at 51
        s_lo_max = max(S0 - 1.0, 51.0)
        s_lo_default = float(np.clip(S0 * 0.8, 50.0, s_lo_max))
        # Guard: if S0 is at its minimum (50), widget bounds collapse — widen artificially
        if s_lo_max <= 50.0:
            s_lo_max = 51.0
            s_lo_default = 50.0
        S_lo = c1.number_input("S min", 50.0, s_lo_max, s_lo_default, 1.0)

        s_hi_min = S0 + 1.0
        s_hi_default = float(np.clip(S0 * 1.2, s_hi_min, 400.0))
        S_hi = c2.number_input("S max", s_hi_min, 400.0, s_hi_default, 1.0)

        # v_lo: clamp default so it is always < sig0
        v_lo_max = max(round(sig0 - 0.01, 2), 0.02)
        v_lo_default = float(np.clip(0.10, 0.01, v_lo_max))
        v_lo = c1.number_input("σ min", 0.01, v_lo_max, v_lo_default, 0.01, format="%.2f")

        # v_hi: clamp default so it is always > sig0
        v_hi_min = round(sig0 + 0.01, 2)
        v_hi_default = float(np.clip(0.40, v_hi_min, 0.99))
        v_hi = c2.number_input("σ max", v_hi_min, 0.99, v_hi_default, 0.01, format="%.2f")

        n_pts = st.slider("Grid Resolution", 20, 80, 40)
        # FIX #1: Removed `if run or True:` — the run button is now the actual gate.
        # Bot 1 computes on every interaction since it's fast and stateless.
        st.button("🚀 Compute Surfaces", use_container_width=True)  # visual only; always renders

    with col_chart:
        # FIX #1: Always render (the computation is lightweight). Button is cosmetic here.
        # If you need gating, wrap in st.session_state instead of `or True`.
        S_range   = np.linspace(S_lo, S_hi, n_pts)
        sig_range = np.linspace(v_lo, v_hi, n_pts)
        X, Y      = np.meshgrid(S_range, sig_range, indexing='ij')
        D, G      = risk_surface(options, S_range, sig_range, T, r)
        PnL       = pnl_surface(options, S_range, sig_range, T, r, S0, sig0)

        bd, bg, bv = portfolio_greeks(options, S0, sig0, T, r)
        # FIX #11: Removed unused variable `bp`

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-box"><div class="label">Portfolio Δ</div><div class="value">{bd:.3f}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><div class="label">Portfolio Γ</div><div class="value">{bg:.4f}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-box"><div class="label">Portfolio Vega</div><div class="value">{bv:.3f}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-box"><div class="label">Base Vol</div><div class="value">{sig0*100:.0f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📈 Delta Surface", "📉 Gamma Surface", "💰 PnL Surface"])

        # make_3d defined at render time (uses no closure vars — safe to inline here)
        def make_3d(Xg, Yg, Z, title, cmap):
            fig = go.Figure(go.Surface(
                x=Xg, y=Yg, z=Z,
                colorscale=cmap, showscale=True, opacity=0.93,
                contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True))
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=title, font=dict(size=16, color="#f0f6fc")),
                scene=dict(
                    xaxis=dict(title="Spot Price", gridcolor="#21262d", backgroundcolor="#0d1117"),
                    yaxis=dict(title="Implied Vol",  gridcolor="#21262d", backgroundcolor="#0d1117"),
                    zaxis=dict(gridcolor="#21262d",   backgroundcolor="#0d1117"),
                    bgcolor="#0d1117"
                ),
                height=480
            )
            return fig

        with tab1:
            st.plotly_chart(make_3d(X, Y, D,   "Delta Surface",                      "Plasma"), use_container_width=True)
        with tab2:
            st.plotly_chart(make_3d(X, Y, G,   "Gamma Surface",                      "Cividis"), use_container_width=True)
        with tab3:
            st.plotly_chart(make_3d(X, Y, PnL, "P&L Surface (Δ·dS + ½Γ·dS² + V·dσ)", "Magma"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# BOT 2 — LATENT REGIME DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def render_regime_bot():
    st.markdown(
        '<div class="hero"><h1>🔍 Latent Regime Detection Bot</h1>'
        '<p>Hidden Markov Model identifies latent market states — bull, bear, or sideways — from historical price data</p></div>',
        unsafe_allow_html=True
    )

    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    from collections import Counter  # FIX #14: moved import to top of function

    def compute_features(df, window=20):
        if len(df) < window + 2:
            raise ValueError(
                f"Not enough data: need at least {window + 2} rows for window={window}, "
                f"got {len(df)}. Reduce the Feature Window or load more data."
            )
        f = pd.DataFrame(index=df.index)
        f['returns']    = np.log(df['Close']).diff()
        f['volatility'] = f['returns'].rolling(window).std()
        f['momentum']   = df['Close'].pct_change(window)
        result = f.dropna()
        if result.empty:
            raise ValueError("Feature computation produced no valid rows. Check your data for gaps or constant prices.")
        return result

    def fit_hmm(features, n_states, seed):
        raw    = features[['returns', 'volatility', 'momentum']].values
        scaler = StandardScaler()
        X      = scaler.fit_transform(raw)
        model  = GaussianHMM(n_components=n_states, covariance_type="diag",
                             n_iter=1000, random_state=seed)
        model.fit(X)
        states      = model.predict(X)
        state_probs = model.predict_proba(X)
        return states, state_probs, model.transmat_, model, scaler

    # FIX #10: Two HMM states can legitimately share a regime label.
    # Disambiguate by appending a suffix when collisions occur.
    def map_regimes(features, states):
        regime_map    = {}
        label_counts  = Counter()
        for s in np.unique(states):
            mask = states == s
            mr   = features['returns'][mask].mean()
            mv   = features['volatility'][mask].mean()
            med  = features['volatility'].quantile(0.4)
            if mv < med:
                base = 'sideways'
            elif mr < 0:
                base = 'bear'
            else:
                base = 'bull'
            count = label_counts[base]
            # BUG-12 FIX: store keys as plain Python int so dict.get(i) never
            # silently misses due to numpy-int vs Python-int key mismatch.
            regime_map[int(s)] = base if count == 0 else f"{base}_{count}"
            label_counts[base] += 1
        return [regime_map[int(s)] for s in states], regime_map

    col_ctrl, col_chart = st.columns([1, 2.8], gap="large")

    with col_ctrl:
        st.markdown("### 📂 Data Source")
        data_src = st.radio("", ["📥 Upload CSV", "🎲 Synthetic Data"], label_visibility="collapsed")

        df = None
        if data_src == "📥 Upload CSV":
            st.markdown(
                '<div class="info-box">CSV must have <b>Date</b> and <b>Close</b> columns.</div>',
                unsafe_allow_html=True
            )
            upload = st.file_uploader("Choose CSV", type=["csv"])
            if upload:
                # FIX #20/#21: Validate required columns before proceeding
                try:
                    raw_df = pd.read_csv(upload)
                    missing = [c for c in ['Date', 'Close'] if c not in raw_df.columns]
                    if missing:
                        st.error(f"CSV is missing required column(s): {missing}")
                    else:
                        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
                        df = raw_df.sort_values('Date').set_index('Date')
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
        else:
            n_rows      = st.slider("Synthetic Rows", 300, 2000, 800)
            regime_seed = st.number_input("Random Seed", 0, 9999, 42)
            # FIX #7: Use a local RandomState instead of global np.random.seed()
            rng    = np.random.RandomState(int(regime_seed))
            prices = [100.0]
            for _ in range(n_rows - 1):
                r_t = rng.normal(0.0003, 0.015)
                prices.append(prices[-1] * np.exp(r_t))
            dates = pd.date_range("2020-01-01", periods=n_rows, freq='B')
            df    = pd.DataFrame({'Close': prices}, index=dates)
            st.success(f"Generated {n_rows} synthetic price rows ✓")

        st.markdown("### ⚙️ HMM Settings")
        n_states  = st.selectbox("Hidden States", [2, 3, 4], index=1)
        window    = st.slider("Feature Window (days)", 5, 60, 20)
        hmm_seed  = st.number_input("HMM Seed", 0, 9999, 42)
        run       = st.button("🔍 Detect Regimes", use_container_width=True)

    with col_chart:
        if df is not None and run:
            try:
                with st.spinner("Fitting HMM …"):
                    features              = compute_features(df, window=window)
                    states, state_probs, transmat, model, scaler = fit_hmm(features, int(n_states), int(hmm_seed))
                    regimes, regime_map   = map_regimes(features, states)
                    df_aligned            = df.loc[features.index].copy()
                    df_aligned['regime']  = regimes
            except ValueError as e:
                st.error(f"❌ HMM fitting failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error during regime detection: {e}")
                st.stop()

            counts = Counter(regimes)
            cols   = st.columns(max(len(counts), 1))
            badges = {'bull': '#3fb950', 'bear': '#f85149', 'sideways': '#c9d1d9'}
            for idx, (regime, cnt) in enumerate(sorted(counts.items())):
                pct   = cnt / len(regimes) * 100
                color = badges.get(regime.split('_')[0], '#aaa')  # handle suffixed labels
                cols[idx].markdown(
                    f'<div class="metric-box">'
                    f'<div class="label" style="color:{color}">{regime.upper()}</div>'
                    f'<div class="value" style="color:{color}">{pct:.1f}%</div>'
                    f'<div style="font-size:11px;color:#8b949e">{cnt} days</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📈 Price + Regimes", "📊 Probabilities", "🔵 Scatter", "🔄 Transition Matrix"]
            )

            REGIME_COLORS = {'bull': '#3fb950', 'bear': '#f85149', 'sideways': '#8b949e'}
            def regime_color(label):
                base = label.split('_')[0]
                return REGIME_COLORS.get(base, '#888888')

            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_aligned.index, y=df_aligned['Close'],
                    line=dict(color='#58a6ff', width=1.5), name='Price'
                ))

                # FIX #6/#8: Correct regime segment drawing + legend
                # Build segment list first, then draw — handles all edge cases
                segments = []
                prev_regime, start_i = regimes[0], 0
                for i in range(1, len(regimes)):
                    if regimes[i] != prev_regime:
                        segments.append((start_i, i - 1, prev_regime))
                        start_i, prev_regime = i, regimes[i]
                segments.append((start_i, len(regimes) - 1, prev_regime))  # always flush final segment

                # BUG-8 FIX: x0==x1 on single-day segments renders as invisible.
                # Use the next date index (or nudge by 1 day) as x1 minimum.
                legend_seen = set()
                for seg_start, seg_end, regime in segments:
                    x0 = df_aligned.index[seg_start]
                    x1 = df_aligned.index[seg_end]
                    # BUG-14 FIX: if x0 == x1 (single-day segment), nudge x1 by 1 calendar day
                    # using pd.Timedelta so it works for any index frequency (daily, business, irregular).
                    if x0 == x1:
                        x1 = x1 + pd.Timedelta(days=1)
                    show = regime not in legend_seen
                    legend_seen.add(regime)
                    fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor=regime_color(regime),
                        opacity=0.18, layer="below", line_width=0
                    )
                    if show:
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode='markers',
                            marker=dict(color=regime_color(regime), size=10, symbol='square'),
                            name=regime.capitalize(), showlegend=True
                        ))

                fig.update_layout(
                    **PLOTLY_LAYOUT, title="Price with Regime Shading",
                    xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                    height=420, legend=dict(bgcolor='#161b27', bordercolor='#21262d')
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig2    = go.Figure()
                palette = ['#58a6ff', '#3fb950', '#f85149', '#e3b341']
                # BUG-12 FIX: use int(state_idx) to match Python-int keys stored in regime_map
                # BUG-15 FIX: renamed loop var from `i` to `state_idx` to avoid any future
                # collision with outer-scope `r` (risk-free rate) if loop var were named `r`
                for state_idx in range(state_probs.shape[1]):
                    r_name = regime_map.get(int(state_idx), f"State {state_idx}").capitalize()
                    fig2.add_trace(go.Scatter(
                        x=df_aligned.index, y=state_probs[:, state_idx],
                        fill='tozeroy', name=r_name,
                        line=dict(color=palette[state_idx % len(palette)], width=1.2)
                    ))
                fig2.update_layout(
                    **PLOTLY_LAYOUT, title="Regime State Probabilities",
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d', range=[0, 1]),
                    height=380
                )
                st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                fig3 = go.Figure()
                unique_regimes = list(set(regimes))
                for regime in unique_regimes:
                    mask = np.array(regimes) == regime
                    if mask.any():
                        fig3.add_trace(go.Scatter(
                            x=features['volatility'][mask],
                            y=features['returns'][mask],
                            mode='markers', name=regime.capitalize(),
                            marker=dict(color=regime_color(regime), size=5, opacity=0.7)
                        ))
                fig3.update_layout(
                    **PLOTLY_LAYOUT, title="Volatility vs Returns by Regime",
                    xaxis=dict(title="Rolling Volatility", gridcolor='#21262d'),
                    yaxis=dict(title="Log Returns",        gridcolor='#21262d'),
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

            with tab4:
                fig4 = px.imshow(
                    np.round(transmat, 3),
                    labels=dict(x="To State", y="From State", color="Probability"),
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                fig4.update_layout(**PLOTLY_LAYOUT, title="HMM Transition Matrix", height=350)
                st.plotly_chart(fig4, use_container_width=True)

        elif df is None and run:
            st.warning("Please upload a CSV file or select Synthetic Data first.")
        else:
            if df is not None:
                st.markdown(
                    '<div class="warn-box">Click <b>🔍 Detect Regimes</b> to run the HMM analysis.</div>',
                    unsafe_allow_html=True
                )
                st.line_chart(df['Close'])
            else:
                st.markdown(
                    '<div class="info-box">Select a data source on the left and click <b>Detect Regimes</b> to begin.</div>',
                    unsafe_allow_html=True
                )


# ═════════════════════════════════════════════════════════════════════════════
# BOT 3 — VOLATILITY SURFACE FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
def render_vol_surface_bot():
    st.markdown(
        '<div class="hero"><h1>🌐 Volatility Surface Forecasting Bot</h1>'
        '<p>Random Forest model trained on options data to predict and visualize the implied volatility surface</p></div>',
        unsafe_allow_html=True
    )

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # FIX #17: Use local RandomState instead of global np.random.seed()
    def simulate_data(n, spot, seed):
        rng        = np.random.RandomState(seed)
        strikes    = rng.uniform(spot * 0.7, spot * 1.3, n)
        maturities = rng.uniform(0.05, 2.0, n)
        iv         = 0.15 + 0.25 * np.exp(-maturities) * np.exp(-((strikes / spot - 1)**2) / 0.1)
        iv        += rng.normal(0, 0.01, n)
        return pd.DataFrame({
            'strike':           strikes,
            'maturity':         maturities,
            'implied_volatility': np.clip(iv, 0.01, 1.0),
            'spot':             spot
        })

    def add_features(df):
        df = df.copy()
        df['moneyness']             = df['strike'] / df['spot']
        df['time_to_maturity']      = df['maturity']
        df['historical_volatility'] = 0.18 + 0.05 * np.exp(-df['maturity'])
        # BUG-10 FIX: clip moneyness above a tiny floor before log to prevent -inf / NaN
        df['log_moneyness']         = np.log(df['moneyness'].clip(lower=1e-8))
        df['sqrt_ttm']              = np.sqrt(df['maturity'].clip(lower=0.0))
        return df

    def create_grid(s_range, m_range, spot, n=60):
        strikes    = np.linspace(*s_range, n)
        maturities = np.linspace(*m_range, n)
        gs, gm     = np.meshgrid(strikes, maturities)
        return pd.DataFrame({
            'strike':   gs.ravel(),
            'maturity': gm.ravel(),
            'spot':     spot
        }), gs, gm

    col_ctrl, col_chart = st.columns([1, 2.8], gap="large")

    with col_ctrl:
        st.markdown("### ⚙️ Data & Model")
        spot   = st.number_input("Spot Price", 50.0, 500.0, 100.0, 5.0)
        n_samp = st.slider("Training Samples", 300, 5000, 1000)
        seed   = st.number_input("Random Seed", 0, 9999, 42)
        algo   = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])
        test_p = st.slider("Test Split %", 10, 40, 20) / 100

        st.markdown("### 🌐 Surface Grid")
        c1, c2 = st.columns(2)

        # FIX #5/#8: Clamp Strike min/max/default so they never violate constraints
        # Use spot*0.5 as the floor (not hardcoded 50) so the range is always valid
        s_lo_min     = max(spot * 0.5, 1.0)
        s_lo_max     = spot - 1.0
        s_lo_default = float(np.clip(spot * 0.75, s_lo_min, s_lo_max))
        s_lo = c1.number_input("Strike min", s_lo_min, s_lo_max, s_lo_default, 1.0)

        s_hi_min     = spot + 1.0
        s_hi_default = float(np.clip(spot * 1.25, s_hi_min, spot * 2.0))
        s_hi = c2.number_input("Strike max", s_hi_min, spot * 2.0, s_hi_default, 1.0)

        m_lo = c1.number_input("Maturity min", 0.01, 0.5,  0.05, 0.01, format="%.2f")
        m_hi = c2.number_input("Maturity max", 0.5,  3.0,  2.0,  0.10, format="%.2f")

        st.markdown("### 🔮 Custom Prediction")
        st.markdown(
            '<div class="info-box">Predict IV for a single (Strike, Maturity) point.</div>',
            unsafe_allow_html=True
        )
        pred_k = st.number_input("Strike",         float(s_lo), float(s_hi), float(np.clip(spot, s_lo, s_hi)), 1.0,  key="pk")
        pred_m = st.number_input("Maturity (yr)",  float(m_lo), float(m_hi), float(np.clip(0.5, m_lo, m_hi)),  0.05, format="%.2f", key="pm")

        run = st.button("🚀 Train & Forecast", use_container_width=True)

    with col_chart:
        if run:
            try:
                with st.spinner("Training model …"):
                    FEATURES = ['moneyness', 'time_to_maturity', 'historical_volatility', 'log_moneyness', 'sqrt_ttm']
                    df = simulate_data(int(n_samp), spot, int(seed))
                    df = add_features(df)

                    X, y = df[FEATURES].values, df['implied_volatility'].values
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_p, random_state=int(seed))

                    if len(X_tr) == 0 or len(X_te) == 0:
                        st.error("Train/test split produced an empty set. Reduce Test Split % or increase Training Samples.")
                        st.stop()

                    if algo == "Random Forest":
                        mdl = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=int(seed))
                    else:
                        mdl = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=int(seed))

                    mdl.fit(X_tr, y_tr)
                    y_pred = mdl.predict(X_te)
                    mse    = mean_squared_error(y_te, y_pred)
                    r2     = r2_score(y_te, y_pred)

                    grid_df, gs, gm = create_grid((s_lo, s_hi), (m_lo, m_hi), spot)
                    grid_df         = add_features(grid_df)
                    Z               = mdl.predict(grid_df[FEATURES].values).reshape(gs.shape)

                    cp_df  = pd.DataFrame({'strike': [pred_k], 'maturity': [pred_m], 'spot': [spot]})
                    cp_df  = add_features(cp_df)
                    cp_iv  = mdl.predict(cp_df[FEATURES].values)[0]
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
                st.stop()

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-box"><div class="label">R² Score</div><div class="value">{r2:.4f}</div></div>',           unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="label">RMSE</div><div class="value">{np.sqrt(mse)*100:.3f}%</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="label">Predicted IV</div><div class="value">{cp_iv*100:.2f}%</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-box"><div class="label">Samples</div><div class="value">{n_samp}</div></div>',             unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(
                ["🌐 Vol Surface", "🎯 Actual vs Predicted", "📊 Contour", "🔍 Feature Importance"]
            )

            with tab1:
                fig = go.Figure(go.Surface(
                    x=gs, y=gm, z=Z,
                    colorscale='Viridis', showscale=True, opacity=0.93,
                    contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Predicted Implied Volatility Surface",
                    scene=dict(
                        xaxis=dict(title="Strike",         gridcolor='#21262d', backgroundcolor='#0d1117'),
                        yaxis=dict(title="Maturity (yrs)", gridcolor='#21262d', backgroundcolor='#0d1117'),
                        zaxis=dict(title="Implied Vol",    gridcolor='#21262d', backgroundcolor='#0d1117'),
                        bgcolor='#0d1117'
                    ),
                    height=500
                )
                fig.add_trace(go.Scatter3d(
                    x=[pred_k], y=[pred_m], z=[cp_iv],
                    mode='markers+text',
                    marker=dict(color='#f85149', size=8, symbol='diamond'),
                    text=[f"IV={cp_iv*100:.2f}%"], textposition="top center",
                    name="Custom Point"
                ))
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=y_te, y=y_pred, mode='markers',
                    marker=dict(color='#58a6ff', size=5, opacity=0.6), name='Predictions'
                ))
                mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
                fig2.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode='lines',
                    line=dict(color='#f85149', dash='dash', width=2), name='Perfect Fit'
                ))
                fig2.update_layout(
                    **PLOTLY_LAYOUT, title="Actual vs Predicted Implied Volatility",
                    xaxis=dict(title="Actual IV",    gridcolor='#21262d'),
                    yaxis=dict(title="Predicted IV", gridcolor='#21262d'),
                    height=420
                )
                st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                # BUG-9 FIX: create_grid uses meshgrid(strikes, maturities) with default 'xy'
                # indexing, so gs[i,j]=strikes[j], gm[i,j]=maturities[i], Z shape=(n_mat, n_str).
                # Contour x must match columns (strikes), y must match rows (maturities). Correct.
                fig3 = go.Figure(go.Contour(
                    x=np.linspace(s_lo, s_hi, 60),
                    y=np.linspace(m_lo, m_hi, 60),
                    z=Z,  # shape (60,60): rows=maturities, cols=strikes — matches x/y above
                    colorscale='Viridis', showscale=True,
                    contours=dict(showlabels=True)
                ))
                fig3.update_layout(
                    **PLOTLY_LAYOUT, title="Volatility Surface Contour Map",
                    xaxis=dict(title="Strike",         gridcolor='#21262d'),
                    yaxis=dict(title="Maturity (yrs)", gridcolor='#21262d'),
                    height=420
                )
                st.plotly_chart(fig3, use_container_width=True)

            with tab4:
                imp  = mdl.feature_importances_
                fig4 = go.Figure(go.Bar(
                    x=imp, y=FEATURES, orientation='h',
                    marker=dict(color='#58a6ff', opacity=0.85)
                ))
                fig4.update_layout(
                    **PLOTLY_LAYOUT, title="Feature Importances",
                    xaxis=dict(title="Importance", gridcolor='#21262d'),
                    yaxis=dict(autorange='reversed'),
                    height=360
                )
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.markdown(
                '<div class="info-box">Configure parameters on the left and click '
                '<b>🚀 Train & Forecast</b> to run the model.</div>',
                unsafe_allow_html=True
            )


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 16px 0;">
        <div style="font-size:32px">📊</div>
        <div style="font-size:17px; font-weight:700; color:#f0f6fc; margin-top:6px">Quant Bot Suite</div>
        <div style="font-size:11px; color:#8b949e; margin-top:2px">by aj1m153</div>
    </div>
    <hr style="border-color:#21262d; margin:0 0 16px 0">
    """, unsafe_allow_html=True)

    BOT_OPTIONS = {
        "📐  Delta-Gamma Risk Surface": "delta_gamma",
        "🔍  Latent Regime Detection":  "regime",
        "🌐  Volatility Surface Forecast": "vol_surface",
    }
    selected = st.radio("", list(BOT_OPTIONS.keys()), label_visibility="collapsed")
    bot_key  = BOT_OPTIONS[selected]

    st.markdown("""
    <hr style="border-color:#21262d; margin:16px 0">
    <div style="font-size:12px; color:#8b949e; padding: 0 4px">
        <b style="color:#c9d1d9">Tech Stack</b><br><br>
        🐍 Python · Streamlit<br>
        📐 Black-Scholes · Scipy<br>
        🤖 HMM · hmmlearn<br>
        🌲 Scikit-learn · Plotly<br><br>
        <a href="https://github.com/aj1m153/Delta-Gamma-Risk-Surface-Bot"      target="_blank" style="color:#58a6ff">Delta-Gamma Bot ↗</a><br>
        <a href="https://github.com/aj1m153/Latent-Regime-Detection-Bot"       target="_blank" style="color:#58a6ff">Regime Bot ↗</a><br>
        <a href="https://github.com/aj1m153/Volatility-Surface-Forecasting-Bot" target="_blank" style="color:#58a6ff">Vol Surface Bot ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER SELECTED BOT
# ─────────────────────────────────────────────────────────────────────────────
if bot_key == "delta_gamma":
    render_delta_gamma_bot()
elif bot_key == "regime":
    render_regime_bot()
elif bot_key == "vol_surface":
    render_vol_surface_bot()
