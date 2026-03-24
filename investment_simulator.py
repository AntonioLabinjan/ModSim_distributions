import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Pro Wealth Simulator", layout="wide", initial_sidebar_state="expanded")

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .main { background-color: #0e1117; }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 Strategic Wealth & Risk Simulator")
st.markdown("---")

# 2. Sidebar: Input Parameters with Explanations
with st.sidebar:
    st.header("📈 Investment Setup")
    initial_inv = st.number_input("Initial Lump Sum ($)", value=10000, step=1000, 
                                  help="Your starting capital. The 'seed' money.")
    monthly_dep = st.number_input("Monthly Contribution ($)", value=500, step=100, 
                                  help="New cash added every month. This is your 'savings rate'.")
    
    st.header("📊 Market Dynamics")
    exp_return = st.slider("Expected Annual Return (%)", 0.0, 20.0, 8.0, 
                           help="The average yearly growth (Drift). S&P 500 is ~10% historically.") / 100
    volatility = st.slider("Market Volatility (σ %)", 5.0, 50.0, 18.0, 
                           help="The 'swing' factor. Higher σ means more risk and wider outcomes.") / 100
    
    st.header("⏳ Time & Depth")
    years = st.number_input("Years to Sim", value=15, min_value=1, 
                            help="The investment horizon. More years = more compounding.")
    sims = st.select_slider("Parallel Universes", options=[10, 50, 100, 500, 1000], value=100,
                            help="How many different 'lives' your portfolio lives through.")

# 3. Simulation Logic (Geometric Brownian Motion)
days = int(years * 252)
dt = 1 / 252
mu_daily = exp_return / 252
sigma_daily = volatility / np.sqrt(252)

# Total Cash Invested (The Break-Even Point)
total_invested = initial_inv + (monthly_dep * 12 * years)

# Path Calculation
paths = np.zeros((days, sims))
paths[0] = initial_inv

for t in range(1, days):
    shocks = np.random.normal(0, 1, sims)
    # GBM formula: Price moves based on Drift - Drag (0.5*sigma^2) + Volatility
    growth = np.exp((mu_daily - 0.5 * sigma_daily**2) + sigma_daily * shocks)
    paths[t] = paths[t-1] * growth
    if t % 21 == 0: # Add monthly deposit roughly every month (21 trading days)
        paths[t] += monthly_dep

# 4. Data Preparation for Plots (Renaming columns to avoid 'Variable 0')
df_paths = pd.DataFrame(paths)
df_paths.columns = [f"Scenario {i+1}" for i in range(sims)]
final_wealths = paths[-1]

# 5. Visualizations
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Projected Growth Paths")
    fig_paths = px.line(df_paths, labels={"value": "Wealth ($)", "index": "Trading Days", "variable": "Universe"})
    fig_paths.update_layout(template="plotly_dark", showlegend=False, hovermode="x", height=500)
    # Custom hover label to remove 'Variable' and show 'Scenario'
    fig_paths.update_traces(hovertemplate="Wealth: %{y:$.0f}<extra></extra>")
    st.plotly_chart(fig_paths, use_container_width=True)

with col2:
    st.subheader("Final Wealth Outcomes")
    fig_dist = px.histogram(x=final_wealths, nbins=30, labels={'x': 'Final Wealth ($)'}, color_discrete_sequence=['#00d4ff'])
    
    # Add the Break-Even Vertical Line
    fig_dist.add_vline(x=total_invested, line_dash="dash", line_color="red", 
                       annotation_text="Break-Even", annotation_position="top left")
    
    fig_dist.update_layout(template="plotly_dark", showlegend=False, yaxis_title="Number of Scenarios", height=500)
    st.plotly_chart(fig_dist, use_container_width=True)

# 6. Analytics Metrics
st.divider()
st.markdown("### 📊 Live Performance Analytics")

avg_val = np.mean(final_wealths)
med_val = np.median(final_wealths)
win_rate = (np.sum(final_wealths > total_invested) / sims) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Invested Cash", f"${total_invested:,.0f}", help="The sum of all your deposits. Your 'cost basis'.")
c2.metric("Median Outcome", f"${med_val:,.0f}", help="The 50/50 mark. This is the most realistic expectation.")
c3.metric("Prob. of Profit", f"{win_rate:.1f}%", help="Percent of scenarios that ended above your red break-even line.")
c4.metric("Avg. Outcome", f"${avg_val:,.0f}", help="Heavily skewed by 'lotto' scenarios on the far right of the histogram.")

st.info(f"Summary: In {sims} simulations, you have a {win_rate:.1f}% chance of making a profit after {years} years.")
