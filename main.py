import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats

# Page configuration
st.set_page_config(page_title="Stats Laboratory", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e445e;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 Statistical Distribution Lab")
st.markdown("---")

# Sidebar for global controls
with st.sidebar:
    st.header("🎛️ Global Controls")
    dist_type = st.selectbox(
        "Select Distribution:",
        ["Normal", "Beta", "Log-Normal", "Exponential", "Poisson", "Chi-Squared"]
    )
    
    samples = st.select_slider(
        "Sample Size (N):", 
        options=[100, 500, 1000, 5000, 10000], 
        value=1000,
        help="How many random data points we generate. More points = smoother curve."
    )
    bins = st.slider(
        "Histogram Resolution:", 10, 100, 40,
        help="Number of bars in the histogram. Higher resolution shows more detail."
    )
    show_curve = st.checkbox(
        "Show Density Curve (KDE)", 
        value=True, 
        help="Kernel Density Estimation (KDE) is a way to estimate the probability density function."
    )

# Layout: Params vs Visuals
col_params, col_viz = st.columns([1, 3])

with col_params:
    st.subheader("⚙️ Parameters")
    data = np.array([])
    formula = ""

    if dist_type == "Normal":
        mu = st.number_input("Mean (μ)", value=0.0, 
                             help="The peak of the bell curve. Most data points will cluster around this value.")
        sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01, 
                                help="The 'stretch' of the curve. Low σ is skinny/tall; high σ is fat/short.")
        data = np.random.normal(mu, sigma, samples)
        formula = r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}"

    elif dist_type == "Beta":
        a = st.number_input("Alpha (α)", value=2.0, min_value=0.1, 
                            help="Shape parameter that pulls the mass toward 1. Often used in Bayesian stats.")
        b = st.number_input("Beta (β)", value=5.0, min_value=0.1, 
                            help="Shape parameter that pulls the mass toward 0.")
        data = np.random.beta(a, b, samples)
        formula = r"f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}"

    elif dist_type == "Log-Normal":
        mu = st.number_input("Log-Mean (μ)", value=0.0, 
                             help="The mean of the data's natural logarithm. Used for skewed growth data.")
        sigma = st.number_input("Log-Sigma (σ)", value=0.5, min_value=0.1, 
                                help="The standard deviation of the data's natural logarithm.")
        data = np.random.lognormal(mu, sigma, samples)
        formula = r"f(x) = \frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}"

    elif dist_type == "Exponential":
        scale = st.number_input("Scale (β = 1/λ)", value=1.0, min_value=0.1, 
                                help="Average time between events. Higher scale means the curve decays slower.")
        data = np.random.exponential(scale, samples)
        formula = r"f(x) = \lambda e^{-\lambda x}"

    elif dist_type == "Poisson":
        lam = st.number_input("Rate (λ)", value=4.0, min_value=0.1, 
                              help="The average number of times an event occurs in a set interval.")
        data = np.random.poisson(lam, samples)
        formula = r"P(k) = \frac{\lambda^k e^{-\lambda}}{k!}"

    elif dist_type == "Chi-Squared":
        df = st.number_input("Degrees of Freedom (k)", value=3, min_value=1, 
                             help="Number of independent variables. As k increases, it starts looking more 'Normal'.")
        data = np.random.chisquare(df, samples)
        formula = r"f(x; k) = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}"

    st.markdown("---")
    st.latex(formula)

# Visualization
with col_viz:
    if data.any():
        fig = ff.create_distplot(
            [data], [f"{dist_type} Dist"], 
            bin_size=[(np.max(data) - np.min(data)) / bins],
            show_curve=show_curve, show_rug=False,
            colors=['#00d4ff']
        )
        fig.update_layout(
            template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20),
            height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# Live Statistics Footer
st.markdown("### 📊 Real-time Analytics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean", f"{np.mean(data):.3f}")
m2.metric("Median", f"{np.median(data):.3f}")
m3.metric("Std Dev", f"{np.std(data):.3f}")
m4.metric("Variance", f"{np.var(data):.3f}")
