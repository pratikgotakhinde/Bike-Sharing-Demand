import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Bike Sharing Demand Dashboard",
    layout="wide",
    page_icon="🚲"
)

# ---------- GLOBAL THEME & CSS (BIKE STYLE) ----------
st.markdown("""
<style>
/* Dark asphalt background, bike-green accent */
.stApp {
    background: radial-gradient(circle at top left, #222831 0%, #0f141a 40%, #000000 100%);
    color: #f5f5f5;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main content container */
.block-container {
    padding-top: 1rem;
}

/* Titles */
h1, h2, h3 {
    font-family: "Poppins", "Inter", sans-serif;
    color: #e8f9fd;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 16, 24, 0.95);
    border-right: 1px solid #1f2933;
}

/* Card look */
.card {
    background: linear-gradient(135deg, #111827, #020617);
    padding: 16px 24px;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.3);
    margin-bottom: 1rem;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #4ade80;  /* bike green */
    font-weight: 600;
}

/* Widgets accent (buttons/sliders) */
.stButton>button {
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #a3e635);
    color: #020617;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    filter: brightness(1.05);
}
.stSlider>div>div>div {
    border-radius: 999px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] p {
    font-weight: 500;
}

/* Popover button for charts */
.chart-pop-btn {
    background: none;
    border: none;
    color: #38bdf8;
    cursor: pointer;
    font-size: 0.85rem;
    padding: 0;
}

/* Small caption under charts */
.chart-caption {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: -6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center; margin-bottom:0;'>
    🚲 Bike Sharing Demand Dashboard
</h1>
<p style='text-align:center; color:#9ca3af; font-size:0.95rem;'>
    Explore how time, season and weather shape bike rental behaviour in Washington, D.C.
</p>
<hr style="border-color:#1f2933;">
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour

    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)

    def get_day_period(h):
        if h < 6:
            return "night"
        elif h < 12:
            return "morning"
        elif h < 18:
            return "afternoon"
        else:
            return "evening"

    df["day_period"] = df["hour"].apply(get_day_period)
    return df

df = load_data()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")

year_opt = st.sidebar.selectbox(
    "Year",
    ["Both", 2011, 2012]
)

season_opt = st.sidebar.multiselect(
    "Season",
    options=sorted(df["season_name"].dropna().unique()),
    default=sorted(df["season_name"].dropna().unique())
)

weather_opt = st.sidebar.multiselect(
    "Weather category",
    options=sorted(df["weather"].unique()),
    default=sorted(df["weather"].unique())
)

workingday_map = {"Both": None, "Working days only": 1, "Non-working days only": 0}
workingday_label = st.sidebar.selectbox(
    "Working day filter",
    list(workingday_map.keys())
)

min_hour, max_hour = st.sidebar.slider(
    "Hour range",
    0, 23, (0, 23)
)

use_registered = st.sidebar.checkbox(
    "Show registered users instead of total count",
    value=False
)

# ---------- APPLY FILTERS ----------
df_filtered = df.copy()

if year_opt != "Both":
    df_filtered = df_filtered[df_filtered["year"] == year_opt]

df_filtered = df_filtered[df_filtered["season_name"].isin(season_opt)]
df_filtered = df_filtered[df_filtered["weather"].isin(weather_opt)]
df_filtered = df_filtered[(df_filtered["hour"] >= min_hour) & (df_filtered["hour"] <= max_hour)]

wd_val = workingday_map[workingday_label]
if wd_val is not None:
    df_filtered = df_filtered[df_filtered["workingday"] == wd_val]

target_col = "registered" if use_registered else "count"

# ---------- KPI METRICS (CARD) ----------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        st.metric(
            "Total rides (filtered)",
            f"{df_filtered[target_col].sum():,.0f}"
        )

    with col_kpi2:
        st.metric(
            "Average rides per hour",
            f"{df_filtered[target_col].mean():.1f}"
        )

    with col_kpi3:
        if not df_filtered.empty:
            peak_hour = df_filtered.groupby("hour")[target_col].mean().idxmax()
        else:
            peak_hour = "-"
        st.metric("Peak hour (filtered)", f"{peak_hour}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["Time patterns", "Season & Weather", "Correlations"])

# Helper to render chart + popover
def chart_with_popover(fig, title: str, btn_key: str):
    col_main, col_btn = st.columns([8, 1])
    with col_main:
        st.pyplot(fig)
    with col_btn:
        with st.popover("🔍 Enlarge", use_container_width=True):
            st.markdown(f"**{title}**", unsafe_allow_html=True)
            st.pyplot(fig)

# ----- TAB 1: TIME PATTERNS -----
with tab1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Mean rentals by hour")
            fig, ax = plt.subplots()
            sns.lineplot(
                data=df_filtered,
                x="hour",
                y=target_col,
                estimator=np.mean,
                ci=95,
                marker="o",
                ax=ax,
                color="#22c55e"
            )
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Mean rentals")
            ax.grid(alpha=0.25)
            chart_with_popover(fig, "Mean rentals by hour", "hour_pop")
            st.markdown("<p class='chart-caption'>Typical commuter peaks around morning and evening hours.</p>", unsafe_allow_html=True)

        with col2:
            st.subheader("Mean rentals by period of day")
            fig2, ax2 = plt.subplots()
            order = ["night", "morning", "afternoon", "evening"]
            sns.barplot(
                data=df_filtered,
                x="day_period",
                y=target_col,
                estimator=np.mean,
                ci=95,
                order=order,
                palette="Greens"
            )
            ax2.set_xlabel("Period of day")
            ax2.set_ylabel("Mean rentals")
            chart_with_popover(fig2, "Mean rentals by period of day", "period_pop")
            st.markdown("<p class='chart-caption'>Evening and morning windows highlight strong rush-hour usage.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Hourly rentals by day of week")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                data=df_filtered,
                x="hour",
                y=target_col,
                hue="day_of_week",
                estimator=np.mean,
                ci=None,
                marker="o",
                ax=ax3
            )
            ax3.set_xlabel("Hour")
            ax3.set_ylabel("Mean rentals")
            ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            ax3.grid(alpha=0.25)
            chart_with_popover(fig3, "Hourly rentals by day of week", "dow_pop")
            st.markdown("<p class='chart-caption'>Compare workdays vs weekend profiles to see commuting effects.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ----- TAB 2: SEASON & WEATHER -----
with tab2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col4, col5 = st.columns(2)

        with col4:
            st.subheader("Mean rentals by season")
            fig4, ax4 = plt.subplots()
            sns.barplot(
                data=df_filtered,
                x="season_name",
                y=target_col,
                estimator=np.mean,
                ci=95,
                palette="YlGn",
                ax=ax4
            )
            ax4.set_xlabel("Season")
            ax4.set_ylabel("Mean rentals")
            chart_with_popover(fig4, "Mean rentals by season", "season_pop")
            st.markdown("<p class='chart-caption'>Warm seasons boost usage; winter typically shows a drop.</p>", unsafe_allow_html=True)

        with col5:
            st.subheader("Mean rentals by weather")
            fig5, ax5 = plt.subplots()
            sns.barplot(
                data=df_filtered,
                x="weather",
                y=target_col,
                estimator=np.mean,
                ci=95,
                palette="GnBu",
                ax=ax5
            )
            ax5.set_xlabel("Weather category")
            ax5.set_ylabel("Mean rentals")
            chart_with_popover(fig5, "Mean rentals by weather", "weather_pop")
            st.markdown("<p class='chart-caption'>Clear days drive more rides; harsh conditions dampen demand.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ----- TAB 3: CORRELATIONS -----
with tab3:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Correlation heatmap (numeric features)")
        num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 1 and not df_filtered.empty:
            corr = df_filtered[num_cols].corr()
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=False, cmap="rocket_r", ax=ax6)
            chart_with_popover(fig6, "Correlation heatmap", "corr_pop")
            st.markdown("<p class='chart-caption'>Check how temperature, humidity and other factors move with demand.</p>", unsafe_allow_html=True)
        else:
            st.info("Not enough numeric data after filtering to compute correlations.")

        st.markdown("</div>", unsafe_allow_html=True)
