import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG & STYLE ----------
st.set_page_config(page_title="Bike Sharing Demand Dashboard",
                   layout="wide",
                   page_icon="🚲")

st.markdown(
    """
    <h1 style='text-align:center; color:#2e86de;'>
        Bike Sharing Demand Dashboard
    </h1>
    <p style='text-align:center; color:#555555;'>
        Explore how time, weather and season affect bike rentals.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

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

use_registered = st.sidebar.checkbox("Show registered users instead of total count", value=False)

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

# ---------- KPI METRICS ----------
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.metric("Total rides (filtered)",
              f"{df_filtered[target_col].sum():,.0f}")

with col_kpi2:
    st.metric("Average rides per hour",
              f"{df_filtered[target_col].mean():.1f}")

with col_kpi3:
    peak_hour = (
        df_filtered.groupby("hour")[target_col].mean()
        .idxmax()
        if not df_filtered.empty else "-"
    )
    st.metric("Peak hour (filtered)", f"{peak_hour}")

st.write("")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["Time patterns", "Season & Weather", "Correlations"])

# ----- TAB 1: TIME PATTERNS -----
with tab1:
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
            color="#2e86de"
        )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean rentals")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

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
            palette="Blues"
        )
        ax2.set_xlabel("Period of day")
        ax2.set_ylabel("Mean rentals")
        st.pyplot(fig2)

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
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)

# ----- TAB 2: SEASON & WEATHER -----
with tab2:
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
            palette="coolwarm",
            ax=ax4
        )
        ax4.set_xlabel("Season")
        ax4.set_ylabel("Mean rentals")
        st.pyplot(fig4)

    with col5:
        st.subheader("Mean rentals by weather")
        fig5, ax5 = plt.subplots()
        sns.barplot(
            data=df_filtered,
            x="weather",
            y=target_col,
            estimator=np.mean,
            ci=95,
            palette="viridis",
            ax=ax5
        )
        ax5.set_xlabel("Weather category")
        ax5.set_ylabel("Mean rentals")
        st.pyplot(fig5)

# ----- TAB 3: CORRELATIONS -----
with tab3:
    st.subheader("Correlation heatmap (numeric features)")
    num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 1:
        corr = df_filtered[num_cols].corr()
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=False, cmap="RdBu_r", ax=ax6)
        st.pyplot(fig6)
    else:
        st.info("Not enough numeric columns after filtering to compute correlations.")
