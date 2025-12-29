import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bike Sharing Demand", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour

    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season"] = df["season"].map(season_map)

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

st.title("Bike Sharing Demand Dashboard")
st.markdown("Interactive overview of bike rentals in 2011–2012.")

# -------- SIDEBAR FILTERS --------
st.sidebar.header("Filters")

year_opt = st.sidebar.selectbox(
    "Year",
    ["Both", 2011, 2012]
)

season_opt = st.sidebar.multiselect(
    "Season",
    options=df["season"].unique(),
    default=list(df["season"].unique())
)

workingday_map = {"Both": None, "Working days only": 1, "Non-working days only": 0}
workingday_label = st.sidebar.selectbox(
    "Working day filter",
    list(workingday_map.keys())
)

# Apply filters
df_filtered = df.copy()
if year_opt != "Both":
    df_filtered = df_filtered[df_filtered["year"] == year_opt]

df_filtered = df_filtered[df_filtered["season"].isin(season_opt)]

wd_val = workingday_map[workingday_label]
if wd_val is not None:
    df_filtered = df_filtered[df_filtered["workingday"] == wd_val]

st.sidebar.write(f"Rows after filtering: {len(df_filtered)}")

# -------- ROW 1: HOURLY + DAY PERIOD --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mean rentals by hour")
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_filtered,
        x="hour", y="count",
        estimator=np.mean, ci=95, marker="o", ax=ax
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean rentals")
    st.pyplot(fig)

with col2:
    st.subheader("Mean rentals by period of day")
    fig2, ax2 = plt.subplots()
    order = ["night", "morning", "afternoon", "evening"]
    sns.barplot(
        data=df_filtered,
        x="day_period", y="count",
        estimator=np.mean, ci=95, order=order, ax=ax2
    )
    ax2.set_xlabel("Period of day")
    ax2.set_ylabel("Mean rentals")
    st.pyplot(fig2)

# -------- ROW 2: DAY OF WEEK + SEASON --------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Hourly rentals by day of week")
    fig3, ax3 = plt.subplots()
    sns.lineplot(
        data=df_filtered,
        x="hour", y="count",
        hue="day_of_week",
        estimator=np.mean, ci=95, marker="o", ax=ax3
    )
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Mean rentals")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig3)

with col4:
    st.subheader("Hourly rentals by season")
    fig4 = sns.relplot(
        data=df_filtered,
        x="hour", y="count",
        col="season",
        estimator=np.mean, ci=95, kind="line",
        height=3, aspect=1.1
    )
    st.pyplot(fig4)

# -------- ROW 3: WEATHER + CORRELATION --------
st.subheader("Mean rentals by weather category")
fig5, ax5 = plt.subplots()
sns.barplot(
    data=df_filtered,
    x="weather", y="count",
    estimator=np.mean, ci=95, ax=ax5
)
ax5.set_xlabel("Weather category")
ax5.set_ylabel("Mean rentals")
st.pyplot(fig5)

st.subheader("Correlation heatmap (numerical variables)")
num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns
corr = df_filtered[num_cols].corr()
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=False, cmap="RdBu_r", ax=ax6)
st.pyplot(fig6)
st.markdown("Developed by Pratik Chandrakant Gotakhinde - 88215836")
st.markdown("Data source: [Bike Sharing Demand on Kaggle](https://www.kaggle.com/c/bike-sharing-demand)")
