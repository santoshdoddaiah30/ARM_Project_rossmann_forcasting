import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# PATHS (dashboard is inside scripts/)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")

TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, "train_features.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")


# ----------------------------
# PAGE SETTINGS
# ----------------------------
st.set_page_config(page_title="Rossmann Demand Forecast Dashboard", layout="wide")


# ----------------------------
# SAFE LOADERS
# ----------------------------
@st.cache_data
def load_train():
    df = pd.read_csv(TRAIN_FEATURES_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_submission():
    if os.path.exists(SUBMISSION_PATH):
        return pd.read_csv(SUBMISSION_PATH)
    return None


def must_have_files():
    missing = []
    if not os.path.exists(TRAIN_FEATURES_PATH):
        missing.append("data/train_features.csv")
    if len(missing) > 0:
        st.error("Required file(s) missing:\n\n- " + "\n- ".join(missing))
        st.info("Run the pipeline first:\n\n`python scripts/pipeline.py`")
        st.stop()


# ----------------------------
# APP TITLE
# ----------------------------
st.title("📊 Rossmann Demand Forecasting Dashboard")
st.caption("ARM Project | Forecasting Consumer Demand using ML (XGBoost Final Model)")


# ----------------------------
# CHECK REQUIRED FILES
# ----------------------------
must_have_files()

# Load data
train = load_train()
submission = load_submission()


# ----------------------------
# SIDEBAR FILTERS (FORM + APPLY BUTTON)
# ----------------------------
st.sidebar.header("Filters")

with st.sidebar.form("filter_form"):
    # Date range
    min_date = train["Date"].min().date()
    max_date = train["Date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Store filter
    store_list = sorted(train["Store"].unique().tolist())
    selected_store = st.selectbox("Store", options=["All"] + store_list, index=0)

    # Promo filter
    promo_filter = st.selectbox("Promo", options=["All", 0, 1], index=0)

    # Month filter
    month_filter = st.selectbox("Month", options=["All"] + list(range(1, 13)), index=0)

    submitted = st.form_submit_button("✅ Apply Filters")

# Filtered dataframe (default = full data)
df = train.copy()

if submitted:
    # Date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]

    # Store filter
    if selected_store != "All":
        df = df[df["Store"] == selected_store]

    # Promo filter
    if promo_filter != "All":
        df = df[df["Promo"] == promo_filter]

    # Month filter
    if month_filter != "All":
        df = df[df["Month"] == month_filter]
else:
    st.sidebar.info("Select filters and click **Apply Filters**")


# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Overview", "📈 EDA (Charts)", "🖼️ Figures (PNG)", "🤖 Model Performance", "📤 Forecast Output"]
)


# ============================
# TAB 1: OVERVIEW
# ============================
with tab1:
    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (Filtered)", f"{df.shape[0]:,}")
    col2.metric("Stores (Filtered)", f"{df['Store'].nunique():,}")
    col3.metric("Date Start", f"{df['Date'].min().date()}")
    col4.metric("Date End", f"{df['Date'].max().date()}")

    st.markdown("### Final Model Validation Metrics (XGBoost)")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", "508.97")
    c2.metric("MAE", "361.30")
    c3.metric("R²", "0.9733")

    st.info("To regenerate results, figures, and submission file run:\n\n`python scripts/pipeline.py`")


# ============================
# TAB 2: EDA (interactive charts)
# ============================
with tab2:
    st.subheader("Exploratory Data Analysis (Interactive)")

    left, right = st.columns(2)

    # Daily Sales Trend
    with left:
        st.markdown("### Daily Sales Trend")
        daily_sales = df.groupby("Date")["Sales"].sum().sort_index()

        fig = plt.figure(figsize=(8, 4))
        plt.plot(daily_sales.index, daily_sales.values)
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        st.pyplot(fig)

    # Promo vs No Promo
    with right:
        st.markdown("### Promo vs No Promo (Avg Sales)")
        promo_sales = df.groupby("Promo")["Sales"].mean()

        fig = plt.figure(figsize=(6, 4))
        labels = ["No Promo (0)", "Promo (1)"]
        values = [promo_sales.get(0, 0), promo_sales.get(1, 0)]
        plt.bar(labels, values)
        plt.ylabel("Average Sales")
        st.pyplot(fig)

    left2, right2 = st.columns(2)

    # Monthly Seasonality
    with left2:
        st.markdown("### Monthly Seasonality (Avg Sales)")
        monthly_sales = df.groupby("Month")["Sales"].mean().sort_index()

        fig = plt.figure(figsize=(7, 4))
        plt.plot(monthly_sales.index, monthly_sales.values)
        plt.xticks(range(1, 13))
        plt.xlabel("Month")
        plt.ylabel("Avg Sales")
        st.pyplot(fig)

    # Holiday effect
    with right2:
        st.markdown("### Holiday Effect (Avg Sales)")
        holiday_sales = df.groupby("StateHoliday")["Sales"].mean()

        fig = plt.figure(figsize=(7, 4))
        plt.bar(holiday_sales.index.astype(str), holiday_sales.values)
        plt.xlabel("StateHoliday")
        plt.ylabel("Avg Sales")
        st.pyplot(fig)

    st.markdown("---")

    colA, colB = st.columns(2)

    # Day of week effect
    with colA:
        st.markdown("### Avg Sales by DayOfWeek")
        dow_sales = df.groupby("DayOfWeek")["Sales"].mean().sort_index()

        fig = plt.figure(figsize=(7, 4))
        plt.bar(dow_sales.index.astype(str), dow_sales.values)
        plt.xlabel("DayOfWeek (1=Mon ... 7=Sun)")
        plt.ylabel("Avg Sales")
        st.pyplot(fig)

    # StoreType effect
    with colB:
        st.markdown("### Avg Sales by StoreType")
        storetype_sales = df.groupby("StoreType")["Sales"].mean().sort_values(ascending=False)

        fig = plt.figure(figsize=(7, 4))
        plt.bar(storetype_sales.index.astype(str), storetype_sales.values)
        plt.xlabel("StoreType")
        plt.ylabel("Avg Sales")
        st.pyplot(fig)


# ============================
# TAB 3: Figures (PNG)
# ============================
with tab3:
    st.subheader("Saved Figures (Generated by Pipeline)")

    st.write("These figures are saved in the `figures/` folder. Run pipeline again to regenerate.")

    figure_files = [
        "Figure1_Sales_Trend.png",
        "Figure2_Promo_vs_NoPromo.png",
        "Figure3_Holiday_Effect.png",
        "Figure4_Monthly_Seasonality.png",
        "Figure5_Correlation_Heatmap.png",
        "Figure6_Model_RMSE_Comparison.png",
        "Figure7_Actual_vs_Predicted.png",
        "Figure8_Feature_Importance.png",
        "Figure9_AvgSales_by_DayOfWeek.png",
        "Figure10_AvgSales_by_StoreType.png",
    ]

    available = []
    for f in figure_files:
        p = os.path.join(FIG_DIR, f)
        if os.path.exists(p):
            available.append(p)

    if not available:
        st.warning("No figures found in `figures/`. Run: `python scripts/pipeline.py`")
    else:
        # show in grid
        cols = st.columns(2)
        for i, path in enumerate(available):
            with cols[i % 2]:
                st.image(path, caption=os.path.basename(path), use_container_width=True)


# ============================
# TAB 4: Model Performance
# ============================
with tab4:
    st.subheader("Model Performance")

    st.markdown("### Model Comparison (RMSE)")
    models = ["Linear Regression", "Random Forest", "XGBoost"]
    rmse_values = [2636.46, 1167.59, 508.97]

    fig = plt.figure(figsize=(8, 4))
    plt.bar(models, rmse_values)
    plt.ylabel("RMSE")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.markdown("### Interpretation")
    st.write(
        "XGBoost performed best because it models nonlinear relationships and interaction effects "
        "between promotions, seasonality, store attributes, and competition features."
    )


# ============================
# TAB 5: Forecast Output
# ============================
with tab5:
    st.subheader("Forecast Output")

    if submission is None:
        st.warning("submission.csv not found. Run pipeline first: `python scripts/pipeline.py`")
    else:
        st.markdown("### Preview: data/submission.csv")
        st.dataframe(submission.head(25))

        st.download_button(
            label="⬇️ Download submission.csv",
            data=submission.to_csv(index=False),
            file_name="submission.csv",
            mime="text/csv"
        )

        st.success("This file contains final forecasted sales predictions for the test dataset.")
