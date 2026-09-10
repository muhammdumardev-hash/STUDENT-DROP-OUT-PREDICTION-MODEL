import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =================================================
       MAIN APP
    ================================================= */

    .stApp,
    .main {
        background: #0f172a !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =================================================
       GLOBAL TEXT
    ================================================= */

    body,
    p,
    span,
    label,
    div {
        color: #ffffff;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #ffffff !important;
    }


    /* =================================================
       MAIN TITLE
    ================================================= */

    .main-title {
        color: #ffffff !important;
        font-size: 2.25rem;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #e0e0e0 !important;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }


    /* =================================================
       METRIC CARDS
    ================================================= */

    .metric-card {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 108px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .metric-title {
        color: #1e1e1e !important;
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #000000 !important;
        font-size: 1.75rem;
        font-weight: 850;
    }


    /* =================================================
       SECTION BOX
    ================================================= */

    .section-box {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }


    /* =================================================
       PREDICTION RESULT CARDS
    ================================================= */

    .result-card {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    }

    .result-title {
        color: #94A3B8 !important;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 6px;
        letter-spacing: 0.2px;
    }

    .result-value {
        color: #FFFFFF !important;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.2;
    }


    /* =================================================
       RISK-SPECIFIC RESULT VALUES
    ================================================= */

    .result-low .result-value {
        color: #34D399 !important;
    }

    .result-medium .result-value {
        color: #FBBF24 !important;
    }

    .result-high .result-value {
        color: #F87171 !important;
    }


    /* =================================================
       PREDICTED CLASS CARD
    ================================================= */

    .predicted-class-card {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    }

    .predicted-class-title {
        color: #94A3B8 !important;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .predicted-class-value {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .predicted-dropout {
        color: #F87171 !important;
    }

    .predicted-not-dropout {
        color: #34D399 !important;
    }


    /* =================================================
       RISK INTERPRETATION CARDS
    ================================================= */

    .risk-info-card {
        background: #111827 !important;
        border-radius: 10px;
        padding: 18px;
        min-height: 155px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16);
    }

    .risk-info-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .risk-info-card p {
        margin: 0;
        line-height: 1.55;
        font-size: 0.92rem;
        font-weight: 500;
    }


    /* =================================================
       LOW RISK
    ================================================= */

    .risk-info-low {
        border: 1px solid #34D399;
    }

    .risk-info-low h4 {
        color: #34D399 !important;
    }

    .risk-info-low p {
        color: #A7F3D0 !important;
    }


    /* =================================================
       MEDIUM RISK
    ================================================= */

    .risk-info-medium {
        border: 1px solid #FBBF24;
    }

    .risk-info-medium h4 {
        color: #FBBF24 !important;
    }

    .risk-info-medium p {
        color: #FDE68A !important;
    }


    /* =================================================
       HIGH RISK
    ================================================= */

    .risk-info-high {
        border: 1px solid #F87171;
    }

    .risk-info-high h4 {
        color: #F87171 !important;
    }

    .risk-info-high p {
        color: #FECACA !important;
    }


    /* =================================================
       ACTUAL TEST OUTCOME
    ================================================= */

    .actual-outcome-card {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 18px 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    }

    .actual-outcome-card h4 {
        color: #FFFFFF !important;
        margin-top: 0;
        margin-bottom: 8px;
    }

    .actual-outcome-card p {
        color: #E2E8F0 !important;
        margin: 0;
    }

    .actual-outcome-card strong {
        color: #FFFFFF !important;
    }


    /* =================================================
       SIDEBAR
    ================================================= */

    section[data-testid="stSidebar"] {
        background: #111827 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p {
        color: #e0e0e0 !important;
    }

    section[data-testid="stSidebar"] .caption {
        color: #cbd5e1 !important;
    }


    /* =================================================
       TABS
    ================================================= */

    button[data-baseweb="tab"] {
        color: #ffffff !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
    }


    /* =================================================
       EXPANDERS
    ================================================= */

    div[data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
    }


    /* =================================================
       FORMS
    ================================================= */

    div[data-testid="stForm"] {
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        background: #111827;
    }

    div[data-testid="stForm"] label {
        color: #ffffff !important;
    }


    /* =================================================
       INPUTS
    ================================================= */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    div[data-baseweb="select"] * {
        color: #111827 !important;
    }

    input {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
    }


    /* =================================================
       DATAFRAME
    ================================================= */

    div[data-testid="stDataFrame"] {
        border: 1px solid #475569;
        border-radius: 10px;
        overflow: hidden;
        background: #ffffff;
    }


    /* =================================================
       CAPTIONS
    ================================================= */

    .stCaption,
    div[data-testid="stCaptionContainer"] {
        color: #e0e0e0 !important;
    }


    /* =================================================
       ALERTS
    ================================================= */

    div[data-testid="stAlert"] * {
        color: #111827 !important;
    }


    /* =================================================
       DIVIDERS
    ================================================= */

    hr {
        border-color: #334155 !important;
    }


    /* =================================================
       CONTAINERS
    ================================================= */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #334155 !important;
    }


    /* =================================================
       CHECKBOX / RADIO
    ================================================= */

    div[data-testid="stCheckbox"] label,
    div[data-testid="stRadio"] label {
        color: #ffffff !important;
    }


    /* =================================================
       MARKDOWN
    ================================================= */

    .stMarkdown {
        color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

DEFAULT_FILE = "datasett.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DEFAULT_FILE)

    df.columns = df.columns.str.strip()

    return df


# =========================================================
# CLEAN DATA
# =========================================================

@st.cache_data
def clean_data(df):

    df = df.copy()

    if "Target" in df.columns:
        target_column = "Target"

    elif "target" in df.columns:
        target_column = "target"

    else:
        raise ValueError(
            "Target column was not found in the dataset."
        )

    df = df.rename(
        columns={
            target_column: "target"
        }
    )

    df["target"] = df["target"].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0
        }
    )

    df["target"] = df["target"].astype(int)

    return df


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model(df):

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )

    y_dropout_probability = y_prob[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_dropout_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Dropout",
            "Dropout"
        ],
        output_dict=True,
        zero_division=0
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_dropout_probability
    )

    return {
        "model": model,
        "scaler": scaler,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }


# =========================================================
# LOAD + CLEAN DATA
# =========================================================

try:

    df_original = load_data()

    df = clean_data(
        df_original
    )

except FileNotFoundError:

    st.error(
        f"Dataset file `{DEFAULT_FILE}` was not found. "
        "Make sure datasett.csv is present in the GitHub repository."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error while loading dataset: {e}"
    )

    st.stop()


# =========================================================
# TRAIN
# =========================================================

try:

    results = train_model(
        df
    )

except Exception as e:

    st.error(
        f"Model training failed: {e}"
    )

    st.stop()


model = results["model"]
scaler = results["scaler"]

X_train_raw = results["X_train_raw"]
X_test_raw = results["X_test_raw"]

y_train = results["y_train"]
y_test = results["y_test"]

y_pred = results["y_pred"]
y_prob = results["y_prob"]

accuracy = results["accuracy"]
precision = results["precision"]
recall = results["recall"]
f1 = results["f1"]
roc_auc = results["roc_auc"]

cm = results["cm"]
report = results["report"]

fpr = results["fpr"]
tpr = results["tpr"]


# =========================================================
# BASIC DATA INFORMATION
# =========================================================

total_students = len(df)

dropout_count = int(
    df["target"].sum()
)

not_dropout_count = int(
    (df["target"] == 0).sum()
)

dropout_rate = (
    dropout_count / total_students
) * 100


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.45rem;
            font-weight:850;
            color:#ffffff;
            margin-bottom:5px;
        ">
            🎓 Student Dropout
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#cbd5e1;
            margin-bottom:20px;
        ">
            Prediction & Risk Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Exploratory Analysis",
            "🤖 Model Performance",
            "🔮 Predict Risk"
        ]
    )

    st.divider()

    st.markdown(
        "### Dataset"
    )

    st.metric(
        "Total Students",
        f"{total_students:,}"
    )

    st.metric(
        "Dropout Rate",
        f"{dropout_rate:.1f}%"
    )

    st.caption(
        "Dataset is automatically loaded from datasett.csv."
    )


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <div class="main-title">
            🎓 Student Dropout Prediction
        </div>

        <div class="subtitle">
            Machine Learning system for identifying students
            who may be at risk of dropping out.
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Total Students
                </div>

                <div class="metric-value">
                    {total_students:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Dropout Students
                </div>

                <div class="metric-value">
                    {dropout_count:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Not Dropout
                </div>

                <div class="metric-value">
                    {not_dropout_count:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Dropout Rate
                </div>

                <div class="metric-value">
                    {dropout_rate:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")

    st.markdown(
        "## 📌 Project Summary"
    )

    summary_cols = st.columns(4)


    with summary_cols[0]:

        st.markdown(
            """
            <div class="section-box">

                <h4>🎯 Objective</h4>

                <p>
                    Predict whether a student is likely to
                    drop out using academic, demographic,
                    socioeconomic and enrollment information.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with summary_cols[1]:

        st.markdown(
            """
            <div class="section-box">

                <h4>🧠 ML Algorithm</h4>

                <p>
                    Logistic Regression is used as the
                    classification algorithm.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with summary_cols[2]:

        st.markdown(
            """
            <div class="section-box">

                <h4>📊 Problem Type</h4>

                <p>
                    Binary Classification:
                    Dropout or Not Dropout.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with summary_cols[3]:

        st.markdown(
            """
            <div class="section-box">

                <h4>💡 Real-World Use</h4>

                <p>
                    Identify students who may require
                    academic, financial or counseling support.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "## 📁 Dataset Information"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Dataset Preview",
            "Data Types",
            "Target Distribution"
        ]
    )


    with tab1:

        st.dataframe(
            df_original.head(10),
            use_container_width=True,
            hide_index=True
        )


    with tab2:

        dtype_df = pd.DataFrame(
            {
                "Column": df_original.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df_original.dtypes
                ]
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True
        )


    with tab3:

        target_chart_df = pd.DataFrame(
            {
                "Status": [
                    "Not Dropout",
                    "Dropout"
                ],
                "Students": [
                    not_dropout_count,
                    dropout_count
                ]
            }
        )

        fig = px.bar(
            target_chart_df,
            x="Status",
            y="Students",
            text="Students",
            title="Student Dropout Distribution",
            category_orders={
                "Status": [
                    "Not Dropout",
                    "Dropout"
                ]
            }
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                color="black",
                size=14
            )
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            ),
            xaxis=dict(
                title="Student Status",
                color="black",
                showline=True,
                linecolor="black",
                mirror=True,
                gridcolor="#e5e7eb"
            ),
            yaxis=dict(
                title="Number of Students",
                color="black",
                showline=True,
                linecolor="black",
                mirror=True,
                gridcolor="#e5e7eb"
            ),
            height=450
        )

        fig.update_traces(
            marker_line_color="black",
            marker_line_width=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================

elif page == "📊 Exploratory Analysis":

    st.markdown(
        """
        <div class="main-title">
            📊 Exploratory Data Analysis
        </div>

        <div class="subtitle">
            Understanding academic, financial and
            demographic patterns related to student dropout.
        </div>
        """,
        unsafe_allow_html=True
    )


    plot_df = df.copy()

    plot_df["Status"] = plot_df["target"].map(
        {
            0: "Not Dropout",
            1: "Dropout"
        }
    )


    st.markdown(
        "## 1️⃣ Dropout Distribution"
    )

    distribution_df = (
        plot_df["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Students")
    )

    distribution_df["Status"] = pd.Categorical(
        distribution_df["Status"],
        categories=[
            "Not Dropout",
            "Dropout"
        ],
        ordered=True
    )

    distribution_df = distribution_df.sort_values(
        "Status"
    )

    fig1 = px.bar(
        distribution_df,
        x="Status",
        y="Students",
        text="Students",
        title="Dropout vs Not Dropout"
    )

    fig1.update_traces(
        textposition="outside",
        textfont=dict(
            color="black",
            size=14
        )
    )

    fig1.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            title="Status",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        yaxis=dict(
            title="Number of Students",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        height=430
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.info(
        "Most students in the dataset did not drop out, "
        "while a smaller group was classified as dropout."
    )


    st.markdown(
        "## 2️⃣ Tuition Fees Status vs Dropout Rate"
    )

    tuition_df = (
        plot_df.groupby("Tuition fees up to date")["target"]
        .agg(
            Count="count",
            Dropout_Rate="mean"
        )
        .reset_index()
    )

    tuition_df["Status"] = tuition_df[
        "Tuition fees up to date"
    ].map(
        {
            0: "Fees Not Up to Date",
            1: "Fees Up to Date"
        }
    )

    tuition_df["Dropout_Rate_Percent"] = (
        tuition_df["Dropout_Rate"] * 100
    )

    tuition_df = tuition_df.sort_values(
        "Tuition fees up to date"
    )


    fig2 = px.bar(
        tuition_df,
        x="Status",
        y="Dropout_Rate_Percent",
        text="Dropout_Rate_Percent",
        title="Dropout Rate by Tuition Fee Status"
    )

    fig2.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(
            color="black",
            size=14
        )
    )

    fig2.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            title="Tuition Fee Status",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        yaxis=dict(
            title="Dropout Rate",
            color="black",
            range=[0, 100],
            dtick=20,
            ticksuffix="%",
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        height=450
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info(
        "Students whose tuition fees were not up to date "
        "had a much higher dropout rate than students "
        "whose fees were up to date."
    )


    st.markdown(
        "## 3️⃣ 1st Semester Grade vs Dropout"
    )

    fig3 = px.box(
        plot_df,
        x="Status",
        y="Curricular units 1st sem (grade)",
        color="Status",
        points="outliers",
        title="1st Semester Grade Distribution"
    )

    fig3.update_traces(
        marker=dict(
            size=5,
            opacity=0.55
        )
    )

    fig3.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        showlegend=False,
        xaxis=dict(
            title="Student Status",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        yaxis=dict(
            title="1st Semester Grade",
            color="black",
            range=[0, 20],
            dtick=2,
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        height=480
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.info(
        "Dropout students generally show lower 1st semester "
        "grades compared with students who did not drop out."
    )


    st.markdown(
        "## 4️⃣ 2nd Semester Grade vs Dropout"
    )

    fig4 = px.box(
        plot_df,
        x="Status",
        y="Curricular units 2nd sem (grade)",
        color="Status",
        points="outliers",
        title="2nd Semester Grade Distribution"
    )

    fig4.update_traces(
        marker=dict(
            size=5,
            opacity=0.55
        )
    )

    fig4.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        showlegend=False,
        xaxis=dict(
            title="Student Status",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        yaxis=dict(
            title="2nd Semester Grade",
            color="black",
            range=[0, 20],
            dtick=2,
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        height=480
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.info(
        "Lower 2nd semester grades, especially zero grades, "
        "are strongly associated with dropout in this dataset."
    )


    st.markdown(
        "## 🔎 Key Observations"
    )

    observation_cols = st.columns(2)


    with observation_cols[0]:

        st.markdown(
            """
            <div class="section-box">

                <h4>📚 Academic Performance</h4>

                <p>
                    Lower 1st and 2nd semester grades are
                    associated with higher dropout levels.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with observation_cols[1]:

        st.markdown(
            """
            <div class="section-box">

                <h4>💰 Tuition Fee Status</h4>

                <p>
                    Students whose tuition fees were not
                    up to date had a substantially higher
                    dropout rate.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="section-box">

            <h4>⚠️ Important Note</h4>

            <p>
                These relationships show associations in the
                dataset. They do not prove that one factor
                directly causes student dropout.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    with st.expander(
        "🔍 Check Constant Columns"
    ):

        constant_columns = df.nunique()

        constant_columns = constant_columns[
            constant_columns == 1
        ]

        if len(constant_columns) == 0:

            st.success(
                "No constant columns were found."
            )

        else:

            st.dataframe(
                constant_columns.to_frame(
                    "Unique Values"
                ),
                use_container_width=True
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.markdown(
        """
        <div class="main-title">
            🤖 Logistic Regression Performance
        </div>

        <div class="subtitle">
            Evaluation of the trained classification model
            on unseen test data.
        </div>
        """,
        unsafe_allow_html=True
    )


    metric_cols = st.columns(5)


    with metric_cols[0]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Accuracy
                </div>

                <div class="metric-value">
                    {accuracy * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric_cols[1]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Precision
                </div>

                <div class="metric-value">
                    {precision * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric_cols[2]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Recall
                </div>

                <div class="metric-value">
                    {recall * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric_cols[3]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    F1 Score
                </div>

                <div class="metric-value">
                    {f1 * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric_cols[4]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    ROC-AUC
                </div>

                <div class="metric-value">
                    {roc_auc * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    st.markdown(
        "## 🔲 Confusion Matrix"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Not Dropout",
            "Actual Dropout"
        ],
        columns=[
            "Predicted Not Dropout",
            "Predicted Dropout"
        ]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix",
        aspect="auto"
    )

    fig_cm.update_traces(
        textfont=dict(
            color="black",
            size=18
        )
    )

    fig_cm.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            title="Predicted",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True
        ),
        yaxis=dict(
            title="Actual",
            color="black",
            showline=True,
            linecolor="black",
            mirror=True
        ),
        height=500
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    tn, fp, fn, tp = cm.ravel()

    cm_cols = st.columns(4)


    with cm_cols[0]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    True Negative
                </div>

                <div class="metric-value">
                    {tn}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with cm_cols[1]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    False Positive
                </div>

                <div class="metric-value">
                    {fp}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with cm_cols[2]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    False Negative
                </div>

                <div class="metric-value">
                    {fn}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with cm_cols[3]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    True Positive
                </div>

                <div class="metric-value">
                    {tp}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "## 📈 ROC Curve"
    )

    fig_roc = go.Figure()

    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.3f})",
            line=dict(
                width=3
            )
        )
    )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(
                dash="dash",
                width=2
            )
        )
    )

    fig_roc.update_layout(
        title="Receiver Operating Characteristic Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            color="black",
            range=[0, 1],
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        yaxis=dict(
            color="black",
            range=[0, 1],
            showline=True,
            linecolor="black",
            mirror=True,
            gridcolor="#e5e7eb"
        ),
        height=500
    )

    st.plotly_chart(
        fig_roc,
        use_container_width=True
    )


    st.markdown(
        "## 📋 Classification Report"
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df = report_df.drop(
        index=["accuracy"],
        errors="ignore"
    )

    report_df = report_df.round(3)

    st.dataframe(
        report_df,
        use_container_width=True
    )


    st.markdown(
        "## 🧠 Model Interpretation"
    )

    interpretation_cols = st.columns(2)


    with interpretation_cols[0]:

        st.markdown(
            f"""
            <div class="section-box">

                <h4>📊 Overall Performance</h4>

                <p>
                    The Logistic Regression model achieved
                    an accuracy of
                    <strong>{accuracy * 100:.2f}%</strong>
                    on the test dataset.
                </p>

                <p>
                    The ROC-AUC score was
                    <strong>{roc_auc * 100:.2f}%</strong>,
                    showing strong ability to distinguish
                    between the two classes.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with interpretation_cols[1]:

        st.markdown(
            f"""
            <div class="section-box">

                <h4>⚠️ Dropout Detection</h4>

                <p>
                    The model correctly identified
                    <strong>{tp}</strong> dropout students.
                </p>

                <p>
                    However, <strong>{fn}</strong> actual
                    dropout students were classified as
                    not dropout.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PREDICT RISK
# =========================================================

elif page == "🔮 Predict Risk":

    st.markdown(
        """
        <div class="main-title">
            🔮 Student Risk Prediction
        </div>

        <div class="subtitle">
            Enter student information to estimate the
            probability of dropout and determine the
            student's risk level.
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # FEATURE NAMES
    # =====================================================

    feature_names = list(
        X_train_raw.columns
    )


    # =====================================================
    # RANDOM TEST CASE
    # =====================================================

    if "sample_index" not in st.session_state:

        st.session_state.sample_index = None


    quick_col1, quick_col2 = st.columns(
        [1, 4]
    )


    with quick_col1:

        if st.button(
            "🎲 Quick Test Case",
            use_container_width=True
        ):

            st.session_state.sample_index = np.random.randint(
                0,
                len(X_test_raw)
            )

            st.rerun()


    if st.session_state.sample_index is not None:

        sample_index = st.session_state.sample_index

        sample_row = X_test_raw.iloc[
            sample_index
        ]

        st.info(
            "A random unseen test case has been loaded. "
            "You can modify the values before prediction."
        )

    else:

        sample_row = X_train_raw.iloc[0]


    # =====================================================
    # INPUT RENDER FUNCTION
    # =====================================================

    def render_input_group(
        columns,
        sample
    ):

        input_values = {}

        cols = st.columns(2)

        for i, column in enumerate(columns):

            with cols[i % 2]:

                series = df[column]

                unique_values = sorted(
                    series.dropna().unique().tolist()
                )

                current_value = sample[column]


                # -----------------------------------------
                # SMALL CATEGORICAL FEATURES
                # -----------------------------------------

                if len(unique_values) <= 10:

                    if set(unique_values).issubset(
                        {0, 1}
                    ):

                        selected = st.selectbox(
                            column,
                            options=[
                                0,
                                1
                            ],
                            index=(
                                1
                                if int(current_value) == 1
                                else 0
                            ),
                            format_func=lambda x:
                                "Yes"
                                if x == 1
                                else "No"
                        )

                    else:

                        current_index = 0

                        try:

                            current_index = unique_values.index(
                                current_value
                            )

                        except ValueError:

                            current_index = 0

                        selected = st.selectbox(
                            column,
                            options=unique_values,
                            index=current_index
                        )

                    input_values[column] = selected


                # -----------------------------------------
                # NUMERIC FEATURES
                # -----------------------------------------

                else:

                    min_value = float(
                        series.min()
                    )

                    max_value = float(
                        series.max()
                    )

                    step = 1.0

                    if (
                        series.dtype == np.float64
                        or series.dtype == np.float32
                    ):

                        step = 0.01

                    value = st.number_input(
                        column,
                        min_value=min_value,
                        max_value=max_value,
                        value=float(current_value),
                        step=step
                    )

                    input_values[column] = value

        return input_values


    # =====================================================
    # INPUT FORM
    # =====================================================

    with st.form(
        "prediction_form"
    ):

        # -------------------------------------------------
        # DEMOGRAPHIC INFORMATION
        # -------------------------------------------------

        st.markdown(
            "### 👤 Demographic Information"
        )

        demographic_features = [
            column
            for column in feature_names
            if column in [
                "Marital status",
                "Gender",
                "Age at enrollment",
                "Nationality",
                "International",
                "Educational special needs"
            ]
        ]

        demographic_values = render_input_group(
            demographic_features,
            sample_row
        )


        # -------------------------------------------------
        # ENROLLMENT INFORMATION
        # -------------------------------------------------

        st.markdown(
            "### 🎓 Enrollment Information"
        )

        enrollment_features = [
            column
            for column in feature_names
            if column in [
                "Application mode",
                "Application order",
                "Course",
                "Daytime/evening attendance",
                "Previous qualification",
                "Previous qualification (grade)",
                "Admission grade"
            ]
        ]

        enrollment_values = render_input_group(
            enrollment_features,
            sample_row
        )


        # -------------------------------------------------
        # ACADEMIC INFORMATION
        # -------------------------------------------------

        st.markdown(
            "### 📚 Academic Performance"
        )

        academic_features = [
            column
            for column in feature_names
            if column in [
                "Curricular units 1st sem (credited)",
                "Curricular units 1st sem (enrolled)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
                "Curricular units 1st sem (without evaluations)",
                "Curricular units 2nd sem (credited)",
                "Curricular units 2nd sem (enrolled)",
                "Curricular units 2nd sem (evaluations)",
                "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)",
                "Curricular units 2nd sem (without evaluations)"
            ]
        ]

        academic_values = render_input_group(
            academic_features,
            sample_row
        )


        # -------------------------------------------------
        # FINANCIAL INFORMATION
        # -------------------------------------------------

        st.markdown(
            "### 💰 Financial & Socioeconomic Information"
        )

        financial_features = [
            column
            for column in feature_names
            if column in [
                "Debtor",
                "Tuition fees up to date",
                "Scholarship holder",
                "Unemployment rate",
                "Inflation rate",
                "GDP"
            ]
        ]

        financial_values = render_input_group(
            financial_features,
            sample_row
        )


        # -------------------------------------------------
        # OTHER FEATURES
        # -------------------------------------------------

        used_features = (
            demographic_features
            + enrollment_features
            + academic_features
            + financial_features
        )

        other_features = [
            column
            for column in feature_names
            if column not in used_features
        ]

        if other_features:

            st.markdown(
                "### ⚙️ Additional Information"
            )

            other_values = render_input_group(
                other_features,
                sample_row
            )

        else:

            other_values = {}


        # -------------------------------------------------
        # SUBMIT
        # -------------------------------------------------

        submitted = st.form_submit_button(
            "🔍 Predict Student Risk",
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        input_values = {}

        input_values.update(
            demographic_values
        )

        input_values.update(
            enrollment_values
        )

        input_values.update(
            academic_values
        )

        input_values.update(
            financial_values
        )

        input_values.update(
            other_values
        )


        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_df = pd.DataFrame(
            [input_values]
        )

        input_df = input_df[
            feature_names
        ]


        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        input_scaled = scaler.transform(
            input_df
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction_probability = model.predict_proba(
            input_scaled
        )[0, 1]

        predicted_class = int(
            model.predict(
                input_scaled
            )[0]
        )


        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if prediction_probability < 0.30:

            risk_level = "Low Risk"
            result_class = "result-low"

        elif prediction_probability < 0.60:

            risk_level = "Medium Risk"
            result_class = "result-medium"

        else:

            risk_level = "High Risk"
            result_class = "result-high"


        probability_percent = (
            prediction_probability * 100
        )


        # =================================================
        # PREDICTED CLASS
        # =================================================

        if predicted_class == 1:

            predicted_class_text = "Dropout"
            predicted_class_style = "predicted-dropout"

        else:

            predicted_class_text = "Not Dropout"
            predicted_class_style = "predicted-not-dropout"


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            "## 🎯 Prediction Result"
        )


        result_col1, result_col2 = st.columns(
            [1.2, 1]
        )


        # -------------------------------------------------
        # GAUGE
        # -------------------------------------------------

        with result_col1:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percent,
                    title={
                        "text": "Dropout Probability"
                    },
                    number={
                        "suffix": "%",
                        "font": {
                            "color": "black",
                            "size": 40
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "black",
                            "tickfont": {
                                "color": "black"
                            }
                        },
                        "bar": {
                            "color": "#2563eb"
                        },
                        "bgcolor": "#ffffff",
                        "borderwidth": 2,
                        "bordercolor": "#111827"
                    }
                )
            )

            gauge.update_layout(
                paper_bgcolor="white",
                font=dict(
                    color="black"
                ),
                height=390
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )


        # -------------------------------------------------
        # RESULT DETAILS
        # -------------------------------------------------

        with result_col2:

            # ---------------------------------------------
            # RISK LEVEL
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="result-card {result_class}">

                    <div class="result-title">
                        Risk Level
                    </div>

                    <div class="result-value">
                        {risk_level}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ---------------------------------------------
            # DROPOUT PROBABILITY
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="result-card {result_class}">

                    <div class="result-title">
                        Dropout Probability
                    </div>

                    <div class="result-value">
                        {probability_percent:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ---------------------------------------------
            # PREDICTED CLASS
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="predicted-class-card">

                    <div class="predicted-class-title">
                        Predicted Class
                    </div>

                    <div class="predicted-class-value {predicted_class_style}">
                        {predicted_class_text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # INTERPRETATION MESSAGE
        # =================================================

        if predicted_class == 1:

            st.warning(
                "The model predicts that this student may be "
                "at risk of dropping out."
            )

        else:

            st.success(
                "The model predicts that this student is "
                "likely to remain enrolled or graduate."
            )


        # =================================================
        # ACTUAL TEST DATASET OUTCOME
        # =================================================

        if st.session_state.sample_index is not None:

            actual_index = st.session_state.sample_index

            actual_target = int(
                y_test.iloc[
                    actual_index
                ]
            )

            actual_text = (
                "Dropout"
                if actual_target == 1
                else "Not Dropout"
            )

            st.markdown(
                f"""
                <div class="actual-outcome-card">

                    <h4>
                        📌 Actual Test Dataset Outcome
                    </h4>

                    <p>
                        Actual outcome for this unseen test
                        student:
                        <strong>{actual_text}</strong>
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # RISK INTERPRETATION
        # =================================================

        st.markdown(
            "## 📖 Risk Interpretation"
        )

        risk_cols = st.columns(3)


        # -------------------------------------------------
        # LOW RISK
        # -------------------------------------------------

        with risk_cols[0]:

            st.markdown(
                """
                <div class="risk-info-card risk-info-low">

                    <h4>
                        🟢 Low Risk
                    </h4>

                    <p>
                        Dropout probability is below 30%.
                        The model considers the student to
                        have a relatively low predicted risk.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # MEDIUM RISK
        # -------------------------------------------------

        with risk_cols[1]:

            st.markdown(
                """
                <div class="risk-info-card risk-info-medium">

                    <h4>
                        🟡 Medium Risk
                    </h4>

                    <p>
                        Dropout probability is between 30%
                        and 60%. The student may benefit
                        from additional monitoring.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # HIGH RISK
        # -------------------------------------------------

        with risk_cols[2]:

            st.markdown(
                """
                <div class="risk-info-card risk-info-high">

                    <h4>
                        🔴 High Risk
                    </h4>

                    <p>
                        Dropout probability is above 60%.
                        The student may require closer
                        academic or support intervention.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        margin-top:45px;
        padding-top:20px;
        border-top:1px solid #334155;
        font-size:0.9rem;
    ">
        Student Dropout Prediction • Logistic Regression
        • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
