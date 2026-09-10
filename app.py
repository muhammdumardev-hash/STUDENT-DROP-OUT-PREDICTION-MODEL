"""
Student Dropout Prediction — Streamlit App

Covers:
- Dataset Overview
- Exploratory Data Analysis
- Logistic Regression Model
- Model Evaluation
- Phase 7: Student Risk Prediction

Dataset:
    datasett.csv

Run locally:
    streamlit run app.py

Deployment:
    Push app.py + requirements.txt + datasett.csv to GitHub
    and deploy using Streamlit Community Cloud.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL APP BACKGROUND
    ====================================================== */

    .stApp {
        background: #0f172a !important;
    }

    .main {
        background: #0f172a !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }


    /* ======================================================
       GLOBAL TEXT
    ====================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stText,
    p,
    li {
        color: #ffffff !important;
    }

    strong,
    b {
        color: #ffffff !important;
        font-weight: 750 !important;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #ffffff !important;
    }

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: #ffffff !important;
    }


    /* ======================================================
       MAIN TITLE
    ====================================================== */

    .main-title {
        color: #ffffff !important;
        font-size: 2.25rem;
        font-weight: 850;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #e0e0e0 !important;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        line-height: 1.65;
    }


    /* ======================================================
       METRIC CARDS
    ====================================================== */

    .metric-card {
        background: #ffffff !important;
        border: 1px solid #d1d5db;
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 108px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.20);
    }

    .metric-title {
        color: #1e1e1e !important;
        font-size: 0.90rem;
        font-weight: 750;
        letter-spacing: 0.1px;
    }

    .metric-value {
        color: #000000 !important;
        font-size: 1.85rem;
        font-weight: 850;
        margin-top: 5px;
    }


    /* ======================================================
       SECTION BOX
    ====================================================== */

    .section-box {
        background: #111827 !important;
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 15px;
    }


    /* ======================================================
       RESULT CARDS
    ====================================================== */

    .result-card {
        border-radius: 14px;
        padding: 22px;
        margin-top: 5px;
        margin-bottom: 10px;
    }

    .result-low {
        background: #f0fdf4 !important;
        border: 2px solid #86efac;
    }

    .result-medium {
        background: #fffbeb !important;
        border: 2px solid #fcd34d;
    }

    .result-high {
        background: #fef2f2 !important;
        border: 2px solid #fca5a5;
    }

    .result-title {
        color: #1e1e1e !important;
        font-size: 1.05rem;
        font-weight: 750;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 850;
        margin-top: 5px;
    }

    .result-low .result-value {
        color: #15803d !important;
    }

    .result-medium .result-value {
        color: #b45309 !important;
    }

    .result-high .result-value {
        color: #b91c1c !important;
    }


    /* ======================================================
       RISK BADGES
    ====================================================== */

    .risk-badge {
        display: inline-block;
        padding: 11px 24px;
        border-radius: 999px;
        font-weight: 850;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 10px;
    }

    .risk-low {
        background: #dcfce7 !important;
        color: #15803d !important;
        border: 2px solid #86efac;
    }

    .risk-medium {
        background: #fef3c7 !important;
        color: #b45309 !important;
        border: 2px solid #fcd34d;
    }

    .risk-high {
        background: #fee2e2 !important;
        color: #b91c1c !important;
        border: 2px solid #fca5a5;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #334155 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p {
        color: #e0e0e0 !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* ======================================================
       SIDEBAR RADIO
    ====================================================== */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label p {
        color: #ffffff !important;
    }


    /* ======================================================
       CAPTIONS
    ====================================================== */

    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #e0e0e0 !important;
        font-size: 0.90rem !important;
    }


    /* ======================================================
       TABS
    ====================================================== */

    button[data-baseweb="tab"] {
        color: #d1d5db !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"] p {
        color: #d1d5db !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 850 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
    }


    /* ======================================================
       EXPANDERS
    ====================================================== */

    [data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 750 !important;
    }

    [data-testid="stExpander"] summary p {
        color: #ffffff !important;
        font-weight: 750 !important;
    }


    /* ======================================================
       FORM LABELS
    ====================================================== */

    label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #ffffff !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }


    /* ======================================================
       INPUTS
    ====================================================== */

    input {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    textarea {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #6b7280 !important;
    }


    /* ======================================================
       SELECTBOX
    ====================================================== */

    div[data-baseweb="select"] {
        color: #111827 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #9ca3af !important;
    }

    div[data-baseweb="select"] * {
        color: #111827 !important;
    }


    /* ======================================================
       SELECTBOX DROPDOWN
    ====================================================== */

    div[role="listbox"] {
        background: #ffffff !important;
    }

    div[role="option"] {
        color: #111827 !important;
        background: #ffffff !important;
    }

    div[role="option"]:hover {
        background: #f3f4f6 !important;
    }


    /* ======================================================
       BUTTONS
    ====================================================== */

    button {
        font-weight: 700 !important;
    }

    button p {
        font-weight: 700 !important;
    }


    /* ======================================================
       DATAFRAME
    ====================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        overflow: hidden;
        background: #ffffff !important;
    }


    /* ======================================================
       ALERTS
    ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    [data-testid="stAlert"] p {
        color: #111827 !important;
        font-weight: 550;
    }

    [data-testid="stAlert"] strong {
        color: #111827 !important;
    }


    /* ======================================================
       DIVIDERS
    ====================================================== */

    hr {
        border-color: #334155 !important;
    }


    /* ======================================================
       CONTAINER BORDERS
    ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #334155 !important;
        border-radius: 14px !important;
    }


    /* ======================================================
       NUMBER INPUT / TEXT INPUT
    ====================================================== */

    div[data-baseweb="input"] {
        background: #ffffff !important;
        border-radius: 8px !important;
    }


    /* ======================================================
       CHECKBOX / RADIO TEXT
    ====================================================== */

    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: #ffffff !important;
    }

    [data-testid="stCheckbox"] label p,
    [data-testid="stRadio"] label p {
        color: #ffffff !important;
    }


    /* ======================================================
       FILE / MISC STREAMLIT TEXT
    ====================================================== */

    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DEFAULT_FILE = "datasett.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    df = pd.read_csv(DEFAULT_FILE)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# CLEAN DATA
# ============================================================

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):

    df = df.copy()

    if "target" not in df.columns:

        raise ValueError(
            "The dataset must contain a column named 'target'."
        )

    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
    )

    df["target"] = df["target"].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0,
        }
    )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="raise"
    ).astype(int)

    return df


# ============================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

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
        y_prob
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,

        "X_train": X_train,
        "X_test": X_test,

        "y_train": y_train,
        "y_test": y_test,

        "y_pred": y_pred,
        "y_prob": y_prob,

        "cm": cm,
        "report": report,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,

        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


# ============================================================
# LOAD + CLEAN DATASET
# ============================================================

try:

    raw_df = load_data()

    df = clean_data(
        raw_df
    )

except FileNotFoundError:

    st.error(
        "❌ datasett.csv was not found."
    )

    st.info(
        "Make sure datasett.csv is in the same GitHub "
        "repository/folder as app.py."
    )

    st.stop()

except Exception as error:

    st.error(
        "❌ Error while loading the dataset."
    )

    st.exception(error)

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

try:

    results = train_model(
        df
    )

except Exception as error:

    st.error(
        "❌ Model training failed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🎓 Dropout Predictor"
)

st.sidebar.caption(
    "Logistic Regression • Internship Project"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "🤖 Model Performance",
        "🔮 Predict Risk",
    ],
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Total Students",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Dropout Rate",
    f"{df['target'].mean() * 100:.1f}%"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Dataset is automatically loaded from "
    "`datasett.csv`."
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<p class="main-title">'
        '🎓 Student Dropout Prediction'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Logistic Regression model to identify students '
        'who may be at risk of dropping out.'
        '</p>',
        unsafe_allow_html=True
    )

    total_students = len(df)

    dropout_students = int(
        (df["target"] == 1).sum()
    )

    non_dropout_students = int(
        (df["target"] == 0).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Students</div>
            <div class="metric-value">{total_students:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Dropout Students</div>
            <div class="metric-value">{dropout_students:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Not Dropout</div>
            <div class="metric-value">{non_dropout_students:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Dropout Rate</div>
            <div class="metric-value">
                {dropout_students / total_students * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader(
        "Project Summary"
    )

    with st.container(border=True):

        s1, s2, s3, s4 = st.columns(4)

        s1.write("**Machine Learning Task**")
        s1.write("Binary Classification")

        s2.write("**Algorithm**")
        s2.write("Logistic Regression")

        s3.write("**Input Features**")
        s3.write(f"{df.shape[1] - 1}")

        s4.write("**Train / Test Split**")
        s4.write("80% / 20%")

    st.write("")

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Dataset Preview",
            "🔤 Data Types",
            "🎯 Target Distribution",
        ]
    )

    with tab1:

        st.subheader(
            "Dataset Preview"
        )

        with st.container(border=True):

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

    with tab2:

        st.subheader(
            "Column Data Types"
        )

        dtypes_df = pd.DataFrame(
            {
                "Column": df.dtypes.index,
                "Data Type": df.dtypes.astype(str).values,
            }
        )

        with st.container(border=True):

            st.dataframe(
                dtypes_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )

    with tab3:

        st.subheader(
            "Target Distribution"
        )

        target_data = pd.DataFrame(
            {
                "Status": [
                    "Not Dropout",
                    "Dropout",
                ],
                "Students": [
                    non_dropout_students,
                    dropout_students,
                ],
            }
        )

        target_data["Percentage"] = (
            target_data["Students"]
            / total_students
            * 100
        )

        target_data["Label"] = target_data.apply(
            lambda row:
            f"{int(row['Students']):,} "
            f"({row['Percentage']:.1f}%)",
            axis=1
        )

        fig = px.bar(
            target_data,
            x="Status",
            y="Students",
            color="Status",
            text="Label",
            color_discrete_map={
                "Not Dropout": "#3b82f6",
                "Dropout": "#ef4444",
            },
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont=dict(
                color="#000000",
                size=13
            )
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Students",
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(
                l=55,
                r=30,
                t=50,
                b=55
            ),
            font=dict(
                color="#111827"
            ),
        )

        fig.update_xaxes(
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000"
            )
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000"
            )
        )

        with st.container(border=True):

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.info(
            "This chart shows the distribution of students "
            "between the two project classes. Most students "
            "in the dataset are classified as Not Dropout, "
            "while a smaller group is classified as Dropout."
        )


# ============================================================
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "📊 Exploratory Analysis":

    st.markdown(
        '<p class="main-title">'
        '📊 Exploratory Data Analysis'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Visual patterns behind student dropout.'
        '</p>',
        unsafe_allow_html=True
    )

    plot_df = df.copy()

    plot_df["Status"] = (
        plot_df["target"]
        .map(
            {
                0: "Not Dropout",
                1: "Dropout",
            }
        )
    )

    # ========================================================
    # ROW 1
    # ========================================================

    col1, col2 = st.columns(2)

    # ========================================================
    # DROPOUT DISTRIBUTION
    # ========================================================

    with col1:

        st.subheader(
            "Dropout vs Not Dropout"
        )

        counts = (
            plot_df["Status"]
            .value_counts()
            .reindex(
                [
                    "Not Dropout",
                    "Dropout",
                ]
            )
            .reset_index()
        )

        counts.columns = [
            "Status",
            "Count",
        ]

        counts["Percentage"] = (
            counts["Count"]
            / len(plot_df)
            * 100
        )

        counts["Label"] = counts.apply(
            lambda row:
            f"{int(row['Count']):,} "
            f"({row['Percentage']:.1f}%)",
            axis=1
        )

        fig = px.bar(
            counts,
            x="Status",
            y="Count",
            color="Status",
            text="Label",
            color_discrete_map={
                "Not Dropout": "#3b82f6",
                "Dropout": "#ef4444",
            },
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont=dict(
                color="#000000",
                size=13
            )
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="Number of Students",
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(
                l=55,
                r=30,
                t=50,
                b=55
            ),
            font=dict(
                color="#111827"
            ),
        )

        fig.update_xaxes(
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000"
            )
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000"
            )
        )

        with st.container(border=True):

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown(
            """
            **What this shows:**   
            The chart compares the number of students who
            dropped out with those who did not.

            **Observation:**   
            The dataset contains more **Not Dropout** students
            than **Dropout** students. Approximately 67.9% of
            students are Not Dropout, while 32.1% are Dropout.

            This difference is important because the model needs
            to correctly identify the smaller but important
            dropout group.
            """
        )

    # ========================================================
    # TUITION FEES
    # ========================================================

    with col2:

        if "Tuition fees up to date" in df.columns:

            st.subheader(
                "Tuition Fees Status vs Dropout Rate"
            )

            rate = (
                plot_df
                .groupby(
                    "Tuition fees up to date"
                )["target"]
                .mean()
                .reset_index()
            )

            rate[
                "Tuition fees up to date"
            ] = rate[
                "Tuition fees up to date"
            ].map(
                {
                    0: "Not Up to Date",
                    1: "Up to Date",
                }
            )

            rate["Dropout Rate (%)"] = (
                rate["target"] * 100
            )

            rate["Label"] = rate[
                "Dropout Rate (%)"
            ].apply(
                lambda x:
                f"{x:.1f}%"
            )

            fig = px.bar(
                rate,
                x="Tuition fees up to date",
                y="Dropout Rate (%)",
                color="Tuition fees up to date",
                text="Label",
                color_discrete_map={
                    "Not Up to Date": "#f59e0b",
                    "Up to Date": "#10b981",
                },
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                textfont=dict(
                    color="#000000",
                    size=13
                )
            )

            fig.update_layout(
                showlegend=False,
                xaxis_title="Tuition Fee Status",
                yaxis_title="Dropout Rate (%)",
                height=450,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(
                    l=55,
                    r=30,
                    t=50,
                    b=55
                ),
                font=dict(
                    color="#111827"
                ),
            )

            fig.update_xaxes(
                showline=True,
                linecolor="#6b7280",
                tickfont=dict(
                    color="#000000",
                    size=13
                ),
                title_font=dict(
                    color="#000000"
                )
            )

            fig.update_yaxes(
                range=[0, 100],
                dtick=20,
                ticksuffix="%",
                showgrid=True,
                gridcolor="#e5e7eb",
                showline=True,
                linecolor="#6b7280",
                tickfont=dict(
                    color="#000000",
                    size=13
                ),
                title_font=dict(
                    color="#000000"
                )
            )

            with st.container(border=True):

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.markdown(
                """
                **What this shows:**   
                This chart compares the dropout rate of students
                according to whether their tuition fees were
                up to date.

                **Observation:**   
                Students whose tuition fees were **not up to date**
                have a much higher dropout rate than students whose
                fees were up to date.

                In this dataset, the dropout rate is approximately
                **86.6%** for students whose fees were not up to date,
                compared with approximately **24.7%** for students
                whose fees were up to date.

                This is a strong relationship in the dataset, but
                it should not be interpreted as proof that fee status
                directly causes dropout.
                """
            )

        else:

            st.info(
                "'Tuition fees up to date' column "
                "was not found."
            )

    # ========================================================
    # ROW 2
    # ========================================================

    col3, col4 = st.columns(2)

    # ========================================================
    # 1ST SEMESTER
    # ========================================================

    with col3:

        if (
            "Curricular units 1st sem (grade)"
            in df.columns
        ):

            st.subheader(
                "1st Semester Grade vs Student Status"
            )

            fig = px.box(
                plot_df,
                x="Status",
                y="Curricular units 1st sem (grade)",
                color="Status",
                points="outliers",
                color_discrete_map={
                    "Not Dropout": "#3b82f6",
                    "Dropout": "#ef4444",
                },
            )

            fig.update_traces(
                line=dict(
                    width=2
                ),
                marker=dict(
                    size=5,
                    opacity=0.55
                )
            )

            fig.update_layout(
                showlegend=False,
                height=500,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(
                    l=55,
                    r=30,
                    t=55,
                    b=55
                ),
                font=dict(
                    color="#111827"
                ),
            )

            fig.update_xaxes(
                title="Student Status",
                showline=True,
                linecolor="#6b7280",
                tickfont=dict(
                    color="#000000",
                    size=13
                ),
                title_font=dict(
                    color="#000000"
                )
            )

            fig.update_yaxes(
                range=[0, 20],
                dtick=2,
                title="1st Semester Grade",
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=True,
                showline=True,
                linecolor="#6b7280",
                tickfont=dict(
                    color="#000000",
                    size=13
                ),
                title_font=dict(
                    color="#000000"
                )
            )

            with st.container(border=True):

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.caption(
                "Grade scale: 0–20. Points outside the "
                "box represent statistical outliers."
            )

            st.markdown(
                """
                **What this shows:**   
                The box plot compares the distribution of
                first-semester grades between Dropout and
                Not Dropout students.

                **Observation:**   
                Dropout students generally have lower
                first-semester grades. The dropout group also
                contains many very low or zero grades.

                Non-dropout students show a higher and more
                stable grade distribution overall.

                This suggests that first-semester academic
                performance can be a useful indicator when
                identifying students who may require additional
                support.
                """
            )

    # ========================================================
    # 2ND SEMESTER
    # ========================================================

    with col4:

        if (
            "Curricular units 2nd sem (grade)"
            in df.columns
        ):

            st.subheader(
                "2nd Semester Grade vs Student Status"
            )

            fig = px.box(
                plot_df,
                x="Status",
                y="Curricular units 2nd sem (grade)",
                color="Status",
                points="outliers",
                color_discrete_map={
                    "Not Dropout": "#3b82f6",
                    "Dropout": "#ef4444",
                },
            )

            fig.update_traces(
                line=dict(
                    width=2
                ),
                marker=dict(
                    size=5,
                    opacity=0.55
                )
            )

            fig.update_layout(
                showlegend=False,
                height=500,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(
                    l=55,
                    r=30,
                    t=55,
                    b=55
                ),
                font=dict(
                    color="#111827"
                ),
            )

            fig.update_xaxes(
                title="Student Status",
                showline=True,
                linecolor="#6b7280",
                tickfont=dict(
                    color="#000000",
                    size=13
                ),
                title_font=dict(
                    color="#000000"
                )
            )

            fig.update_yaxes(
                range=[0, 20],
                dtick=2,
                title="2nd Semester Grade",
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=True,
                showline=True,
                linecolor="#6b7280",
                tickfont=dict(
                    color="#000000",
                    size=13
                ),
                title_font=dict(
                    color="#000000"
                )
            )

            with st.container(border=True):

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.caption(
                "Grade scale: 0–20. Points outside the "
                "box represent statistical outliers."
            )

            st.markdown(
                """
                **What this shows:**   
                This box plot compares second-semester grades
                for Dropout and Not Dropout students.

                **Observation:**   
                The difference between the two groups is more
                pronounced in the second semester.

                Dropout students have substantially lower grades,
                and a large number of dropout students have a
                second-semester grade of **0**.

                This makes second-semester academic performance
                an important warning indicator in this dataset.
                """
            )

    # ========================================================
    # KEY OBSERVATIONS
    # ========================================================

    st.write("")

    with st.container(border=True):

        st.subheader(
            "💡 Key Observations"
        )

        st.markdown(
            """
            **1. Student Distribution**

            Most students in the dataset are classified as
            **Not Dropout**, while approximately one-third are
            classified as **Dropout**.

            **2. Tuition Fee Status**

            Students whose tuition fees were not up to date show
            a much higher dropout rate than students whose fees
            were up to date.

            **3. 1st Semester Performance**

            Lower first-semester grades are associated with a
            higher proportion of dropout students.

            **4. 2nd Semester Performance**

            The relationship becomes stronger in the second
            semester, particularly because many dropout students
            have very low or zero grades.

            **Important:** These observations describe
            **relationships in the dataset**. They do not prove
            that any individual factor directly causes a student
            to drop out.
            """
        )

    # ========================================================
    # CONSTANT COLUMNS
    # ========================================================

    with st.expander(
        "🔎 Constant Columns / Zero Variance"
    ):

        const_cols = (
            df.nunique()[
                df.nunique() == 1
            ]
        )

        if len(const_cols) == 0:

            st.success(
                "No constant columns were found."
            )

        else:

            with st.container(border=True):

                st.dataframe(
                    const_cols.rename(
                        "Unique Values"
                    ),
                    use_container_width=True
                )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<p class="main-title">'
        '🤖 Model Performance'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Logistic Regression evaluated on the held-out test set.'
        '</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">
                {results["accuracy"] * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">
                {results["precision"] * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">
                {results["recall"] * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">F1-Score</div>
            <div class="metric-value">
                {results["f1"] * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c5.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">ROC-AUC</div>
            <div class="metric-value">
                {results["roc_auc"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    with col1:

        st.subheader(
            "Confusion Matrix"
        )

        cm = results["cm"]

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Students",
            },
            x=[
                "Not Dropout",
                "Dropout",
            ],
            y=[
                "Not Dropout",
                "Dropout",
            ],
        )

        fig_cm.update_traces(
            textfont=dict(
                color="#000000",
                size=18
            )
        )

        fig_cm.update_layout(
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                color="#000000"
            ),
            margin=dict(
                l=75,
                r=30,
                t=55,
                b=65
            ),
        )

        fig_cm.update_xaxes(
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000",
                size=14
            )
        )

        fig_cm.update_yaxes(
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000",
                size=14
            )
        )

        with st.container(border=True):

            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )

    # ========================================================
    # ROC CURVE
    # ========================================================

    with col2:

        st.subheader(
            "ROC Curve"
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=results["fpr"],
                y=results["tpr"],
                mode="lines",
                name="Logistic Regression",
                line=dict(
                    color="#3b82f6",
                    width=3
                ),
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(
                    color="#6b7280",
                    dash="dash"
                ),
            )
        )

        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                color="#000000"
            ),
            legend=dict(
                font=dict(
                    color="#000000",
                    size=12
                )
            ),
            margin=dict(
                l=65,
                r=30,
                t=55,
                b=65
            ),
        )

        fig_roc.update_xaxes(
            range=[0, 1],
            dtick=0.2,
            showgrid=True,
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000",
                size=14
            )
        )

        fig_roc.update_yaxes(
            range=[0, 1],
            dtick=0.2,
            showgrid=True,
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#6b7280",
            tickfont=dict(
                color="#000000",
                size=13
            ),
            title_font=dict(
                color="#000000",
                size=14
            )
        )

        with st.container(border=True):

            st.plotly_chart(
                fig_roc,
                use_container_width=True
            )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "Classification Report"
    )

    report_df = (
        pd.DataFrame(
            results["report"]
        )
        .transpose()
        .round(3)
    )

    with st.container(border=True):

        st.dataframe(
            report_df,
            use_container_width=True
        )

    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    st.subheader(
        "Prediction Details"
    )

    tn, fp, fn, tp = (
        results["cm"].ravel()
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "True Negative",
        int(tn)
    )

    e2.metric(
        "False Positive",
        int(fp)
    )

    e3.metric(
        "False Negative",
        int(fn)
    )

    e4.metric(
        "True Positive",
        int(tp)
    )

    st.info(
        f"**False Negatives:** {fn} actual dropout students "
        "were predicted as Not Dropout.\n\n"
        f"**False Positives:** {fp} students were predicted "
        "as Dropout but were actually Not Dropout."
    )


# ============================================================
# PAGE 4 — PHASE 7: PREDICT RISK
# ============================================================

elif page == "🔮 Predict Risk":

    st.markdown(
        '<p class="main-title">'
        '🔮 Student Risk Prediction'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Enter student information to estimate dropout probability '
        'and risk category.'
        '</p>',
        unsafe_allow_html=True
    )

    # ========================================================
    # FEATURE / TEST DATA
    # ========================================================

    feature_names = results[
        "feature_names"
    ]

    X_train_raw = results[
        "X_train"
    ]

    X_test_raw = results[
        "X_test"
    ]

    y_test_raw = results[
        "y_test"
    ]

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "sample_index" not in st.session_state:

        st.session_state.sample_index = None

    # ========================================================
    # QUICK TEST CASE
    # ========================================================

    with st.container(border=True):

        st.subheader(
            "🎲 Quick Test Case"
        )

        st.caption(
            "Load a real unseen student record from the test set. "
            "All 36 input features will be automatically loaded "
            "into the prediction fields."
        )

        if st.button(
            "Load Random Test Student",
            use_container_width=True
        ):

            # ------------------------------------------------
            # Select one REAL unseen student from X_test
            # ------------------------------------------------

            random_index = np.random.choice(
                X_test_raw.index
            )

            # ------------------------------------------------
            # Save selected student's original index
            # ------------------------------------------------

            st.session_state.sample_index = (
                random_index
            )

            # ------------------------------------------------
            # IMPORTANT FIX
            #
            # Load ALL 36 test-student feature values
            # into Streamlit session state.
            #
            # Target is NOT included.
            # ------------------------------------------------

            selected_student = (
                X_test_raw.loc[
                    random_index
                ]
            )

            for col_name in feature_names:

                st.session_state[
                    f"input_{col_name}"
                ] = selected_student[
                    col_name
                ]

            # ------------------------------------------------
            # Clear previous prediction
            # ------------------------------------------------

            st.session_state.pop(
                "last_prediction",
                None
            )

            st.session_state.pop(
                "last_probability",
                None
            )

            st.session_state.pop(
                "last_input_index",
                None
            )

            # ------------------------------------------------
            # Rerun so all widgets receive the selected values
            # ------------------------------------------------

            st.rerun()

    # ========================================================
    # CURRENT SAMPLE INDEX
    # ========================================================

    sample_index = (
        st.session_state.sample_index
    )

    if sample_index is not None:

        st.success(
            f"Random test student loaded "
            f"(record index: {sample_index})."
        )

        st.caption(
            "This student belongs to the held-out test set. "
            "The target value is kept separate and is not used "
            "as an input feature."
        )

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    if sample_index is not None:

        # Use the selected test student's REAL values
        defaults = X_test_raw.loc[
            sample_index
        ]

    else:

        defaults = None

    # ========================================================
    # PREDICTION FORM
    # ========================================================

    with st.form(
        "prediction_form"
    ):

        st.markdown(
            "### 👤 Demographic Information"
        )

        demographic_columns = [
            "Marital status",
            "Nacionality",
            "Gender",
            "Age at enrollment",
            "International",
            "Displaced",
            "Educational special needs",
        ]

        demographic_columns = [
            col
            for col in demographic_columns
            if col in feature_names
        ]

        enrollment_columns = [
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance",
            "Previous qualification",
            "Mother's qualification",
            "Father's qualification",
            "Mother's occupation",
            "Father's occupation",
        ]

        enrollment_columns = [
            col
            for col in enrollment_columns
            if col in feature_names
        ]

        academic_columns = [
            "Previous qualification (grade)",
            "Admission grade",

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
            "Curricular units 2nd sem (without evaluations)",
        ]

        academic_columns = [
            col
            for col in academic_columns
            if col in feature_names
        ]

        financial_columns = [
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
            "Unemployment rate",
            "Inflation rate",
            "GDP",
        ]

        financial_columns = [
            col
            for col in financial_columns
            if col in feature_names
        ]

        grouped_columns = (
            demographic_columns
            + enrollment_columns
            + academic_columns
            + financial_columns
        )

        other_columns = [
            col
            for col in feature_names
            if col not in grouped_columns
        ]

        inputs = {}

        # ====================================================
        # INPUT FUNCTION
        # ====================================================

        def render_input_group(
            title,
            columns
        ):

            if not columns:
                return

            st.markdown(
                f"#### {title}"
            )

            input_cols = st.columns(3)

            for i, col_name in enumerate(
                columns
            ):

                # ------------------------------------------------
                # IMPORTANT FIX:
                #
                # Use COMPLETE dataset for available values.
                # Do NOT use X_train here.
                #
                # A value may exist in X_test even if that exact
                # value does not appear in X_train.
                # ------------------------------------------------

                series = df[
                    col_name
                ]

                unique_values = sorted(
                    series
                    .dropna()
                    .unique()
                    .tolist()
                )

                # ------------------------------------------------
                # Determine default value
                # ------------------------------------------------

                if defaults is not None:

                    default_value = defaults[
                        col_name
                    ]

                else:

                    default_value = series.median()

                with input_cols[
                    i % 3
                ]:

                    # =================================================
                    # CATEGORICAL / LOW-UNIQUE-VALUE COLUMNS
                    # =================================================

                    if len(unique_values) <= 10:

                        # ------------------------------------------------
                        # Make sure selected/default value exists
                        # in the dropdown options.
                        # ------------------------------------------------

                        if (
                            default_value
                            in unique_values
                        ):

                            selected_index = (
                                unique_values.index(
                                    default_value
                                )
                            )

                        else:

                            selected_index = 0

                        binary_columns = [
                            "Debtor",
                            "Tuition fees up to date",
                            "Scholarship holder",
                            "Displaced",
                            "Educational special needs",
                            "International",
                        ]

                        # =================================================
                        # BINARY COLUMNS
                        # =================================================

                        if (
                            col_name
                            in binary_columns
                        ):

                            def format_binary(
                                value
                            ):

                                if value == 1:
                                    return "Yes (1)"

                                return "No (0)"

                            value = st.selectbox(
                                col_name,
                                options=unique_values,
                                index=selected_index,
                                format_func=format_binary,
                                key=f"input_{col_name}",
                                help="0 = No, 1 = Yes",
                            )

                        # =================================================
                        # OTHER CATEGORICAL COLUMNS
                        # =================================================

                        else:

                            value = st.selectbox(
                                col_name,
                                options=unique_values,
                                index=selected_index,
                                key=f"input_{col_name}",
                                help="Value/code from the dataset.",
                            )

                    # =================================================
                    # NUMERIC COLUMNS
                    # =================================================

                    else:

                        min_value = float(
                            series.min()
                        )

                        max_value = float(
                            series.max()
                        )

                        default_number = float(
                            default_value
                        )

                        # ------------------------------------------------
                        # Keep default value inside complete dataset range
                        # ------------------------------------------------

                        if (
                            default_number
                            < min_value
                        ):

                            default_number = (
                                min_value
                            )

                        if (
                            default_number
                            > max_value
                        ):

                            default_number = (
                                max_value
                            )

                        value = st.number_input(
                            col_name,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_number,
                            key=f"input_{col_name}",
                        )

                    # ------------------------------------------------
                    # Save current widget value
                    # ------------------------------------------------

                    inputs[
                        col_name
                    ] = value

        # ====================================================
        # RENDER FORM
        # ====================================================

        render_input_group(
            "👤 Demographic Information",
            demographic_columns
        )

        st.divider()

        render_input_group(
            "📝 Enrollment Information",
            enrollment_columns
        )

        st.divider()

        render_input_group(
            "📚 Academic Information",
            academic_columns
        )

        st.divider()

        render_input_group(
            "💰 Financial & Socioeconomic Information",
            financial_columns
        )

        if other_columns:

            st.divider()

            render_input_group(
                "➕ Additional Information",
                other_columns
            )

        st.write("")

        submitted = st.form_submit_button(
            "🔍 Predict Dropout Risk",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        # ----------------------------------------------------
        # Create input dataframe using ONLY 36 features
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [inputs]
        )

        input_df = input_df[
            feature_names
        ]

        # ----------------------------------------------------
        # Scale input
        # ----------------------------------------------------

        input_scaled = (
            results["scaler"]
            .transform(
                input_df
            )
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = (
            results["model"]
            .predict_proba(
                input_scaled
            )[0, 1]
        )

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_class = (
            results["model"]
            .predict(
                input_scaled
            )[0]
        )

        # ====================================================
        # SAVE PREDICTION IN SESSION STATE
        # ====================================================

        st.session_state.last_prediction = int(
            predicted_class
        )

        st.session_state.last_probability = float(
            probability
        )

        st.session_state.last_input_index = (
            sample_index
        )

        # ====================================================
        # RISK CATEGORY
        # ====================================================

        if probability < 0.30:

            risk_label = "Low Risk"
            risk_css = "risk-low"
            result_css = "result-low"
            risk_color = "#16a34a"

        elif probability < 0.60:

            risk_label = "Medium Risk"
            risk_css = "risk-medium"
            result_css = "result-medium"
            risk_color = "#d97706"

        else:

            risk_label = "High Risk"
            risk_css = "risk-high"
            result_css = "result-high"
            risk_color = "#dc2626"

        # ====================================================
        # RESULT
        # ====================================================

        st.write("---")

        st.markdown(
            "## 🎯 Prediction Result"
        )

        result_col1, result_col2 = st.columns(
            [1.1, 0.9]
        )

        # ====================================================
        # GAUGE
        # ====================================================

        with result_col1:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 38,
                            "color": "#000000"
                        }
                    },
                    title={
                        "text":
                        "Dropout Probability",
                        "font": {
                            "size": 18,
                            "color": "#000000"
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100
                            ],
                            "tickwidth": 1,
                            "tickcolor": "#374151",
                            "tickfont": {
                                "color": "#000000"
                            }
                        },

                        "bar": {
                            "color":
                            risk_color
                        },

                        "borderwidth": 1,
                        "bordercolor": "#9ca3af",

                        "steps": [
                            {
                                "range": [
                                    0,
                                    30
                                ],
                                "color":
                                "#dcfce7"
                            },
                            {
                                "range": [
                                    30,
                                    60
                                ],
                                "color":
                                "#fef3c7"
                            },
                            {
                                "range": [
                                    60,
                                    100
                                ],
                                "color":
                                "#fee2e2"
                            },
                        ],

                        "threshold": {
                            "line": {
                                "color": risk_color,
                                "width": 5
                            },
                            "thickness": 0.8,
                            "value": probability * 100
                        }
                    },
                )
            )

            gauge.update_layout(
                height=350,
                margin=dict(
                    l=25,
                    r=25,
                    t=55,
                    b=20
                ),
                paper_bgcolor="white",
                font=dict(
                    color="#000000"
                )
            )

            with st.container(
                border=True
            ):

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

        # ====================================================
        # RESULT DETAILS
        # ====================================================

        with result_col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"""
                    <div class="risk-badge {risk_css}">
                        {risk_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                st.markdown(
                    f"""
                    <div class="result-card {result_css}">
                        <div class="result-title">
                            Dropout Probability
                        </div>
                        <div class="result-value">
                            {probability * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                predicted_label = (
                    "Dropout"
                    if predicted_class == 1
                    else "Not Dropout"
                )

                if predicted_class == 1:

                    class_color = "#dc2626"
                    class_background = "#fef2f2"
                    class_border = "#fca5a5"

                else:

                    class_color = "#16a34a"
                    class_background = "#f0fdf4"
                    class_border = "#86efac"

                # ====================================================
                # FIXED PREDICTED CLASS CARD
                # ====================================================

                st.markdown(
                    f'<div style="background:{class_background};border:2px solid {class_border};border-radius:14px;padding:18px;margin-bottom:12px;">'
                    f'<div style="color:#374151 !important;font-size:1rem;font-weight:700;">Predicted Class</div>'
                    f'<div style="color:{class_color} !important;font-size:1.55rem;font-weight:800;margin-top:5px;">{predicted_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if predicted_class == 1:

                    st.warning(
                        "⚠️ The model predicts that this "
                        "student has a higher likelihood "
                        "of dropping out."
                    )

                else:

                    st.success(
                        "✅ The model predicts that this "
                        "student has a lower likelihood "
                        "of dropping out."
                    )

        # ====================================================
        # ACTUAL OUTCOME
        # ====================================================

        if sample_index is not None:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Confirm that the user did NOT modify the
            # automatically loaded test student's values.
            #
            # Only then show the actual outcome.
            # ------------------------------------------------

            original_values = (
                X_test_raw
                .loc[
                    sample_index,
                    feature_names
                ]
                .astype(float)
                .to_numpy()
            )

            entered_values = (
                input_df
                .iloc[0]
                .astype(float)
                .to_numpy()
            )

            same_as_original = np.allclose(
                original_values,
                entered_values,
                rtol=1e-5,
                atol=1e-8
            )

            if same_as_original:

                # ------------------------------------------------
                # IMPORTANT FIX:
                #
                # Get actual target from y_test.
                # This confirms it belongs to held-out test set.
                # ------------------------------------------------

                actual_outcome = int(
                    results["y_test"].loc[
                        sample_index
                    ]
                )

                actual_label = (
                    "Dropout"
                    if actual_outcome == 1
                    else "Not Dropout"
                )

                st.write("")

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "📌 Actual Outcome"
                    )

                    if actual_outcome == 1:

                        st.markdown(
                            """
                            <div class="result-card result-high">
                                <div class="result-title">
                                    Actual Student Outcome
                                </div>
                                <div class="result-value">
                                    🔴 Dropout
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="result-card result-low">
                                <div class="result-title">
                                    Actual Student Outcome
                                </div>
                                <div class="result-value">
                                    🟢 Not Dropout
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    if (
                        predicted_class
                        == actual_outcome
                    ):

                        st.success(
                            "✅ The model prediction "
                            "matches the actual outcome."
                        )

                    else:

                        st.warning(
                            "⚠️ The model prediction "
                            "does not match the actual outcome."
                        )

            else:

                st.info(
                    "ℹ️ The test student's input values were "
                    "modified. Therefore, the original actual "
                    "outcome is not shown for this prediction."
                )

        # ====================================================
        # RISK INTERPRETATION
        # ====================================================

        st.write("")

        with st.container(
            border=True
        ):

            st.subheader(
                "📖 Risk Interpretation"
            )

            r1, r2, r3 = st.columns(3)

            # ====================================================
            # LOW RISK
            # ====================================================

            r1.markdown(
                '<div style="background:#14532d;border:1px solid #86efac;border-radius:12px;padding:15px;">'
                '<h4 style="color:#15803d !important;">🟢 Low Risk</h4>'
                '<p style="color:#166534 !important;">Dropout probability below 30%.</p>'
                '</div>',
                unsafe_allow_html=True
            )

            # ====================================================
            # MEDIUM RISK
            # ====================================================

            r2.markdown(
                '<div style="background:#78350f;border:1px solid #fcd34d;border-radius:12px;padding:15px;">'
                '<h4 style="color:#b45309 !important;">🟡 Medium Risk</h4>'
                '<p style="color:#92400e !important;">Probability from 30% to below 60%.</p>'
                '</div>',
                unsafe_allow_html=True
            )

            # ====================================================
            # HIGH RISK
            # ====================================================

            r3.markdown(
                '<div style="background:#7f1d1d;border:1px solid #fca5a5;border-radius:12px;padding:15px;">'
                '<h4 style="color:#b91c1c !important;">🔴 High Risk</h4>'
                '<p style="color:#991b1b !important;">Probability of 60% or higher.</p>'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "These risk bands are project-level thresholds "
                "for interpreting model probability and are "
                "not official institutional policies."
            )
