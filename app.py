import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
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
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
    }

    p, span, li {
        color: #1f2937;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] h2 {
        color: #111827 !important;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );

        border-radius: 22px;
        padding: 32px 35px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #d1d5db !important;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
    }


    /* ========================================================
       SECTION HEADER
    ======================================================== */

    .section-header {
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 3px;
    }

    .section-description {
        color: #6b7280;
        font-size: 0.95rem;
    }


    /* ========================================================
       KPI CARDS
    ======================================================== */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }

    .kpi-label {
        color: #6b7280 !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        color: #111827 !important;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 7px;
    }

    .kpi-small {
        color: #6b7280 !important;
        font-size: 0.8rem;
        margin-top: 4px;
    }


    /* ========================================================
       INFO CARDS
    ======================================================== */

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.035);
        margin-bottom: 15px;
    }

    .info-title {
        color: #111827;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .info-text {
        color: #4b5563;
        line-height: 1.6;
        font-size: 0.92rem;
    }


    /* ========================================================
       RISK CARDS
    ======================================================== */

    .risk-low {
        background: #ecfdf5;
        border: 1px solid #86efac;
        border-radius: 17px;
        padding: 22px;
    }

    .risk-medium {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 17px;
        padding: 22px;
    }

    .risk-high {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 17px;
        padding: 22px;
    }

    .risk-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .risk-probability {
        font-size: 2rem;
        font-weight: 900;
        margin-top: 8px;
    }


    /* ========================================================
       RESULT
    ======================================================== */

    .prediction-result {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.04);
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    .stFormSubmitButton > button {
        border-radius: 12px;
        font-weight: 800;
        min-height: 48px;
    }


    /* ========================================================
       DATAFRAME
    ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        overflow: hidden;
    }


    /* ========================================================
       TABS
    ======================================================== */

    button[data-baseweb="tab"] {
        color: #374151 !important;
        font-weight: 700;
    }


    /* ========================================================
       METRICS
    ======================================================== */

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }


    /* ========================================================
       FORM LABELS
    ======================================================== */

    label {
        color: #111827 !important;
        font-weight: 650 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

DATASET_FILE = "datasett.csv"

TARGET_COLUMN = "target"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    df = pd.read_csv(DATASET_FILE)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# CLEAN DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def clean_dataset(df):

    df = df.copy()

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            "The dataset must contain a column named 'target'."
        )

    # Clean target
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0,
        }
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise"
    ).astype(int)

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(show_spinner=True)
def train_model(df):

    X = df.drop(
        TARGET_COLUMN,
        axis=1
    )

    y = df[TARGET_COLUMN]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
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
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        y_pred
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
        y_probability
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    # Logistic Regression coefficients
    coefficients = model.coef_[0]

    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Absolute Impact": np.abs(coefficients)
        }
    ).sort_values(
        "Absolute Impact",
        ascending=False
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
        "y_probability": y_probability,

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

        "feature_importance": feature_importance,
    }


# ============================================================
# SAFE DATA LOADING
# ============================================================

try:

    raw_df = load_dataset()

    df = clean_dataset(
        raw_df
    )

except FileNotFoundError:

    st.error(
        "❌ datasett.csv was not found."
    )

    st.info(
        "Put datasett.csv in the same folder "
        "as app.py."
    )

    st.stop()

except Exception as error:

    st.error(
        "❌ Dataset loading failed."
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
# BASIC STATISTICS
# ============================================================

total_students = len(df)

dropout_students = int(
    (df[TARGET_COLUMN] == 1).sum()
)

non_dropout_students = int(
    (df[TARGET_COLUMN] == 0).sum()
)

dropout_rate = (
    dropout_students
    / total_students
    * 100
)

feature_count = (
    df.shape[1] - 1
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding: 8px 0 15px 0;
    ">
        <div style="
            font-size: 1.55rem;
            font-weight: 900;
            color: #111827;
        ">
            🎓 Dropout Predictor
        </div>

        <div style="
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 4px;
        ">
            Machine Learning Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Exploratory Analysis",
        "🤖 Model Performance",
        "🔮 Predict Risk",
    ]
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    "### 📌 Project Information"
)

st.sidebar.write(
    "**Algorithm:** Logistic Regression"
)

st.sidebar.write(
    "**Task:** Binary Classification"
)

st.sidebar.write(
    "**Split:** 80% Train / 20% Test"
)

st.sidebar.write(
    f"**Features:** {feature_count}"
)

st.sidebar.markdown("---")


st.sidebar.metric(
    "Students",
    f"{total_students:,}"
)

st.sidebar.metric(
    "Dropout Rate",
    f"{dropout_rate:.1f}%"
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Dataset: datasett.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def hero(title, subtitle):

    st.markdown(
        f"""
        <div class="hero">

            <div class="hero-title">
                {title}
            </div>

            <div class="hero-subtitle">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def section(title, description=""):

    st.markdown(
        f"""
        <div class="section-header">

            <div class="section-title">
                {title}
            </div>

            <div class="section-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_card(label, value, description=""):

    st.markdown(
        f"""
        <div class="kpi-card">

            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value">
                {value}
            </div>

            <div class="kpi-small">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def make_plotly_layout(fig, height=430):

    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="#111827"
        ),
        margin=dict(
            l=55,
            r=30,
            t=50,
            b=55
        ),
        hoverlabel=dict(
            font=dict(
                color="#111827"
            )
        )
    )

    fig.update_xaxes(
        showline=True,
        linecolor="#9ca3af",
        tickfont=dict(
            color="#111827"
        ),
        title_font=dict(
            color="#111827"
        ),
        gridcolor="#e5e7eb"
    )

    fig.update_yaxes(
        showline=True,
        linecolor="#9ca3af",
        tickfont=dict(
            color="#111827"
        ),
        title_font=dict(
            color="#111827"
        ),
        gridcolor="#e5e7eb"
    )

    return fig


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    hero(
        "🎓 Student Dropout Prediction",
        "Machine learning dashboard for identifying students who may be at risk of dropping out."
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Total Students",
            f"{total_students:,}",
            "Records in dataset"
        )

    with c2:
        kpi_card(
            "Dropout Students",
            f"{dropout_students:,}",
            "Target = 1"
        )

    with c3:
        kpi_card(
            "Not Dropout",
            f"{non_dropout_students:,}",
            "Target = 0"
        )

    with c4:
        kpi_card(
            "Dropout Rate",
            f"{dropout_rate:.1f}%",
            "Dataset proportion"
        )

    st.write("")

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    section(
        "Project Overview",
        "A quick summary of the machine learning project."
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🎯 Machine Learning Task
                </div>

                <div class="info-text">
                    Binary classification used to predict
                    whether a student belongs to the dropout
                    class or the non-dropout class.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🤖 Algorithm
                </div>

                <div class="info-text">
                    Logistic Regression is used as the
                    classification algorithm.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-title">
                    📊 Input Features
                </div>

                <div class="info-text">
                    The model uses
                    <b>{feature_count}</b>
                    input features from the dataset.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with d:
        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🔀 Train / Test
                </div>

                <div class="info-text">
                    80% of the data is used for training
                    and 20% for testing.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    section(
        "Dataset Preview",
        "First records from the loaded dataset."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Preview",
            "🔤 Data Types",
            "🎯 Target Distribution"
        ]
    )

    with tab1:

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

    with tab2:

        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
                "Missing Values": [
                    int(df[column].isna().sum())
                    for column in df.columns
                ],
                "Unique Values": [
                    int(df[column].nunique())
                    for column in df.columns
                ]
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True
        )

    with tab3:

        target_df = pd.DataFrame(
            {
                "Status": [
                    "Not Dropout",
                    "Dropout"
                ],
                "Students": [
                    non_dropout_students,
                    dropout_students
                ]
            }
        )

        target_df["Percentage"] = (
            target_df["Students"]
            / total_students
            * 100
        )

        target_df["Label"] = target_df.apply(
            lambda row:
            f"{int(row['Students']):,} "
            f"({row['Percentage']:.1f}%)",
            axis=1
        )

        fig = px.bar(
            target_df,
            x="Status",
            y="Students",
            text="Label"
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                color="#111827",
                size=14
            ),
            marker=dict(
                color="#374151"
            )
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="Number of Students"
        )

        fig = make_plotly_layout(
            fig,
            430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 2 — EDA
# ============================================================

elif page == "📊 Exploratory Analysis":

    hero(
        "📊 Exploratory Data Analysis",
        "Explore important relationships between student characteristics and dropout status."
    )

    plot_df = df.copy()

    plot_df["Status"] = plot_df[
        TARGET_COLUMN
    ].map(
        {
            0: "Not Dropout",
            1: "Dropout"
        }
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    section(
        "Student Distribution",
        "Distribution of the two target classes."
    )

    counts = (
        plot_df["Status"]
        .value_counts()
        .reindex(
            [
                "Not Dropout",
                "Dropout"
            ]
        )
        .reset_index()
    )

    counts.columns = [
        "Status",
        "Count"
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
        text="Label"
    )

    fig.update_traces(
        marker=dict(
            color="#374151"
        ),
        textposition="outside",
        textfont=dict(
            color="#111827",
            size=14
        )
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Student Status",
        yaxis_title="Number of Students"
    )

    fig = make_plotly_layout(
        fig,
        430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TUITION FEES
    # --------------------------------------------------------

    if "Tuition fees up to date" in df.columns:

        section(
            "Financial Relationship",
            "Dropout rate according to tuition fee status."
        )

        rate = (
            plot_df
            .groupby(
                "Tuition fees up to date"
            )[TARGET_COLUMN]
            .mean()
            .reset_index()
        )

        rate["Tuition Status"] = (
            rate[
                "Tuition fees up to date"
            ]
            .map(
                {
                    0: "Not Up to Date",
                    1: "Up to Date"
                }
            )
        )

        rate["Dropout Rate (%)"] = (
            rate[TARGET_COLUMN]
            * 100
        )

        rate["Label"] = rate[
            "Dropout Rate (%)"
        ].apply(
            lambda value:
            f"{value:.1f}%"
        )

        fig = px.bar(
            rate,
            x="Tuition Status",
            y="Dropout Rate (%)",
            text="Label"
        )

        fig.update_traces(
            marker=dict(
                color="#374151"
            ),
            textposition="outside",
            textfont=dict(
                color="#111827",
                size=14
            )
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Tuition Fee Status",
            yaxis_title="Dropout Rate (%)"
        )

        fig = make_plotly_layout(
            fig,
            430
        )

        fig.update_yaxes(
            range=[0, 100],
            ticksuffix="%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SEMESTER GRADES
    # --------------------------------------------------------

    grade_columns = [
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)"
    ]

    available_grade_columns = [
        column
        for column in grade_columns
        if column in df.columns
    ]

    if available_grade_columns:

        section(
            "Academic Performance",
            "Comparison of semester grades between student groups."
        )

        for grade_column in available_grade_columns:

            fig = px.box(
                plot_df,
                x="Status",
                y=grade_column,
                points="outliers"
            )

            fig.update_traces(
                marker=dict(
                    color="#374151",
                    opacity=0.55
                ),
                line=dict(
                    color="#111827",
                    width=2
                )
            )

            fig.update_layout(
                showlegend=False,
                xaxis_title="Student Status",
                yaxis_title=grade_column
            )

            fig = make_plotly_layout(
                fig,
                450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    section(
        "Feature Correlation",
        "Correlation between numerical features and the target."
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        correlation = numeric_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto"
        )

        fig.update_traces(
            textfont=dict(
                color="#111827",
                size=10
            )
        )

        fig.update_layout(
            height=650,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#111827"
            ),
            margin=dict(
                l=40,
                r=40,
                t=50,
                b=40
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # KEY OBSERVATIONS
    # --------------------------------------------------------

    section(
        "Key Observations",
        "Important patterns visible in the dataset."
    )

    observation_col1, observation_col2 = st.columns(2)

    with observation_col1:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🎯 Target Distribution
                </div>

                <div class="info-text">
                    The dataset contains both Dropout and
                    Not Dropout students. The model therefore
                    performs a binary classification task.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with observation_col2:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    📚 Academic Performance
                </div>

                <div class="info-text">
                    Semester grades can provide useful
                    information for distinguishing students
                    between the two target classes.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CONSTANT COLUMNS
    # --------------------------------------------------------

    with st.expander(
        "🔎 Check Constant / Zero-Variance Columns"
    ):

        constant_columns = [
            column
            for column in df.columns
            if df[column].nunique() <= 1
        ]

        if constant_columns:

            st.warning(
                f"{len(constant_columns)} constant "
                "column(s) found."
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": constant_columns
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No constant columns were found."
            )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    hero(
        "🤖 Model Performance",
        "Evaluation of Logistic Regression on the held-out test dataset."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card(
            "Accuracy",
            f"{results['accuracy'] * 100:.2f}%",
            "Overall correctness"
        )

    with c2:
        kpi_card(
            "Precision",
            f"{results['precision'] * 100:.2f}%",
            "Positive prediction quality"
        )

    with c3:
        kpi_card(
            "Recall",
            f"{results['recall'] * 100:.2f}%",
            "Dropout detection"
        )

    with c4:
        kpi_card(
            "F1 Score",
            f"{results['f1'] * 100:.2f}%",
            "Precision + Recall"
        )

    with c5:
        kpi_card(
            "ROC-AUC",
            f"{results['roc_auc']:.3f}",
            "Ranking performance"
        )

    st.write("")

    # --------------------------------------------------------
    # CONFUSION MATRIX + ROC
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        section(
            "Confusion Matrix",
            "Actual versus predicted student classes."
        )

        cm = results["cm"]

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=[
                "Not Dropout",
                "Dropout"
            ],
            y=[
                "Not Dropout",
                "Dropout"
            ],
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Students"
            }
        )

        fig_cm.update_traces(
            textfont=dict(
                color="#111827",
                size=20
            )
        )

        fig_cm.update_layout(
            height=450,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#111827"
            ),
            margin=dict(
                l=50,
                r=30,
                t=50,
                b=50
            )
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

    with right:

        section(
            "ROC Curve",
            "True Positive Rate versus False Positive Rate."
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=results["fpr"],
                y=results["tpr"],
                mode="lines",
                name="Logistic Regression",
                line=dict(
                    color="#111827",
                    width=3
                )
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(
                    color="#9ca3af",
                    dash="dash"
                )
            )
        )

        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#111827"
            ),
            margin=dict(
                l=55,
                r=30,
                t=50,
                b=55
            )
        )

        fig_roc.update_xaxes(
            range=[0, 1],
            tickfont=dict(
                color="#111827"
            ),
            title_font=dict(
                color="#111827"
            ),
            gridcolor="#e5e7eb"
        )

        fig_roc.update_yaxes(
            range=[0, 1],
            tickfont=dict(
                color="#111827"
            ),
            title_font=dict(
                color="#111827"
            ),
            gridcolor="#e5e7eb"
        )

        st.plotly_chart(
            fig_roc,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CONFUSION DETAILS
    # --------------------------------------------------------

    section(
        "Prediction Details",
        "Breakdown of correct and incorrect predictions."
    )

    tn, fp, fn, tp = (
        results["cm"].ravel()
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        kpi_card(
            "True Negative",
            f"{int(tn):,}",
            "Correct Not Dropout"
        )

    with d2:
        kpi_card(
            "False Positive",
            f"{int(fp):,}",
            "Predicted Dropout incorrectly"
        )

    with d3:
        kpi_card(
            "False Negative",
            f"{int(fn):,}",
            "Missed Dropout"
        )

    with d4:
        kpi_card(
            "True Positive",
            f"{int(tp):,}",
            "Correct Dropout"
        )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    section(
        "Classification Report",
        "Precision, recall and F1-score for each class."
    )

    report_df = (
        pd.DataFrame(
            results["report"]
        )
        .transpose()
        .round(3)
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    section(
        "Feature Impact",
        "Logistic Regression coefficients showing which features have stronger model influence."
    )

    importance_df = (
        results["feature_importance"]
        .head(15)
        .sort_values(
            "Absolute Impact",
            ascending=True
        )
    )

    fig_importance = px.bar(
        importance_df,
        x="Absolute Impact",
        y="Feature",
        orientation="h"
    )

    fig_importance.update_traces(
        marker=dict(
            color="#374151"
        ),
        textfont=dict(
            color="#111827"
        )
    )

    fig_importance.update_layout(
        showlegend=False,
        xaxis_title="Absolute Coefficient",
        yaxis_title="Feature"
    )

    fig_importance = make_plotly_layout(
        fig_importance,
        600
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.info(
        "Feature impact is based on Logistic Regression "
        "coefficients. A larger absolute coefficient means "
        "the feature has a stronger influence on the model "
        "after standardization."
    )


# ============================================================
# PAGE 4 — PREDICT RISK
# ============================================================

elif page == "🔮 Predict Risk":

    hero(
        "🔮 Student Risk Prediction",
        "Enter student information and estimate the probability of dropout."
    )

    feature_names = results[
        "feature_names"
    ]

    X_train = results[
        "X_train"
    ]

    X_test = results[
        "X_test"
    ]

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "sample_index" not in st.session_state:

        st.session_state.sample_index = None


    # --------------------------------------------------------
    # QUICK TEST
    # --------------------------------------------------------

    section(
        "Quick Test",
        "Load a real student from the test dataset."
    )

    quick1, quick2 = st.columns(
        [1, 3]
    )

    with quick1:

        if st.button(
            "🎲 Load Random Student",
            use_container_width=True
        ):

            st.session_state.sample_index = (
                np.random.choice(
                    X_test.index
                )
            )

            st.rerun()

    with quick2:

        if st.session_state.sample_index is not None:

            st.success(
                f"Test student loaded — "
                f"record index: "
                f"{st.session_state.sample_index}"
            )

        else:

            st.info(
                "You can manually enter student information "
                "or load a real test record."
            )


    sample_index = (
        st.session_state.sample_index
    )


    if sample_index is not None:

        defaults = X_test.loc[
            sample_index
        ]

    else:

        defaults = None


    # --------------------------------------------------------
    # COLUMN GROUPS
    # --------------------------------------------------------

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
        column
        for column in demographic_columns
        if column in feature_names
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
        column
        for column in enrollment_columns
        if column in feature_names
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
        column
        for column in academic_columns
        if column in feature_names
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
        column
        for column in financial_columns
        if column in feature_names
    ]


    grouped_columns = (
        demographic_columns
        + enrollment_columns
        + academic_columns
        + financial_columns
    )


    other_columns = [
        column
        for column in feature_names
        if column not in grouped_columns
    ]


    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    inputs = {}

    with st.form(
        "student_prediction_form"
    ):

        # ====================================================
        # RENDER INPUT GROUP
        # ====================================================

        def render_group(
            title,
            columns
        ):

            if not columns:
                return

            st.markdown(
                f"### {title}"
            )

            input_columns = st.columns(3)

            for index, column in enumerate(
                columns
            ):

                series = X_train[
                    column
                ]

                unique_values = (
                    series
                    .dropna()
                    .unique()
                    .tolist()
                )

                unique_values = sorted(
                    unique_values
                )

                if defaults is not None:

                    default_value = defaults[
                        column
                    ]

                else:

                    default_value = (
                        series.median()
                    )

                with input_columns[
                    index % 3
                ]:

                    # ------------------------------------------------
                    # CATEGORICAL / LOW UNIQUE VALUES
                    # ------------------------------------------------

                    if len(unique_values) <= 10:

                        if default_value in unique_values:

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


                        if column in binary_columns:

                            def binary_label(
                                value
                            ):

                                if value == 1:
                                    return "Yes (1)"

                                return "No (0)"


                            value = st.selectbox(
                                column,
                                options=unique_values,
                                index=selected_index,
                                format_func=binary_label,
                                key=f"predict_{column}",
                                help="0 = No, 1 = Yes"
                            )

                        else:

                            value = st.selectbox(
                                column,
                                options=unique_values,
                                index=selected_index,
                                key=f"predict_{column}",
                                help="Value/code from dataset."
                            )

                    # ------------------------------------------------
                    # NUMERIC VALUES
                    # ------------------------------------------------

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

                        default_number = max(
                            min_value,
                            min(
                                default_number,
                                max_value
                            )
                        )

                        # Integer-looking feature
                        if (
                            pd.api.types.is_integer_dtype(
                                series
                            )
                        ):

                            value = st.number_input(
                                column,
                                min_value=int(
                                    min_value
                                ),
                                max_value=int(
                                    max_value
                                ),
                                value=int(
                                    default_number
                                ),
                                step=1,
                                key=f"predict_{column}"
                            )

                        else:

                            value = st.number_input(
                                column,
                                min_value=min_value,
                                max_value=max_value,
                                value=default_number,
                                key=f"predict_{column}"
                            )


                    inputs[
                        column
                    ] = value


        # ====================================================
        # GROUPS
        # ====================================================

        render_group(
            "👤 Demographic Information",
            demographic_columns
        )

        st.divider()

        render_group(
            "📝 Enrollment Information",
            enrollment_columns
        )

        st.divider()

        render_group(
            "📚 Academic Information",
            academic_columns
        )

        st.divider()

        render_group(
            "💰 Financial & Socioeconomic Information",
            financial_columns
        )

        if other_columns:

            st.divider()

            render_group(
                "➕ Additional Information",
                other_columns
            )

        st.write("")

        submitted = st.form_submit_button(
            "🔍 Predict Student Risk",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        input_df = pd.DataFrame(
            [inputs]
        )

        input_df = input_df[
            feature_names
        ]

        # Scale
        input_scaled = (
            results["scaler"]
            .transform(
                input_df
            )
        )

        # Probability
        probability = (
            results["model"]
            .predict_proba(
                input_scaled
            )[0, 1]
        )

        predicted_class = (
            results["model"]
            .predict(
                input_scaled
            )[0]
        )


        # ====================================================
        # RISK CATEGORY
        # ====================================================

        if probability < 0.30:

            risk_label = "Low Risk"

            risk_class = "risk-low"

            risk_color = "#15803d"

            risk_description = (
                "The model estimates a relatively "
                "low probability of dropout."
            )

        elif probability < 0.60:

            risk_label = "Medium Risk"

            risk_class = "risk-medium"

            risk_color = "#b45309"

            risk_description = (
                "The model estimates a moderate "
                "probability of dropout."
            )

        else:

            risk_label = "High Risk"

            risk_class = "risk-high"

            risk_color = "#b91c1c"

            risk_description = (
                "The model estimates a relatively "
                "high probability of dropout."
            )


        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.write("")

        section(
            "Prediction Result",
            "Model-generated dropout probability and risk category."
        )


        result_left, result_right = st.columns(
            [1.25, 0.75]
        )


        # ====================================================
        # GAUGE
        # ====================================================

        with result_left:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",

                    value=probability * 100,

                    number={
                        "suffix": "%",
                        "font": {
                            "size": 42,
                            "color": "#111827"
                        }
                    },

                    title={
                        "text": "Dropout Probability",
                        "font": {
                            "size": 19,
                            "color": "#111827"
                        }
                    },

                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100
                            ],
                            "tickwidth": 1,
                            "tickcolor": "#111827",
                            "tickfont": {
                                "color": "#111827"
                            }
                        },

                        "bar": {
                            "color": risk_color
                        },

                        "borderwidth": 1,

                        "bordercolor": "#9ca3af",

                        "steps": [
                            {
                                "range": [
                                    0,
                                    30
                                ],
                                "color": "#ecfdf5"
                            },
                            {
                                "range": [
                                    30,
                                    60
                                ],
                                "color": "#fffbeb"
                            },
                            {
                                "range": [
                                    60,
                                    100
                                ],
                                "color": "#fef2f2"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=380,
                paper_bgcolor="white",
                font=dict(
                    color="#111827"
                ),
                margin=dict(
                    l=25,
                    r=25,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )


        # ====================================================
        # RESULT CARD
        # ====================================================

        with result_right:

            st.markdown(
                f"""
                <div class="{risk_class}">

                    <div class="risk-title"
                         style="color:{risk_color};">

                        {risk_label}

                    </div>

                    <div style="
                        color:#374151;
                        font-size:0.92rem;
                        line-height:1.5;
                    ">

                        {risk_description}

                    </div>

                    <div class="risk-probability"
                         style="color:{risk_color};">

                        {probability * 100:.2f}%

                    </div>

                    <div style="
                        color:#6b7280;
                        font-size:0.82rem;
                    ">

                        Estimated dropout probability

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            predicted_label = (
                "Dropout"
                if predicted_class == 1
                else "Not Dropout"
            )

            if predicted_class == 1:

                st.markdown(
                    f"""
                    <div class="prediction-result">

                        <div style="
                            color:#6b7280;
                            font-weight:700;
                        ">
                            Predicted Class
                        </div>

                        <div style="
                            color:#b91c1c;
                            font-size:1.7rem;
                            font-weight:900;
                            margin-top:5px;
                        ">
                            🔴 {predicted_label}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="prediction-result">

                        <div style="
                            color:#6b7280;
                            font-weight:700;
                        ">
                            Predicted Class
                        </div>

                        <div style="
                            color:#15803d;
                            font-size:1.7rem;
                            font-weight:900;
                            margin-top:5px;
                        ">
                            🟢 {predicted_label}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # RISK SCALE
        # ====================================================

        st.write("")

        section(
            "Risk Scale",
            "Project-level interpretation of model probability."
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.markdown(
                """
                <div class="risk-low">

                    <div class="risk-title"
                         style="color:#15803d;">

                        🟢 Low Risk

                    </div>

                    <div style="color:#166534;">

                        Probability below 30%.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:

            st.markdown(
                """
                <div class="risk-medium">

                    <div class="risk-title"
                         style="color:#b45309;">

                        🟡 Medium Risk

                    </div>

                    <div style="color:#92400e;">

                        Probability from 30% to below 60%.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with r3:

            st.markdown(
                """
                <div class="risk-high">

                    <div class="risk-title"
                         style="color:#b91c1c;">

                        🔴 High Risk

                    </div>

                    <div style="color:#991b1b;">

                        Probability of 60% or higher.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "These risk bands are project-level thresholds "
            "for interpreting model probability and are not "
            "official institutional policies."
        )


        # ====================================================
        # ACTUAL OUTCOME
        # ====================================================

        if sample_index is not None:

            original_values = (
                X_test
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

                actual_outcome = int(
                    df.loc[
                        sample_index,
                        TARGET_COLUMN
                    ]
                )

                actual_label = (
                    "Dropout"
                    if actual_outcome == 1
                    else "Not Dropout"
                )

                st.write("")

                section(
                    "Actual Test Student Outcome",
                    "Comparison between the model prediction and the known test-set outcome."
                )

                actual_col1, actual_col2 = st.columns(2)

                with actual_col1:

                    if actual_outcome == 1:

                        st.markdown(
                            """
                            <div class="risk-high">

                                <div class="risk-title"
                                     style="color:#b91c1c;">

                                    🔴 Actual Outcome

                                </div>

                                <div style="
                                    color:#991b1b;
                                    font-size:1.5rem;
                                    font-weight:900;
                                ">

                                    Dropout

                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="risk-low">

                                <div class="risk-title"
                                     style="color:#15803d;">

                                    🟢 Actual Outcome

                                </div>

                                <div style="
                                    color:#166534;
                                    font-size:1.5rem;
                                    font-weight:900;
                                ">

                                    Not Dropout

                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with actual_col2:

                    if predicted_class == actual_outcome:

                        st.success(
                            "✅ Model prediction matches "
                            "the actual test-set outcome."
                        )

                    else:

                        st.warning(
                            "⚠️ Model prediction does not "
                            "match the actual test-set outcome."
                        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>

    <div style="
        text-align:center;
        color:#9ca3af;
        font-size:0.8rem;
        padding:20px 0;
        border-top:1px solid #e5e7eb;
    ">

        🎓 Student Dropout Prediction |
        Logistic Regression |
        Machine Learning Project

    </div>
    """,
    unsafe_allow_html=True
)
