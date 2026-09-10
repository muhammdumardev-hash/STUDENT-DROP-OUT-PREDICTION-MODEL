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

    /* ------------------------------------------------------
       Main title
    ------------------------------------------------------ */

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }


    /* ------------------------------------------------------
       Metric cards
    ------------------------------------------------------ */

    .metric-card {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        min-height: 105px;
    }

    .metric-title {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 5px;
    }


    /* ------------------------------------------------------
       Risk badges
    ------------------------------------------------------ */

    .risk-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }

    .risk-low {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }

    .risk-medium {
        background: #fef9c3;
        color: #a16207;
        border: 1px solid #fde047;
    }

    .risk-high {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }


    /* ------------------------------------------------------
       Section boxes
    ------------------------------------------------------ */

    .section-box {
        border: 1px solid #d1d5db;
        border-radius: 14px;
        padding: 18px;
        background: #ffffff;
        margin-bottom: 15px;
    }


    /* ------------------------------------------------------
       Sidebar
    ------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #d1d5db;
    }


    /* ------------------------------------------------------
       Dataframes
    ------------------------------------------------------ */

    [data-testid="stDataFrame"] {
        border: 1px solid #d1d5db;
        border-radius: 10px;
        overflow: hidden;
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

    # Clean column names
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

    # --------------------------------------------------------
    # Safely convert target
    # --------------------------------------------------------

    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
    )

    # Original UCI target:
    #
    # Dropout  -> 1
    # Graduate -> 0
    # Enrolled -> 0
    #
    # This makes the problem binary:
    #
    # 1 = Dropout
    # 0 = Not Dropout

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

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    feature_names = X.columns.tolist()


    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # --------------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Dataset Metrics
    # --------------------------------------------------------

    total_students = len(df)

    dropout_students = int(
        (df["target"] == 1).sum()
    )

    non_dropout_students = int(
        (df["target"] == 0).sum()
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
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

    c2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Dropout Students
            </div>
            <div class="metric-value">
                {dropout_students:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Not Dropout
            </div>
            <div class="metric-value">
                {non_dropout_students:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Dropout Rate
            </div>
            <div class="metric-value">
                {dropout_students / total_students * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # --------------------------------------------------------
    # Project Summary
    # --------------------------------------------------------

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
        s3.write(
            f"{df.shape[1] - 1}"
        )

        s4.write("**Train / Test Split**")
        s4.write("80% / 20%")


    st.write("")


    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Dataset Preview",
            "🔤 Data Types",
            "🎯 Target Distribution",
        ]
    )


    # --------------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Data Types
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Target Distribution
    # --------------------------------------------------------

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
            cliponaxis=False
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Students",
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(
                l=50,
                r=30,
                t=50,
                b=50
            ),
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#9ca3af"
        )

        with st.container(border=True):

            st.plotly_chart(
                fig,
                use_container_width=True
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


    # --------------------------------------------------------
    # Dropout Distribution
    # --------------------------------------------------------

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
            cliponaxis=False
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Students",
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            showline=True,
            linecolor="#9ca3af"
        )

        with st.container(border=True):

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # Tuition Fees
    # --------------------------------------------------------

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
                cliponaxis=False
            )

            fig.update_layout(
                showlegend=False,
                xaxis_title="Tuition Fee Status",
                yaxis_title="Dropout Rate (%)",
                height=450,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            # IMPORTANT:
            # Fixed 0–100% scale
            fig.update_yaxes(
                range=[0, 100],
                dtick=20,
                ticksuffix="%",
                showgrid=True,
                gridcolor="#e5e7eb",
                showline=True,
                linecolor="#9ca3af"
            )

            with st.container(border=True):

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                "'Tuition fees up to date' column "
                "was not found."
            )


    # ========================================================
    # ROW 2 — FIRST + SECOND SEMESTER
    # ========================================================

    col3, col4 = st.columns(2)


    # --------------------------------------------------------
    # 1st Semester Grade
    # --------------------------------------------------------

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

            # Make outliers cleaner
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
            )

            # FIXED grade scale
            fig.update_yaxes(
                range=[0, 20],
                dtick=2,
                title="1st Semester Grade",
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=True,
                showline=True,
                linecolor="#9ca3af",
            )

            fig.update_xaxes(
                title="Student Status",
                showline=True,
                linecolor="#9ca3af"
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


    # --------------------------------------------------------
    # 2nd Semester Grade
    # --------------------------------------------------------

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

            # Make outliers cleaner
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
            )

            # FIXED grade scale
            fig.update_yaxes(
                range=[0, 20],
                dtick=2,
                title="2nd Semester Grade",
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=True,
                showline=True,
                linecolor="#9ca3af",
            )

            fig.update_xaxes(
                title="Student Status",
                showline=True,
                linecolor="#9ca3af"
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


    # ========================================================
    # OBSERVATIONS
    # ========================================================

    st.write("")

    with st.container(border=True):

        st.subheader(
            "💡 Key Observations"
        )

        st.markdown(
            """
            **1st Semester**

            - Dropout students generally have lower first-semester grades.
            - Very low or zero grades are strongly associated with dropout.
            - Non-dropout students show a more stable grade distribution.

            **2nd Semester**

            - The difference between dropout and non-dropout students becomes more pronounced.
            - A large concentration of dropout students has very low or zero grades.
            - Academic disengagement is therefore a useful warning indicator.

            **Important:** These observations show **relationships in the dataset**.
            They do not prove that low grades directly cause dropout.
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


    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Accuracy
            </div>
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
            <div class="metric-title">
                Precision
            </div>
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
            <div class="metric-title">
                Recall
            </div>
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
            <div class="metric-title">
                F1-Score
            </div>
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
            <div class="metric-title">
                ROC-AUC
            </div>
            <div class="metric-value">
                {results["roc_auc"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # ========================================================
    # CONFUSION MATRIX + ROC
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

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

        fig_cm.update_layout(
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        with st.container(border=True):

            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )


    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

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
                    color="gray",
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
        )

        fig_roc.update_xaxes(
            range=[0, 1],
            showgrid=True,
            gridcolor="#e5e7eb"
        )

        fig_roc.update_yaxes(
            range=[0, 1],
            showgrid=True,
            gridcolor="#e5e7eb"
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
    # CONFUSION MATRIX DETAILS
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


    feature_names = results[
        "feature_names"
    ]

    X_train_raw = results[
        "X_train"
    ]

    X_test_raw = results[
        "X_test"
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
            "Load a real student record from the test set. "
            "You can then modify the values and test the model."
        )

        if st.button(
            "Load Random Test Student",
            use_container_width=True
        ):

            random_index = np.random.choice(
                X_test_raw.index
            )

            st.session_state.sample_index = (
                random_index
            )

            st.rerun()


    sample_index = (
        st.session_state.sample_index
    )


    if sample_index is not None:

        st.success(
            f"Random test student loaded "
            f"(record index: {sample_index})."
        )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    if sample_index is not None:

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

        # ====================================================
        # DEMOGRAPHIC SECTION
        # ====================================================

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


        # ====================================================
        # ENROLLMENT SECTION
        # ====================================================

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


        # ====================================================
        # ACADEMIC SECTION
        # ====================================================

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


        # ====================================================
        # FINANCIAL SECTION
        # ====================================================

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


        # ====================================================
        # OTHER COLUMNS
        # ====================================================

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

                series = X_train_raw[
                    col_name
                ]

                unique_values = sorted(
                    series
                    .dropna()
                    .unique()
                    .tolist()
                )


                # --------------------------------------------
                # Default value
                # --------------------------------------------

                if defaults is not None:

                    default_value = defaults[
                        col_name
                    ]

                else:

                    default_value = (
                        series.median()
                    )


                with input_cols[
                    i % 3
                ]:

                    # ----------------------------------------
                    # Categorical / Low-cardinality
                    # ----------------------------------------

                    if len(unique_values) <= 10:

                        if default_value in unique_values:

                            selected_index = (
                                unique_values.index(
                                    default_value
                                )
                            )

                        else:

                            selected_index = 0

                        # Binary yes/no helper
                        binary_columns = [
                            "Debtor",
                            "Tuition fees up to date",
                            "Scholarship holder",
                            "Displaced",
                            "Educational special needs",
                            "International",
                        ]

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

                        else:

                            value = st.selectbox(
                                col_name,
                                options=unique_values,
                                index=selected_index,
                                key=f"input_{col_name}",
                                help="Value/code from the dataset.",
                            )

                    # ----------------------------------------
                    # Continuous numeric
                    # ----------------------------------------

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

                        # Protect against possible
                        # floating-point boundary issues

                        if default_number < min_value:
                            default_number = min_value

                        if default_number > max_value:
                            default_number = max_value

                        value = st.number_input(
                            col_name,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_number,
                            key=f"input_{col_name}",
                        )

                    inputs[
                        col_name
                    ] = value


        # ====================================================
        # RENDER SECTIONS
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
        # Create input dataframe
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [inputs]
        )

        # Exact feature order
        input_df = input_df[
            feature_names
        ]


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        input_scaled = (
            results["scaler"]
            .transform(
                input_df
            )
        )


        # ----------------------------------------------------
        # Prediction probability
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
        # RISK CATEGORY
        # ====================================================

        if probability < 0.30:

            risk_label = "Low Risk"
            risk_css = "risk-low"

        elif probability < 0.60:

            risk_label = "Medium Risk"
            risk_css = "risk-medium"

        else:

            risk_label = "High Risk"
            risk_css = "risk-high"


        # ====================================================
        # RESULT
        # ====================================================

        st.write("---")

        st.markdown(
            "## 🎯 Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


        # ====================================================
        # GAUGE
        # ====================================================

        with result_col1:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%"
                    },
                    title={
                        "text":
                        "Dropout Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100
                            ]
                        },

                        "bar": {
                            "color":
                            "#1f2937"
                        },

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
                                "#fef9c3"
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
                    },
                )
            )

            gauge.update_layout(
                height=330,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=10
                ),
                paper_bgcolor="white",
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

                st.metric(
                    "Predicted Class",
                    (
                        "Dropout"
                        if predicted_class == 1
                        else "Not Dropout"
                    )
                )

                st.metric(
                    "Dropout Probability",
                    f"{probability * 100:.2f}%"
                )


                if predicted_class == 1:

                    st.warning(
                        "The model predicts that this "
                        "student has a higher likelihood "
                        "of dropping out."
                    )

                else:

                    st.success(
                        "The model predicts that this "
                        "student has a lower likelihood "
                        "of dropping out."
                    )


        # ====================================================
        # ACTUAL OUTCOME FOR RANDOM TEST CASE
        # ====================================================

        if sample_index is not None:

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

                actual_outcome = int(
                    df.loc[
                        sample_index,
                        "target"
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

                    st.metric(
                        "Actual Student Outcome",
                        actual_label
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

            st.markdown(
                """
                **Low Risk:** Dropout probability below 30%

                **Medium Risk:** Dropout probability from 30% to below 60%

                **High Risk:** Dropout probability of 60% or higher
                """
            )

            st.caption(
                "These risk bands are project-level thresholds "
                "for interpreting model probability and are "
                "not official institutional policies."
            )
