"""
Student Dropout Prediction — Streamlit App

Features:
- Dataset Overview
- Exploratory Data Analysis
- Logistic Regression Model
- Model Evaluation
- Student Dropout Risk Prediction

Dataset:
datasett.csv

Run:
streamlit run app.py
"""

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

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    .metric-title {
        font-size: 0.9rem;
        color: #6b7280;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .risk-badge {
        display: inline-block;
        padding: 12px 25px;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .risk-low {
        background: #dcfce7;
        color: #15803d;
    }

    .risk-medium {
        background: #fef9c3;
        color: #a16207;
    }

    .risk-high {
        background: #fee2e2;
        color: #b91c1c;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET
# ============================================================

DATASET_FILE = "datasett.csv"


@st.cache_data
def load_data():

    df = pd.read_csv(DATASET_FILE)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# CLEAN DATA
# ============================================================

@st.cache_data
def clean_data(df):

    df = df.copy()

    if "target" not in df.columns:

        raise ValueError(
            "The dataset does not contain a 'target' column."
        )

    # Convert target safely
    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
    )

    # Original UCI target:
    # Dropout = 1
    # Graduate = 0
    # Enrolled = 0

    df["target"] = df["target"].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0
        }
    )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="raise"
    ).astype(int)

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    feature_names = X.columns.tolist()

    # 80 / 20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # Evaluation
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

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,

        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,

        "y_pred": y_pred,
        "y_prob": y_prob,

        "cm": cm,
        "report": report,

        "accuracy": accuracy,
        "roc_auc": roc_auc,

        "fpr": fpr,
        "tpr": tpr
    }


# ============================================================
# LOAD DATASET
# ============================================================

try:

    raw_df = load_data()

    df = clean_data(
        raw_df
    )

except FileNotFoundError:

    st.error(
        "datasett.csv was not found."
    )

    st.info(
        "Make sure datasett.csv is uploaded to the "
        "same GitHub repository as app.py."
    )

    st.stop()

except Exception as error:

    st.error(
        "There was an error loading the dataset."
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
        "Model training failed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🎓 Student Dropout Predictor"
)

st.sidebar.caption(
    "Logistic Regression • Machine Learning Project"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Student Analysis",
        "🤖 Model Performance",
        "🔮 Predict Student Risk"
    ]
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

st.sidebar.caption(
    "Dataset: UCI Student Dropout Dataset"
)

st.sidebar.caption(
    "Target: Dropout / Not Dropout"
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🎓 Student Dropout Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning system for identifying students '
        'who may be at risk of dropping out.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    total_students = len(df)

    dropout_students = int(
        (df["target"] == 1).sum()
    )

    non_dropout_students = int(
        (df["target"] == 0).sum()
    )

    dropout_rate = (
        dropout_students /
        total_students *
        100
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
                {dropout_rate:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    st.subheader(
        "Student Outcome Distribution"
    )

    distribution = pd.DataFrame(
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

    fig = px.bar(
        distribution,
        x="Status",
        y="Students",
        color="Status",
        text="Students",
        color_discrete_map={
            "Not Dropout": "#3b82f6",
            "Dropout": "#ef4444"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Number of Students",
        xaxis_title="",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader(
        "Dataset Information"
    )

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Features",
        df.shape[1] - 1
    )

    info2.metric(
        "Missing Values",
        int(
            df.isnull().sum().sum()
        )
    )

    info3.metric(
        "Duplicate Rows",
        int(
            df.duplicated().sum()
        )
    )


# ============================================================
# PAGE 2 — STUDENT ANALYSIS
# ============================================================

elif page == "📊 Student Analysis":

    st.markdown(
        '<div class="main-title">'
        '📊 Student Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Exploratory analysis of academic and socioeconomic '
        'factors related to student dropout.'
        '</div>',
        unsafe_allow_html=True
    )

    plot_df = df.copy()

    plot_df["Status"] = plot_df[
        "target"
    ].map(
        {
            0: "Not Dropout",
            1: "Dropout"
        }
    )

    # ========================================================
    # GRAPH 1
    # ========================================================

    st.subheader(
        "1. Dropout vs Not Dropout"
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
        "Students"
    ]

    fig1 = px.bar(
        counts,
        x="Status",
        y="Students",
        color="Status",
        text="Students",
        color_discrete_map={
            "Not Dropout": "#3b82f6",
            "Dropout": "#ef4444"
        }
    )

    fig1.update_traces(
        textposition="outside"
    )

    fig1.update_layout(
        showlegend=False,
        yaxis_title="Number of Students",
        xaxis_title="",
        height=420
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.info(
        "Most students in the dataset are classified as "
        "Not Dropout, while a smaller proportion are classified "
        "as Dropout."
    )

    # ========================================================
    # GRAPH 2 — TUITION
    # ========================================================

    if "Tuition fees up to date" in df.columns:

        st.subheader(
            "2. Tuition Fees Status vs Dropout Rate"
        )

        fee_data = (
            plot_df
            .groupby(
                "Tuition fees up to date"
            )["target"]
            .mean()
            .reset_index()
        )

        fee_data["Status"] = fee_data[
            "Tuition fees up to date"
        ].map(
            {
                0: "Not Up to Date",
                1: "Up to Date"
            }
        )

        fee_data["Dropout Rate"] = (
            fee_data["target"] * 100
        )

        fig2 = px.bar(
            fee_data,
            x="Status",
            y="Dropout Rate",
            color="Status",
            text="Dropout Rate",
            color_discrete_map={
                "Not Up to Date": "#f59e0b",
                "Up to Date": "#10b981"
            }
        )

        fig2.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig2.update_layout(
            showlegend=False,
            xaxis_title="Tuition Fee Status",
            yaxis_title="Dropout Rate (%)",
            yaxis=dict(
                range=[0, 100],
                dtick=20,
                ticksuffix="%"
            ),
            height=450
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.info(
            "Students whose tuition fees were not up to date "
            "showed a much higher dropout rate in this dataset."
        )

    # ========================================================
    # GRAPH 3 — 1ST SEMESTER
    # ========================================================

    if (
        "Curricular units 1st sem (grade)"
        in df.columns
    ):

        st.subheader(
            "3. 1st Semester Grade vs Dropout"
        )

        fig3 = px.box(
            plot_df,
            x="Status",
            y="Curricular units 1st sem (grade)",
            color="Status",
            color_discrete_map={
                "Not Dropout": "#3b82f6",
                "Dropout": "#ef4444"
            },
            points="outliers"
        )

        fig3.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="1st Semester Grade",
            height=450
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.info(
            "Dropout students generally show lower "
            "1st semester grades than non-dropout students."
        )

    # ========================================================
    # GRAPH 4 — 2ND SEMESTER
    # ========================================================

    if (
        "Curricular units 2nd sem (grade)"
        in df.columns
    ):

        st.subheader(
            "4. 2nd Semester Grade vs Dropout"
        )

        fig4 = px.box(
            plot_df,
            x="Status",
            y="Curricular units 2nd sem (grade)",
            color="Status",
            color_discrete_map={
                "Not Dropout": "#3b82f6",
                "Dropout": "#ef4444"
            },
            points="outliers"
        )

        fig4.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="2nd Semester Grade",
            height=450
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        st.info(
            "Lower 2nd semester grades, especially very low "
            "or zero grades, are strongly associated with dropout."
        )

    # ========================================================
    # CONSTANT COLUMNS
    # ========================================================

    with st.expander(
        "🔎 Constant Columns Check"
    ):

        constant_columns = (
            df.nunique()[
                df.nunique() == 1
            ]
        )

        if len(constant_columns) == 0:

            st.success(
                "No constant columns were found."
            )

        else:

            st.dataframe(
                constant_columns.rename(
                    "Unique Values"
                )
            )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="main-title">'
        '🤖 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation of the Logistic Regression classification model.'
        '</div>',
        unsafe_allow_html=True
    )

    report = results["report"]

    # --------------------------------------------------------
    # Dropout metrics
    # --------------------------------------------------------

    accuracy = results["accuracy"]

    precision = report["1"]["precision"]

    recall = report["1"]["recall"]

    f1 = report["1"]["f1-score"]

    roc_auc = results["roc_auc"]

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    m2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    m3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    m4.metric(
        "F1-Score",
        f"{f1 * 100:.2f}%"
    )

    m5.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
    )

    st.write("")

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Confusion Matrix"
        )

        cm = results["cm"]

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
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

        fig_cm.update_layout(
            height=450
        )

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
                    color="gray",
                    dash="dash"
                )
            )
        )

        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450
        )

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
            report
        )
        .transpose()
        .round(3)
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    st.subheader(
        "Prediction Error Analysis"
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
        f"False Positives: {fp} students were predicted as "
        "Dropout but were actually Not Dropout.\n\n"
        f"False Negatives: {fn} actual Dropout students were "
        "predicted as Not Dropout."
    )


# ============================================================
# PAGE 4 — PHASE 7
# ============================================================

elif page == "🔮 Predict Student Risk":

    st.markdown(
        '<div class="main-title">'
        '🔮 Student Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter student information to predict dropout probability '
        'and risk level.'
        '</div>',
        unsafe_allow_html=True
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

    # ========================================================
    # QUICK TEST CASE
    # ========================================================

    st.subheader(
        "Quick Test"
    )

    st.write(
        "You can load a real student record from the test set "
        "and then modify the values."
    )

    if "sample_index" not in st.session_state:

        st.session_state.sample_index = None

    if st.button(
        "🎲 Load Random Test Student"
    ):

        st.session_state.sample_index = (
            np.random.choice(
                X_test.index
            )
        )

    sample_index = (
        st.session_state.sample_index
    )

    if sample_index is not None:

        st.success(
            f"Test student loaded successfully. "
            f"Record index: {sample_index}"
        )

    # ========================================================
    # FORM
    # ========================================================

    with st.form(
        "student_prediction_form"
    ):

        # ----------------------------------------------------
        # DEMOGRAPHIC
        # ----------------------------------------------------

        st.markdown(
            "### 👤 Demographic Information"
        )

        demographic_columns = [
            "Marital Status",
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance",
            "Previous qualification",
            "Previous qualification (grade)",
            "Nacionality",
            "Gender",
            "Age at enrollment",
            "International"
        ]

        demographic_columns = [
            col for col in demographic_columns
            if col in feature_names
        ]

        inputs = {}

        demo_cols = st.columns(3)

        for i, col_name in enumerate(
            demographic_columns
        ):

            series = X_train[
                col_name
            ]

            unique_values = sorted(
                series.dropna()
                .unique()
                .tolist()
            )

            if sample_index is not None:

                default_value = X_test.loc[
                    sample_index,
                    col_name
                ]

            else:

                default_value = (
                    series.median()
                )

            with demo_cols[
                i % 3
            ]:

                if len(unique_values) <= 10:

                    if default_value in unique_values:

                        selected_index = (
                            unique_values.index(
                                default_value
                            )
                        )

                    else:

                        selected_index = 0

                    inputs[col_name] = st.selectbox(
                        col_name,
                        unique_values,
                        index=selected_index
                    )

                else:

                    inputs[col_name] = st.number_input(
                        col_name,
                        min_value=float(
                            series.min()
                        ),
                        max_value=float(
                            series.max()
                        ),
                        value=float(
                            default_value
                        )
                    )

        # ----------------------------------------------------
        # ACADEMIC
        # ----------------------------------------------------

        st.markdown(
            "### 📚 Academic Information"
        )

        academic_keywords = [
            "grade",
            "evaluations",
            "approved",
            "credited",
            "enrolled",
            "without evaluations"
        ]

        academic_columns = [
            col for col in feature_names
            if any(
                keyword in col.lower()
                for keyword in academic_keywords
            )
            and col not in demographic_columns
        ]

        academic_cols = st.columns(3)

        for i, col_name in enumerate(
            academic_columns
        ):

            series = X_train[
                col_name
            ]

            unique_values = sorted(
                series.dropna()
                .unique()
                .tolist()
            )

            if sample_index is not None:

                default_value = X_test.loc[
                    sample_index,
                    col_name
                ]

            else:

                default_value = (
                    series.median()
                )

            with academic_cols[
                i % 3
            ]:

                if len(unique_values) <= 10:

                    if default_value in unique_values:

                        selected_index = (
                            unique_values.index(
                                default_value
                            )
                        )

                    else:

                        selected_index = 0

                    inputs[col_name] = st.selectbox(
                        col_name,
                        unique_values,
                        index=selected_index
                    )

                else:

                    inputs[col_name] = st.number_input(
                        col_name,
                        min_value=float(
                            series.min()
                        ),
                        max_value=float(
                            series.max()
                        ),
                        value=float(
                            default_value
                        )
                    )

        # ----------------------------------------------------
        # FINANCIAL / SOCIOECONOMIC
        # ----------------------------------------------------

        st.markdown(
            "### 💰 Financial & Socioeconomic Information"
        )

        financial_columns = [
            col for col in feature_names
            if col not in demographic_columns
            and col not in academic_columns
        ]

        financial_cols = st.columns(3)

        for i, col_name in enumerate(
            financial_columns
        ):

            series = X_train[
                col_name
            ]

            unique_values = sorted(
                series.dropna()
                .unique()
                .tolist()
            )

            if sample_index is not None:

                default_value = X_test.loc[
                    sample_index,
                    col_name
                ]

            else:

                default_value = (
                    series.median()
                )

            with financial_cols[
                i % 3
            ]:

                if len(unique_values) <= 10:

                    if default_value in unique_values:

                        selected_index = (
                            unique_values.index(
                                default_value
                            )
                        )

                    else:

                        selected_index = 0

                    inputs[col_name] = st.selectbox(
                        col_name,
                        unique_values,
                        index=selected_index
                    )

                else:

                    inputs[col_name] = st.number_input(
                        col_name,
                        min_value=float(
                            series.min()
                        ),
                        max_value=float(
                            series.max()
                        ),
                        value=float(
                            default_value
                        )
                    )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        st.write("")

        submitted = st.form_submit_button(
            "🔍 Predict Dropout Risk",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    if submitted:

        input_df = pd.DataFrame(
            [inputs]
        )

        # Make sure exact training order is used
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

        # Prediction
        prediction = (
            results["model"]
            .predict(
                input_scaled
            )[0]
        )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        if probability < 0.30:

            risk = "Low Risk"
            risk_class = "risk-low"

        elif probability < 0.60:

            risk = "Medium Risk"
            risk_class = "risk-medium"

        else:

            risk = "High Risk"
            risk_class = "risk-high"

        # ====================================================
        # RESULT
        # ====================================================

        st.write("---")

        st.markdown(
            "## 🎯 Prediction Result"
        )

        result1, result2 = st.columns(2)

        # ----------------------------------------------------
        # GAUGE
        # ----------------------------------------------------

        with result1:

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
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=330
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        # ----------------------------------------------------
        # RESULT DETAILS
        # ----------------------------------------------------

        with result2:

            st.markdown(
                f"""
                <div class="risk-badge {risk_class}">
                    {risk}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.metric(
                "Predicted Class",
                "Dropout"
                if prediction == 1
                else "Not Dropout"
            )

            st.metric(
                "Dropout Probability",
                f"{probability * 100:.2f}%"
            )

            if prediction == 1:

                st.warning(
                    "The model predicts that this student "
                    "has a higher likelihood of dropping out."
                )

            else:

                st.success(
                    "The model predicts that this student "
                    "has a lower likelihood of dropping out."
                )

        # ----------------------------------------------------
        # RISK INFORMATION
        # ----------------------------------------------------

        st.info(
            "Risk interpretation: "
            "**Low Risk < 30%**, "
            "**Medium Risk = 30%–59.99%**, "
            "**High Risk ≥ 60%**."
        )
