"""
============================================================
STUDENT DROPOUT PREDICTION SYSTEM
============================================================

Machine Learning:
    Logistic Regression

Dataset:
    datasett.csv

Target:
    Dropout      -> 1
    Graduate     -> 0
    Enrolled     -> 0

Run:
    streamlit run app.py

Deployment:
    Streamlit Community Cloud
============================================================
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# ============================================================
# PAGE CONFIGURATION
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
       GLOBAL
    ====================================================== */

    .stApp {
        background-color: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }

    /* ======================================================
       HEADINGS
    ====================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }

    p, li {
        color: #1f2937;
    }

    /* ======================================================
       MAIN HEADER
    ====================================================== */

    .main-header {
        background: linear-gradient(
            135deg,
            #1d4ed8 0%,
            #2563eb 50%,
            #3b82f6 100%
        );

        padding: 30px 35px;
        border-radius: 18px;
        margin-bottom: 25px;

        box-shadow:
            0 10px 25px rgba(37, 99, 235, 0.15);
    }

    .main-header-title {
        color: white !important;
        font-size: 2.35rem;
        font-weight: 800;
        margin: 0;
    }

    .main-header-subtitle {
        color: #dbeafe !important;
        font-size: 1rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* ======================================================
       METRIC CARDS
    ====================================================== */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;

        box-shadow:
            0 4px 12px rgba(15, 23, 42, 0.05);
    }

    .metric-title {
        color: #6b7280 !important;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #111827 !important;
        font-size: 1.85rem;
        font-weight: 800;
    }

    .metric-description {
        color: #6b7280 !important;
        font-size: 0.78rem;
        margin-top: 5px;
    }

    /* ======================================================
       SECTION CARD
    ====================================================== */

    .section-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;

        box-shadow:
            0 4px 12px rgba(15, 23, 42, 0.04);
    }

    /* ======================================================
       RISK BADGES
    ====================================================== */

    .risk-low {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
    }

    .risk-medium {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
    }

    .risk-high {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }

    .risk-box {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 15px;
    }

    .risk-title {
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .risk-value {
        font-size: 2rem;
        font-weight: 900;
    }

    /* ======================================================
       PREDICTION RESULT
    ====================================================== */

    .prediction-result {
        background: white;
        border-radius: 18px;
        padding: 25px;

        border: 1px solid #e5e7eb;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.07);
    }

    .prediction-label {
        color: #6b7280 !important;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 900;
        margin-top: 5px;
    }

    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] * {
        color: #1f2937;
    }

    /* ======================================================
       FORM LABELS
    ====================================================== */

    label {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* ======================================================
       BUTTONS
    ====================================================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    /* ======================================================
       DATAFRAME
    ====================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #d1d5db;
        border-radius: 12px;
    }

    /* ======================================================
       INFO BOX
    ====================================================== */

    .info-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 18px;
        color: #1e3a8a;
    }

    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "datasett.csv"


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

    data = df.copy()

    if "target" not in data.columns:
        raise ValueError(
            "The dataset must contain a column named 'target'."
        )

    data["target"] = (
        data["target"]
        .astype(str)
        .str.strip()
    )

    data["target"] = data["target"].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0,
        }
    )

    data["target"] = pd.to_numeric(
        data["target"],
        errors="raise"
    ).astype(int)

    return data


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_model(data):

    X = data.drop(
        "target",
        axis=1
    )

    y = data["target"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
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

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    cm = confusion_matrix(
        y_test,
        y_pred
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

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,

        "cm": cm,
        "report": report,

        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


# ============================================================
# HELPER — METRIC CARD
# ============================================================

def metric_card(
    title,
    value,
    description=""
):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HELPER — PLOTLY THEME
# ============================================================

def apply_plotly_theme(fig, height=450):

    fig.update_layout(
        height=height,

        paper_bgcolor="white",
        plot_bgcolor="white",

        font=dict(
            color="#111827",
            size=13
        ),

        margin=dict(
            l=60,
            r=35,
            t=65,
            b=60
        ),

        title_font=dict(
            color="#111827",
            size=19
        ),

        legend=dict(
            font=dict(
                color="#111827",
                size=12
            )
        ),

        hoverlabel=dict(
            bgcolor="white",
            font=dict(
                color="#111827",
                size=13
            )
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,

        tickfont=dict(
            color="#111827",
            size=12
        ),

        title_font=dict(
            color="#111827",
            size=13
        )
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,

        tickfont=dict(
            color="#111827",
            size=12
        ),

        title_font=dict(
            color="#111827",
            size=13
        )
    )

    return fig


# ============================================================
# LOAD DATA
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
        "Place datasett.csv in the same GitHub repository "
        "folder as app.py."
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
# BASIC VALUES
# ============================================================

total_students = len(df)

dropout_students = int(
    (df["target"] == 1).sum()
)

not_dropout_students = int(
    (df["target"] == 0).sum()
)

dropout_rate = (
    dropout_students /
    total_students *
    100
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h2 style="
        color:#111827;
        margin-bottom:0;
    ">
        🎓 Dropout Predictor
    </h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Student Risk Prediction System"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Student Risk Prediction",
        "📊 Exploratory Analysis",
        "🤖 Model Performance",
        "📋 Dataset Overview",
    ]
)


st.sidebar.markdown("---")


st.sidebar.metric(
    "Total Students",
    f"{total_students:,}"
)

st.sidebar.metric(
    "Dropout Rate",
    f"{dropout_rate:.1f}%"
)


st.sidebar.markdown("---")

st.sidebar.info(
    "Dataset: datasett.csv\n\n"
    "Model: Logistic Regression"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="main-header">

            <div class="main-header-title">
                🎓 Student Dropout Prediction
            </div>

            <div class="main-header-subtitle">
                Machine Learning based student risk analysis
                using Logistic Regression
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Total Students",
            f"{total_students:,}",
            "Students in dataset"
        )

    with c2:

        metric_card(
            "Dropout Students",
            f"{dropout_students:,}",
            "Students classified as dropout"
        )

    with c3:

        metric_card(
            "Not Dropout",
            f"{not_dropout_students:,}",
            "Graduate + Enrolled"
        )

    with c4:

        metric_card(
            "Dropout Rate",
            f"{dropout_rate:.1f}%",
            "Overall dataset rate"
        )

    st.write("")

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "📌 Project Overview"
    )

    with st.container(border=True):

        info1, info2, info3, info4 = st.columns(4)

        with info1:

            st.markdown("**Machine Learning Task**")

            st.write(
                "Binary Classification"
            )

        with info2:

            st.markdown("**Algorithm**")

            st.write(
                "Logistic Regression"
            )

        with info3:

            st.markdown("**Input Features**")

            st.write(
                f"{len(results['feature_names'])}"
            )

        with info4:

            st.markdown("**Train / Test Split**")

            st.write(
                "80% / 20%"
            )

    st.write("")

    # --------------------------------------------------------
    # DASHBOARD CHARTS
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    with chart1:

        st.subheader(
            "🎯 Student Status Distribution"
        )

        status_df = pd.DataFrame(
            {
                "Status": [
                    "Not Dropout",
                    "Dropout"
                ],
                "Students": [
                    not_dropout_students,
                    dropout_students
                ]
            }
        )

        status_df["Label"] = status_df[
            "Students"
        ].apply(
            lambda x: f"{x:,}"
        )

        fig = px.bar(
            status_df,
            x="Status",
            y="Students",
            text="Label",
            color="Status",
            color_discrete_map={
                "Not Dropout": "#2563eb",
                "Dropout": "#dc2626"
            }
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                color="#111827",
                size=15
            ),
            cliponaxis=False
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Number of Students",
            xaxis_title="Student Status"
        )

        fig = apply_plotly_theme(
            fig,
            430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TARGET PIE
    # --------------------------------------------------------

    with chart2:

        st.subheader(
            "📈 Student Status Percentage"
        )

        pie_df = pd.DataFrame(
            {
                "Status": [
                    "Not Dropout",
                    "Dropout"
                ],
                "Students": [
                    not_dropout_students,
                    dropout_students
                ]
            }
        )

        fig = px.pie(
            pie_df,
            names="Status",
            values="Students",
            hole=0.48,
            color="Status",
            color_discrete_map={
                "Not Dropout": "#2563eb",
                "Dropout": "#dc2626"
            }
        )

        fig.update_traces(
            textinfo="label+percent",
            textposition="inside",
            textfont=dict(
                color="white",
                size=14
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Students: %{value:,}<br>"
                "Percentage: %{percent}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(
                    color="#111827"
                )
            )
        )

        fig = apply_plotly_theme(
            fig,
            430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # QUICK MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "🤖 Model Performance"
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        metric_card(
            "Accuracy",
            f"{results['accuracy'] * 100:.2f}%",
            "Correct predictions"
        )

    with m2:
        metric_card(
            "Precision",
            f"{results['precision'] * 100:.2f}%",
            "Positive prediction quality"
        )

    with m3:
        metric_card(
            "Recall",
            f"{results['recall'] * 100:.2f}%",
            "Dropout detection"
        )

    with m4:
        metric_card(
            "F1 Score",
            f"{results['f1'] * 100:.2f}%",
            "Balanced metric"
        )

    with m5:
        metric_card(
            "ROC-AUC",
            f"{results['roc_auc']:.3f}",
            "Classification quality"
        )


# ============================================================
# STUDENT RISK PREDICTION
# ============================================================

elif page == "🔮 Student Risk Prediction":

    st.markdown(
        """
        <div class="main-header">

            <div class="main-header-title">
                🔮 Student Risk Prediction
            </div>

            <div class="main-header-subtitle">
                Enter student information and estimate
                dropout probability and risk level.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    feature_names = results[
        "feature_names"
    ]

    X_test = results[
        "X_test"
    ]

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "random_student_index" not in st.session_state:

        st.session_state.random_student_index = None

    # --------------------------------------------------------
    # QUICK TEST
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "🎲 Quick Test Case"
        )

        st.write(
            "Load an actual student record from the test dataset "
            "to test the trained model."
        )

        q1, q2 = st.columns(
            [1, 3]
        )

        with q1:

            if st.button(
                "🎲 Load Random Student",
                use_container_width=True
            ):

                st.session_state.random_student_index = (
                    np.random.choice(
                        X_test.index
                    )
                )

                st.rerun()

        with q2:

            if (
                st.session_state.random_student_index
                is not None
            ):

                st.success(
                    "A real test-set student has been loaded."
                )

    # --------------------------------------------------------
    # DEFAULT RECORD
    # --------------------------------------------------------

    selected_index = (
        st.session_state.random_student_index
    )

    if selected_index is not None:

        default_values = X_test.loc[
            selected_index
        ]

    else:

        default_values = None

    # --------------------------------------------------------
    # INPUT STORAGE
    # --------------------------------------------------------

    inputs = {}

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    with st.form(
        "student_prediction_form"
    ):

        st.subheader(
            "👤 Student Information"
        )

        # ----------------------------------------------------
        # FEATURE GROUPS
        # ----------------------------------------------------

        demographic_columns = [
            "Marital status",
            "Nacionality",
            "Gender",
            "Age at enrollment",
            "International",
            "Displaced",
            "Educational special needs",
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

        financial_columns = [
            "Tuition fees up to date",
            "Scholarship holder",
            "Debtor",
        ]

        other_columns = [
            col
            for col in feature_names
            if col not in (
                demographic_columns
                + enrollment_columns
                + academic_columns
                + financial_columns
            )
        ]

        # Keep only columns actually present
        demographic_columns = [
            col
            for col in demographic_columns
            if col in feature_names
        ]

        enrollment_columns = [
            col
            for col in enrollment_columns
            if col in feature_names
        ]

        academic_columns = [
            col
            for col in academic_columns
            if col in feature_names
        ]

        financial_columns = [
            col
            for col in financial_columns
            if col in feature_names
        ]

        # ----------------------------------------------------
        # INPUT FUNCTION
        # ----------------------------------------------------

        def render_group(
            title,
            columns
        ):

            if not columns:
                return

            st.markdown(
                f"### {title}"
            )

            grid = st.columns(3)

            for index, column in enumerate(columns):

                with grid[index % 3]:

                    if default_values is not None:

                        default_value = default_values[
                            column
                        ]

                    else:

                        default_value = float(
                            df[column].median()
                        )

                    # ----------------------------------------
                    # BINARY FEATURES
                    # ----------------------------------------

                    unique_values = (
                        df[column]
                        .dropna()
                        .unique()
                    )

                    if (
                        len(unique_values) <= 2
                        and set(
                            pd.to_numeric(
                                unique_values,
                                errors="coerce"
                            )
                        ).issubset(
                            {0, 1}
                        )
                    ):

                        try:

                            default_int = int(
                                float(
                                    default_value
                                )
                            )

                        except:

                            default_int = 0

                        option = st.selectbox(
                            column,
                            [0, 1],
                            index=(
                                1
                                if default_int == 1
                                else 0
                            ),
                            key=f"input_{column}"
                        )

                        inputs[column] = option

                    # ----------------------------------------
                    # NORMAL NUMERIC FEATURES
                    # ----------------------------------------

                    else:

                        min_value = float(
                            df[column].min()
                        )

                        max_value = float(
                            df[column].max()
                        )

                        default_number = float(
                            default_value
                        )

                        if default_number < min_value:
                            default_number = min_value

                        if default_number > max_value:
                            default_number = max_value

                        # Integer-like columns
                        if (
                            pd.api.types
                            .is_integer_dtype(
                                df[column]
                            )
                            or
                            (
                                min_value.is_integer()
                                and max_value.is_integer()
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
                                    round(
                                        default_number
                                    )
                                ),
                                step=1,
                                key=f"input_{column}"
                            )

                        else:

                            value = st.number_input(
                                column,
                                min_value=min_value,
                                max_value=max_value,
                                value=default_number,
                                step=0.1,
                                key=f"input_{column}"
                            )

                        inputs[column] = value

            st.write("")

        # ----------------------------------------------------
        # GROUPS
        # ----------------------------------------------------

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
            "🔍 Predict Dropout Risk",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        try:

            # ------------------------------------------------
            # INPUT DATAFRAME
            # ------------------------------------------------

            input_df = pd.DataFrame(
                [inputs]
            )

            input_df = input_df[
                feature_names
            ]

            # Make sure all columns are numeric
            input_df = input_df.apply(
                pd.to_numeric
            )

            # ------------------------------------------------
            # SCALE
            # ------------------------------------------------

            input_scaled = results[
                "scaler"
            ].transform(
                input_df
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            probability = results[
                "model"
            ].predict_proba(
                input_scaled
            )[0, 1]

            predicted_class = int(
                results["model"].predict(
                    input_scaled
                )[0]
            )

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if probability < 0.30:

                risk_label = "Low Risk"

                risk_class = "risk-low"

                risk_color = "#16a34a"

                risk_message = (
                    "The student has a relatively low "
                    "predicted probability of dropping out."
                )

            elif probability < 0.60:

                risk_label = "Medium Risk"

                risk_class = "risk-medium"

                risk_color = "#d97706"

                risk_message = (
                    "The student has a moderate predicted "
                    "probability of dropping out."
                )

            else:

                risk_label = "High Risk"

                risk_class = "risk-high"

                risk_color = "#dc2626"

                risk_message = (
                    "The student has a high predicted "
                    "probability of dropping out."
                )

            predicted_label = (
                "Dropout"
                if predicted_class == 1
                else "Not Dropout"
            )

            # =================================================
            # RESULT HEADER
            # =================================================

            st.write("")

            st.markdown(
                "## 🎯 Prediction Result"
            )

            # =================================================
            # RESULT COLUMNS
            # =================================================

            result_left, result_right = st.columns(
                [1.1, 0.9]
            )

            # -------------------------------------------------
            # GAUGE
            # -------------------------------------------------

            with result_left:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,

                        number={
                            "suffix": "%",
                            "font": {
                                "size": 42,
                                "color": risk_color
                            }
                        },

                        title={
                            "text": "Dropout Probability",
                            "font": {
                                "size": 20,
                                "color": "#111827"
                            }
                        },

                        gauge={

                            "axis": {
                                "range": [0, 100],

                                "tickmode": "array",

                                "tickvals": [
                                    0,
                                    20,
                                    40,
                                    60,
                                    80,
                                    100
                                ],

                                "ticktext": [
                                    "0",
                                    "20",
                                    "40",
                                    "60",
                                    "80",
                                    "100"
                                ],

                                "tickfont": {
                                    "color": "#111827",
                                    "size": 12
                                },

                                "tickcolor": "#111827"
                            },

                            "bar": {
                                "color": risk_color,
                                "thickness": 0.28
                            },

                            "bgcolor": "#f8fafc",

                            "borderwidth": 1,

                            "bordercolor": "#d1d5db",

                            "steps": [

                                {
                                    "range": [0, 30],
                                    "color": "#dcfce7"
                                },

                                {
                                    "range": [30, 60],
                                    "color": "#fef3c7"
                                },

                                {
                                    "range": [60, 100],
                                    "color": "#fee2e2"
                                }
                            ]
                        }
                    )
                )

                gauge.update_layout(
                    height=420,

                    paper_bgcolor="white",

                    margin=dict(
                        l=30,
                        r=30,
                        t=80,
                        b=30
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            with result_right:

                st.markdown(
                    f"""
                    <div class="prediction-result">

                        <div class="prediction-label">
                            RISK CATEGORY
                        </div>

                        <div class="risk-box {risk_class}">

                            <div class="risk-title">
                                Predicted Risk
                            </div>

                            <div class="risk-value">
                                {risk_label}
                            </div>

                        </div>

                        <br>

                        <div class="prediction-label">
                            PREDICTED CLASS
                        </div>

                        <div class="prediction-value"
                             style="color:{risk_color};">

                            {predicted_label}

                        </div>

                        <br>

                        <div class="prediction-label">
                            DROPOUT PROBABILITY
                        </div>

                        <div class="prediction-value"
                             style="color:{risk_color};">

                            {probability * 100:.2f}%

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                if predicted_class == 1:

                    st.error(
                        "⚠️ Model prediction: "
                        "This student is classified as Dropout."
                    )

                else:

                    st.success(
                        "✅ Model prediction: "
                        "This student is classified as Not Dropout."
                    )

            # =================================================
            # RISK EXPLANATION
            # =================================================

            st.write("")

            with st.container(border=True):

                st.subheader(
                    "📌 Risk Interpretation"
                )

                st.write(
                    risk_message
                )

                st.write(
                    f"**Probability:** "
                    f"{probability * 100:.2f}%"
                )

                st.write(
                    f"**Predicted Class:** "
                    f"{predicted_label}"
                )

                st.write(
                    f"**Risk Level:** "
                    f"{risk_label}"
                )

            # =================================================
            # PROBABILITY BREAKDOWN
            # =================================================

            st.write("")

            st.subheader(
                "📊 Prediction Probability Breakdown"
            )

            probability_df = pd.DataFrame(
                {
                    "Class": [
                        "Not Dropout",
                        "Dropout"
                    ],

                    "Probability": [
                        (1 - probability) * 100,
                        probability * 100
                    ]
                }
            )

            probability_df["Label"] = (
                probability_df[
                    "Probability"
                ].map(
                    lambda x:
                    f"{x:.2f}%"
                )
            )

            probability_fig = px.bar(
                probability_df,
                x="Class",
                y="Probability",
                text="Label",
                color="Class",
                color_discrete_map={
                    "Not Dropout": "#2563eb",
                    "Dropout": "#dc2626"
                }
            )

            probability_fig.update_traces(
                textposition="outside",
                textfont=dict(
                    color="#111827",
                    size=15
                ),
                cliponaxis=False
            )

            probability_fig.update_layout(
                yaxis_title="Probability (%)",
                xaxis_title="Prediction Class",
                yaxis=dict(
                    range=[
                        0,
                        max(
                            100,
                            probability_df[
                                "Probability"
                            ].max() + 10
                        )
                    ]
                ),
                showlegend=False
            )

            probability_fig = apply_plotly_theme(
                probability_fig,
                430
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True
            )

            # =================================================
            # REAL TEST STUDENT OUTCOME
            # =================================================

            if selected_index is not None:

                actual_target = int(
                    df.loc[
                        selected_index,
                        "target"
                    ]
                )

                actual_label = (
                    "Dropout"
                    if actual_target == 1
                    else "Not Dropout"
                )

                st.write("")

                st.subheader(
                    "🧪 Actual Test-Set Result"
                )

                a1, a2, a3 = st.columns(3)

                with a1:

                    metric_card(
                        "Predicted Class",
                        predicted_label,
                        "Model prediction"
                    )

                with a2:

                    metric_card(
                        "Actual Class",
                        actual_label,
                        "Dataset result"
                    )

                with a3:

                    if (
                        predicted_label
                        == actual_label
                    ):

                        metric_card(
                            "Prediction",
                            "Correct",
                            "Model matched actual result"
                        )

                    else:

                        metric_card(
                            "Prediction",
                            "Incorrect",
                            "Model differed from actual result"
                        )

        except Exception as error:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.exception(error)


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "📊 Exploratory Analysis":

    st.markdown(
        """
        <div class="main-header">

            <div class="main-header-title">
                📊 Exploratory Data Analysis
            </div>

            <div class="main-header-subtitle">
                Visual analysis of student data and
                dropout-related patterns.
            </div>

        </div>
        """,
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
    # CHART 1
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🎯 Dropout vs Not Dropout"
        )

        counts = (
            plot_df[
                "Status"
            ]
            .value_counts()
            .reindex(
                [
                    "Not Dropout",
                    "Dropout"
                ]
            )
            .fillna(0)
            .reset_index()
        )

        counts.columns = [
            "Status",
            "Count"
        ]

        counts["Label"] = (
            counts["Count"]
            .astype(int)
            .map(
                lambda x:
                f"{x:,}"
            )
        )

        fig = px.bar(
            counts,
            x="Status",
            y="Count",
            text="Label",
            color="Status",
            color_discrete_map={
                "Not Dropout": "#2563eb",
                "Dropout": "#dc2626"
            }
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                color="#111827",
                size=15
            ),
            cliponaxis=False
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="Number of Students"
        )

        fig = apply_plotly_theme(
            fig,
            450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # CHART 2
    # ========================================================

    with col2:

        st.subheader(
            "💰 Tuition Fees vs Dropout"
        )

        if "Tuition fees up to date" in df.columns:

            tuition_df = (
                df.groupby(
                    "Tuition fees up to date"
                )["target"]
                .mean()
                .reset_index()
            )

            tuition_df["Dropout Rate"] = (
                tuition_df["target"] * 100
            )

            tuition_df["Status"] = (
                tuition_df[
                    "Tuition fees up to date"
                ]
                .map(
                    {
                        0: "Not Up to Date",
                        1: "Up to Date"
                    }
                )
                .fillna(
                    tuition_df[
                        "Tuition fees up to date"
                    ].astype(str)
                )
            )

            tuition_df["Label"] = (
                tuition_df[
                    "Dropout Rate"
                ].map(
                    lambda x:
                    f"{x:.1f}%"
                )
            )

            fig = px.bar(
                tuition_df,
                x="Status",
                y="Dropout Rate",
                text="Label"
            )

            fig.update_traces(
                marker_color="#f59e0b",
                textposition="outside",
                textfont=dict(
                    color="#111827",
                    size=15
                ),
                cliponaxis=False
            )

            fig.update_layout(
                yaxis_title="Dropout Rate (%)",
                xaxis_title="Tuition Fee Status"
            )

            fig = apply_plotly_theme(
                fig,
                450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Tuition fees feature is not available."
            )

    # ========================================================
    # CHART 3
    # ========================================================

    if (
        "Curricular units 1st sem (grade)"
        in df.columns
    ):

        st.subheader(
            "📚 1st Semester Grade vs Dropout"
        )

        fig = px.box(
            plot_df,
            x="Status",
            y="Curricular units 1st sem (grade)",
            color="Status",
            color_discrete_map={
                "Not Dropout": "#2563eb",
                "Dropout": "#dc2626"
            },
            points=False
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="1st Semester Grade"
        )

        fig = apply_plotly_theme(
            fig,
            480
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # CHART 4
    # ========================================================

    if (
        "Curricular units 2nd sem (grade)"
        in df.columns
    ):

        st.subheader(
            "📖 2nd Semester Grade vs Dropout"
        )

        fig = px.box(
            plot_df,
            x="Status",
            y="Curricular units 2nd sem (grade)",
            color="Status",
            color_discrete_map={
                "Not Dropout": "#2563eb",
                "Dropout": "#dc2626"
            },
            points=False
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Student Status",
            yaxis_title="2nd Semester Grade"
        )

        fig = apply_plotly_theme(
            fig,
            480
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # FEATURE CORRELATION
    # ========================================================

    st.subheader(
        "🔗 Feature Correlation with Dropout"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if "target" in numeric_df.columns:

        correlations = (
            numeric_df
            .corr()["target"]
            .drop("target")
            .sort_values()
        )

        correlation_display = (
            correlations
            .abs()
            .sort_values(
                ascending=False
            )
            .head(12)
            .sort_values()
            .reset_index()
        )

        correlation_display.columns = [
            "Feature",
            "Absolute Correlation"
        ]

        correlation_display["Label"] = (
            correlation_display[
                "Absolute Correlation"
            ].map(
                lambda x:
                f"{x:.3f}"
            )
        )

        fig = px.bar(
            correlation_display,
            x="Absolute Correlation",
            y="Feature",
            orientation="h",
            text="Label"
        )

        fig.update_traces(
            marker_color="#6366f1",
            textposition="outside",
            textfont=dict(
                color="#111827",
                size=12
            ),
            cliponaxis=False
        )

        fig.update_layout(
            xaxis_title="Absolute Correlation",
            yaxis_title="Feature"
        )

        fig = apply_plotly_theme(
            fig,
            550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        """
        <div class="main-header">

            <div class="main-header-title">
                🤖 Model Performance
            </div>

            <div class="main-header-subtitle">
                Evaluation of the Logistic Regression model
                on the test dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        metric_card(
            "Accuracy",
            f"{results['accuracy'] * 100:.2f}%",
            "Overall correctness"
        )

    with c2:

        metric_card(
            "Precision",
            f"{results['precision'] * 100:.2f}%",
            "Positive prediction precision"
        )

    with c3:

        metric_card(
            "Recall",
            f"{results['recall'] * 100:.2f}%",
            "Dropout detection"
        )

    with c4:

        metric_card(
            "F1 Score",
            f"{results['f1'] * 100:.2f}%",
            "Balanced performance"
        )

    with c5:

        metric_card(
            "ROC-AUC",
            f"{results['roc_auc']:.3f}",
            "Area under ROC curve"
        )

    st.write("")

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    left, right = st.columns(2)

    with left:

        st.subheader(
            "🧩 Confusion Matrix"
        )

        cm = results["cm"]

        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,

                x=[
                    "Predicted Not Dropout",
                    "Predicted Dropout"
                ],

                y=[
                    "Actual Not Dropout",
                    "Actual Dropout"
                ],

                colorscale=[
                    [0, "#dbeafe"],
                    [1, "#2563eb"]
                ],

                text=cm,

                texttemplate="%{text}",

                textfont=dict(
                    color="#111827",
                    size=22
                ),

                hovertemplate=(
                    "Value: %{z}<extra></extra>"
                ),

                colorbar=dict(
                    title="Count"
                )
            )
        )

        cm_fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=100,
                r=40,
                t=70,
                b=100
            ),

            font=dict(
                color="#111827"
            ),

            xaxis=dict(
                tickfont=dict(
                    color="#111827",
                    size=12
                )
            ),

            yaxis=dict(
                tickfont=dict(
                    color="#111827",
                    size=12
                )
            )
        )

        st.plotly_chart(
            cm_fig,
            use_container_width=True
        )

    # ========================================================
    # ROC CURVE
    # ========================================================

    with right:

        st.subheader(
            "📈 ROC Curve"
        )

        roc_fig = go.Figure()

        roc_fig.add_trace(
            go.Scatter(
                x=results["fpr"],
                y=results["tpr"],
                mode="lines",

                name=(
                    f"ROC Curve "
                    f"(AUC = "
                    f"{results['roc_auc']:.3f})"
                ),

                line=dict(
                    width=3,
                    color="#2563eb"
                )
            )
        )

        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",

                name="Random Classifier",

                line=dict(
                    dash="dash",
                    width=2,
                    color="#9ca3af"
                )
            )
        )

        roc_fig.update_layout(
            height=500,

            xaxis=dict(
                title="False Positive Rate",
                range=[0, 1],
                tickfont=dict(
                    color="#111827"
                ),
                title_font=dict(
                    color="#111827"
                )
            ),

            yaxis=dict(
                title="True Positive Rate",
                range=[0, 1],
                tickfont=dict(
                    color="#111827"
                ),
                title_font=dict(
                    color="#111827"
                )
            ),

            paper_bgcolor="white",
            plot_bgcolor="white",

            font=dict(
                color="#111827"
            ),

            legend=dict(
                font=dict(
                    color="#111827"
                )
            ),

            margin=dict(
                l=60,
                r=30,
                t=70,
                b=60
            )
        )

        roc_fig.update_xaxes(
            showgrid=True,
            gridcolor="#e5e7eb"
        )

        roc_fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb"
        )

        st.plotly_chart(
            roc_fig,
            use_container_width=True
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "📋 Classification Report"
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

    # ========================================================
    # CONFUSION MATRIX DETAILS
    # ========================================================

    st.subheader(
        "🔍 Prediction Details"
    )

    tn, fp, fn, tp = (
        results["cm"].ravel()
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        metric_card(
            "True Negative",
            f"{int(tn)}",
            "Correctly predicted Not Dropout"
        )

    with d2:

        metric_card(
            "False Positive",
            f"{int(fp)}",
            "Predicted Dropout incorrectly"
        )

    with d3:

        metric_card(
            "False Negative",
            f"{int(fn)}",
            "Missed dropout students"
        )

    with d4:

        metric_card(
            "True Positive",
            f"{int(tp)}",
            "Correctly predicted Dropout"
        )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "📋 Dataset Overview":

    st.markdown(
        """
        <div class="main-header">

            <div class="main-header-title">
                📋 Dataset Overview
            </div>

            <div class="main-header-subtitle">
                Explore the structure, records and
                statistical information of the dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Rows",
            f"{df.shape[0]:,}",
            "Total records"
        )

    with c2:

        metric_card(
            "Columns",
            f"{df.shape[1]:,}",
            "Total dataset columns"
        )

    with c3:

        metric_card(
            "Features",
            f"{df.shape[1] - 1:,}",
            "Input features"
        )

    with c4:

        missing_values = int(
            df.isnull()
            .sum()
            .sum()
        )

        metric_card(
            "Missing Values",
            f"{missing_values:,}",
            "Total missing cells"
        )

    st.write("")

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📄 Dataset Preview",
            "🔤 Data Types",
            "📊 Statistics",
            "🎯 Target"
        ]
    )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=500
        )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    with tab2:

        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
                "Missing Values": [
                    int(value)
                    for value in df.isnull().sum()
                ],
                "Unique Values": [
                    int(value)
                    for value in df.nunique()
                ]
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True,
            height=600
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    with tab3:

        stats_df = (
            df.describe()
            .transpose()
            .round(3)
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
            height=600
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    with tab4:

        target_counts = (
            df["target"]
            .value_counts()
            .sort_index()
        )

        target_table = pd.DataFrame(
            {
                "Encoded Value": [
                    0,
                    1
                ],
                "Meaning": [
                    "Not Dropout",
                    "Dropout"
                ],
                "Students": [
                    int(
                        target_counts.get(
                            0,
                            0
                        )
                    ),
                    int(
                        target_counts.get(
                            1,
                            0
                        )
                    )
                ]
            }
        )

        st.dataframe(
            target_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Student Dropout Prediction System
        <br>
        Logistic Regression • Streamlit • Scikit-learn • Plotly

    </div>
    """,
    unsafe_allow_html=True
)
