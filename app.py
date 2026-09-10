import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        background-color: #f7f8fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 50%,
            #374151 100%
        );

        padding: 35px 40px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        color: white;
    }

    .hero p {
        font-size: 17px;
        color: #d1d5db;
        margin-bottom: 0;
    }

    /* Section titles */
    .section-title {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 22px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
        text-align: center;
        min-height: 130px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    /* Result cards */
    .result-card {
        background: white;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }

    .result-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    /* Risk boxes */
    .risk-low {
        background: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
    }

    .risk-medium {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e;
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
    }

    .risk-high {
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
    }

    /* Info card */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    .info-card h3 {
        color: #111827;
        margin-bottom: 12px;
    }

    .info-card p {
        color: #4b5563;
        line-height: 1.7;
    }

    /* Footer */
    .footer {
        margin-top: 50px;
        padding: 25px;
        text-align: center;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

DATASET_FILE = "datasett.csv"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET_FILE)

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = load_dataset().copy()

    # --------------------------------------------------------
    # Target encoding
    # --------------------------------------------------------

    if "target" not in df.columns:
        raise ValueError(
            "The dataset must contain a column named 'target'."
        )

    df["target"] = df["target"].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0
        }
    )

    # Convert target to numeric
    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce"
    )

    # Remove rows with invalid target
    df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = df.drop("target", axis=1)

    y = df["target"]

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

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
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "cm": cm,
        "report": report
    }


# ============================================================
# LOAD MODEL
# ============================================================

try:

    data = load_dataset()

    results = train_model()

except Exception as e:

    st.error(
        f"Application could not start: {str(e)}"
    )

    st.stop()


# ============================================================
# EXTRACT MODEL INFORMATION
# ============================================================

df = results["df"]

X = results["X"]

y = results["y"]

model = results["model"]

scaler = results["scaler"]

accuracy = results["accuracy"]

roc_auc = results["roc_auc"]

cm = results["cm"]

report = results["report"]

y_test = results["y_test"]

y_pred = results["y_pred"]

y_prob = results["y_prob"]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:20px 5px;">

        <div style="font-size:45px;">
            🎓
        </div>

        <h2 style="margin-bottom:0;">
            Student AI
        </h2>

        <p style="color:#9ca3af;">
            Dropout Prediction System
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Student Risk Prediction",
        "📊 Dataset Overview",
        "📈 Exploratory Analysis",
        "🤖 Model Performance"
    ]
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    <div style="text-align:center; color:#9ca3af;">

    <small>
    Machine Learning Application
    </small>

    <br>

    <small>
    Logistic Regression
    </small>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>
            Student Dropout Prediction
        </h1>

        <p>
            Machine Learning based student risk analysis
            using Logistic Regression
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            An overview of the dataset, trained model and
            prediction system.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    TOTAL STUDENTS
                </div>

                <div class="metric-value">
                    {len(df):,}
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
                    INPUT FEATURES
                </div>

                <div class="metric-value">
                    {X.shape[1]}
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
                    MODEL ACCURACY
                </div>

                <div class="metric-value">
                    {accuracy * 100:.2f}%
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
                    ROC-AUC
                </div>

                <div class="metric-value">
                    {roc_auc:.3f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Dataset target distribution
    # --------------------------------------------------------

    col1, col2 = st.columns([1.3, 1])

    with col1:

        st.subheader(
            "Student Outcome Distribution"
        )

        target_counts = df["target"].map(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        ).value_counts()

        chart_df = pd.DataFrame(
            {
                "Status": target_counts.index,
                "Students": target_counts.values
            }
        )

        fig = px.bar(
            chart_df,
            x="Status",
            y="Students",
            text="Students"
        )

        fig.update_traces(
            textfont_color="black",
            textposition="outside"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                color="black"
            ),
            xaxis=dict(
                title="Student Status",
                tickfont=dict(color="black"),
                title_font=dict(color="black")
            ),
            yaxis=dict(
                title="Number of Students",
                tickfont=dict(color="black"),
                title_font=dict(color="black")
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Model Information"
        )

        st.markdown(
            """
            <div class="info-card">

            <h3>Logistic Regression</h3>

            <p>
            This application uses Logistic Regression to
            estimate whether a student is at risk of dropping
            out.
            </p>

            <p>
            Before training, numerical features are
            standardized using StandardScaler.
            </p>

            <p>
            <b>Dropout:</b> 1
            <br>
            <b>Not Dropout:</b> 0
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# STUDENT RISK PREDICTION
# ============================================================

elif page == "🔮 Student Risk Prediction":

    st.markdown(
        '<div class="section-title">Student Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Enter student information below to estimate the
            probability of dropout.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    feature_columns = list(X.columns)

    # --------------------------------------------------------
    # Create default values
    # --------------------------------------------------------

    defaults = {}

    for column in feature_columns:

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):

            median_value = X[column].median()

            if pd.isna(median_value):
                median_value = 0

            defaults[column] = float(
                median_value
            )

        else:

            mode_value = X[column].mode()

            if len(mode_value) > 0:
                defaults[column] = mode_value.iloc[0]
            else:
                defaults[column] = 0

    # --------------------------------------------------------
    # Personal and demographic section
    # --------------------------------------------------------

    st.subheader(
        "👤 Personal & Demographic Information"
    )

    col1, col2, col3 = st.columns(3)

    user_values = {}

    with col1:

        field = "Age at enrollment"

        if field in feature_columns:

            user_values[field] = st.number_input(
                field,
                min_value=15.0,
                max_value=80.0,
                value=float(
                    defaults[field]
                ),
                step=1.0
            )

    with col2:

        field = "Gender"

        if field in feature_columns:

            user_values[field] = st.number_input(
                field,
                min_value=0.0,
                max_value=1.0,
                value=float(
                    defaults[field]
                ),
                step=1.0
            )

    with col3:

        field = "Marital status"

        if field in feature_columns:

            user_values[field] = st.number_input(
                field,
                min_value=0.0,
                max_value=10.0,
                value=float(
                    defaults[field]
                ),
                step=1.0
            )

    # --------------------------------------------------------
    # Financial information
    # --------------------------------------------------------

    st.subheader(
        "💰 Financial & Enrollment Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        field = "Tuition fees up to date"

        if field in feature_columns:

            user_values[field] = st.selectbox(
                field,
                options=[0, 1],
                index=int(
                    round(
                        defaults[field]
                    )
                )
            )

    with col2:

        field = "Scholarship holder"

        if field in feature_columns:

            user_values[field] = st.selectbox(
                field,
                options=[0, 1],
                index=int(
                    round(
                        defaults[field]
                    )
                )
            )

    with col3:

        field = "Debtor"

        if field in feature_columns:

            user_values[field] = st.selectbox(
                field,
                options=[0, 1],
                index=int(
                    round(
                        defaults[field]
                    )
                )
            )

    # --------------------------------------------------------
    # Academic performance
    # --------------------------------------------------------

    st.subheader(
        "📚 Academic Performance"
    )

    academic_columns = [
        column
        for column in feature_columns
        if (
            "grade" in column.lower()
            or "units" in column.lower()
        )
    ]

    # Remove fields already manually handled
    academic_columns = [
        column
        for column in academic_columns
        if column not in user_values
    ]

    # --------------------------------------------------------
    # Display academic inputs dynamically
    # --------------------------------------------------------

    if len(academic_columns) > 0:

        academic_cols = st.columns(3)

        for index, column in enumerate(
            academic_columns
        ):

            with academic_cols[index % 3]:

                minimum = 0.0
                maximum = 20.0

                # Unit count fields can have larger ranges
                if (
                    "enrolled" in column.lower()
                    or "approved" in column.lower()
                    or "evaluations" in column.lower()
                ):

                    maximum = 30.0

                user_values[column] = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=float(
                        max(
                            minimum,
                            min(
                                maximum,
                                defaults[column]
                            )
                        )
                    ),
                    step=0.1
                )

    # --------------------------------------------------------
    # Other features
    # --------------------------------------------------------

    other_columns = [
        column
        for column in feature_columns
        if column not in user_values
    ]

    if len(other_columns) > 0:

        with st.expander(
            "⚙️ Additional Student Information"
        ):

            other_cols = st.columns(3)

            for index, column in enumerate(
                other_columns
            ):

                with other_cols[index % 3]:

                    # Binary fields
                    unique_values = X[column].dropna().unique()

                    if (
                        len(unique_values) <= 2
                        and set(
                            pd.to_numeric(
                                pd.Series(
                                    unique_values
                                ),
                                errors="coerce"
                            ).dropna().unique()
                        ).issubset({0, 1})
                    ):

                        user_values[column] = st.selectbox(
                            column,
                            options=[0, 1],
                            index=int(
                                round(
                                    defaults[column]
                                )
                            )
                        )

                    else:

                        user_values[column] = st.number_input(
                            column,
                            value=float(
                                defaults[column]
                            )
                        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Analyze Student Risk",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if predict_button:

        try:

            # ------------------------------------------------
            # Build input using exact training column order
            # ------------------------------------------------

            input_data = {}

            for column in feature_columns:

                if column in user_values:

                    input_data[column] = user_values[
                        column
                    ]

                else:

                    input_data[column] = defaults[
                        column
                    ]

            input_df = pd.DataFrame(
                [input_data]
            )

            # Ensure exact order
            input_df = input_df[
                feature_columns
            ]

            # Convert values to numeric
            for column in input_df.columns:

                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce"
                )

            # Fill missing values
            input_df = input_df.fillna(
                X.median(
                    numeric_only=True
                )
            )

            # ------------------------------------------------
            # Scale input
            # ------------------------------------------------

            input_scaled = scaler.transform(
                input_df
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = model.predict(
                input_scaled
            )[0]

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            # ------------------------------------------------
            # Risk level
            # ------------------------------------------------

            if probability < 0.30:

                risk_level = "LOW RISK"

                risk_class = "risk-low"

            elif probability < 0.60:

                risk_level = "MEDIUM RISK"

                risk_class = "risk-medium"

            else:

                risk_level = "HIGH RISK"

                risk_class = "risk-high"

            # ------------------------------------------------
            # Result heading
            # ------------------------------------------------

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Result cards
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            DROPOUT PROBABILITY
                        </div>

                        <div class="result-value">
                            {probability * 100:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                predicted_status = (
                    "Dropout"
                    if prediction == 1
                    else "Not Dropout"
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            PREDICTED STATUS
                        </div>

                        <div class="result-value">
                            {predicted_status}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            RISK CATEGORY
                        </div>

                        <div class="result-value">
                            {risk_level}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------------
            # Risk message
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="{risk_class}">
                    {risk_level}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------------
            # Charts
            # ------------------------------------------------

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.subheader(
                    "Dropout Probability"
                )

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={
                            "suffix": "%",
                            "font": {
                                "color": "black",
                                "size": 36
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
                                "color": "#111827"
                            },
                            "bgcolor": "#e5e7eb",
                            "borderwidth": 1,
                            "bordercolor": "#9ca3af"
                        }
                    )
                )

                gauge.update_layout(
                    paper_bgcolor="white",
                    font={
                        "color": "black"
                    },
                    height=350
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            with chart_col2:

                st.subheader(
                    "Prediction Breakdown"
                )

                probability_df = pd.DataFrame(
                    {
                        "Outcome": [
                            "Not Dropout",
                            "Dropout"
                        ],
                        "Probability": [
                            (1 - probability) * 100,
                            probability * 100
                        ]
                    }
                )

                probability_fig = px.bar(
                    probability_df,
                    x="Outcome",
                    y="Probability",
                    text="Probability"
                )

                probability_fig.update_traces(
                    texttemplate="%{text:.2f}%",
                    textfont_color="black",
                    textposition="outside"
                )

                probability_fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font={
                        "color": "black"
                    },
                    xaxis={
                        "title": "Outcome",
                        "tickfont": {
                            "color": "black"
                        },
                        "title_font": {
                            "color": "black"
                        }
                    },
                    yaxis={
                        "title": "Probability (%)",
                        "range": [0, 110],
                        "tickfont": {
                            "color": "black"
                        },
                        "title_font": {
                            "color": "black"
                        }
                    },
                    showlegend=False,
                    height=350
                )

                st.plotly_chart(
                    probability_fig,
                    use_container_width=True
                )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            if prediction == 1:

                st.warning(
                    "The model estimates that this student "
                    "has a higher probability of dropping out. "
                    "The student may require additional "
                    "academic or support attention."
                )

            else:

                st.success(
                    "The model estimates that this student "
                    "has a lower probability of dropping out."
                )

            st.info(
                "This prediction is an estimated risk indicator "
                "generated by the Logistic Regression model. "
                "It should not be treated as a definitive "
                "decision about a student's future."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {str(e)}"
            )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "📊 Dataset Overview":

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Explore the structure and quality of the student
            dataset used for model training.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    ROWS
                </div>

                <div class="metric-value">
                    {df.shape[0]:,}
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
                    COLUMNS
                </div>

                <div class="metric-value">
                    {df.shape[1]}
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
                    FEATURES
                </div>

                <div class="metric-value">
                    {X.shape[1]}
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
                    DUPLICATES
                </div>

                <div class="metric-value">
                    {df.duplicated().sum()}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Dataset preview
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=350
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    st.subheader(
        "Missing Values"
    )

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame(
        {
            "Column": missing_values.index,
            "Missing Values": missing_values.values
        }
    )

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if len(missing_df) == 0:

        st.success(
            "No missing values were found in the dataset."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    with st.expander(
        "View Data Types"
    ):

        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ]
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True
        )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "📈 Exploratory Analysis":

    st.markdown(
        '<div class="section-title">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Visual analysis of important factors related to
            student dropout.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    st.subheader(
        "1. Student Outcome Distribution"
    )

    outcome_counts = df["target"].value_counts()

    outcome_df = pd.DataFrame(
        {
            "Status": [
                "Not Dropout",
                "Dropout"
            ],
            "Students": [
                outcome_counts.get(0, 0),
                outcome_counts.get(1, 0)
            ]
        }
    )

    fig1 = px.pie(
        outcome_df,
        names="Status",
        values="Students",
        hole=0.45
    )

    fig1.update_traces(
        textinfo="label+percent",
        textfont_color="black"
    )

    fig1.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={
            "color": "black"
        },
        legend={
            "font": {
                "color": "black"
            }
        }
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Semester grades
    # --------------------------------------------------------

    grade1 = (
        "Curricular units 1st sem (grade)"
    )

    grade2 = (
        "Curricular units 2nd sem (grade)"
    )

    col1, col2 = st.columns(2)

    if grade1 in df.columns:

        with col1:

            st.subheader(
                "1st Semester Grade vs Dropout"
            )

            grade_df = df.copy()

            grade_df["Status"] = grade_df[
                "target"
            ].map(
                {
                    0: "Not Dropout",
                    1: "Dropout"
                }
            )

            fig2 = px.box(
                grade_df,
                x="Status",
                y=grade1,
                points=False
            )

            fig2.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font={
                    "color": "black"
                },
                xaxis={
                    "title": "Student Status",
                    "tickfont": {
                        "color": "black"
                    },
                    "title_font": {
                        "color": "black"
                    }
                },
                yaxis={
                    "title": "Grade",
                    "tickfont": {
                        "color": "black"
                    },
                    "title_font": {
                        "color": "black"
                    }
                },
                showlegend=False
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    if grade2 in df.columns:

        with col2:

            st.subheader(
                "2nd Semester Grade vs Dropout"
            )

            grade_df = df.copy()

            grade_df["Status"] = grade_df[
                "target"
            ].map(
                {
                    0: "Not Dropout",
                    1: "Dropout"
                }
            )

            fig3 = px.box(
                grade_df,
                x="Status",
                y=grade2,
                points=False
            )

            fig3.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font={
                    "color": "black"
                },
                xaxis={
                    "title": "Student Status",
                    "tickfont": {
                        "color": "black"
                    },
                    "title_font": {
                        "color": "black"
                    }
                },
                yaxis={
                    "title": "Grade",
                    "tickfont": {
                        "color": "black"
                    },
                    "title_font": {
                        "color": "black"
                    }
                },
                showlegend=False
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

    # --------------------------------------------------------
    # Tuition fees
    # --------------------------------------------------------

    tuition_column = "Tuition fees up to date"

    if tuition_column in df.columns:

        st.subheader(
            "2. Tuition Fees Status vs Dropout"
        )

        tuition_df = (
            df.groupby(
                tuition_column
            )["target"]
            .mean()
            .reset_index()
        )

        tuition_df["Dropout Rate"] = (
            tuition_df["target"] * 100
        )

        tuition_df["Fee Status"] = (
            tuition_df[
                tuition_column
            ]
            .map(
                {
                    0: "Not Up to Date",
                    1: "Up to Date"
                }
            )
            .fillna(
                tuition_df[
                    tuition_column
                ].astype(str)
            )
        )

        fig4 = px.bar(
            tuition_df,
            x="Fee Status",
            y="Dropout Rate",
            text="Dropout Rate"
        )

        fig4.update_traces(
            texttemplate="%{text:.2f}%",
            textfont_color="black",
            textposition="outside"
        )

        fig4.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={
                "color": "black"
            },
            xaxis={
                "title": "Tuition Fee Status",
                "tickfont": {
                    "color": "black"
                },
                "title_font": {
                    "color": "black"
                }
            },
            yaxis={
                "title": "Dropout Rate (%)",
                "tickfont": {
                    "color": "black"
                },
                "title_font": {
                    "color": "black"
                }
            },
            showlegend=False
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Evaluation results of the Logistic Regression
            model on the test dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    ACCURACY
                </div>

                <div class="metric-value">
                    {accuracy * 100:.2f}%
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
                    ROC-AUC
                </div>

                <div class="metric-value">
                    {roc_auc:.3f}
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
                    TEST SAMPLES
                </div>

                <div class="metric-value">
                    {len(y_test)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

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
            text=cm,
            texttemplate="%{text}",
            textfont={
                "color": "black",
                "size": 18
            },
            colorscale=[
                [0, "#f3f4f6"],
                [1, "#9ca3af"]
            ],
            showscale=False
        )
    )

    cm_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={
            "color": "black"
        },
        xaxis={
            "title": "Predicted Class",
            "tickfont": {
                "color": "black"
            },
            "title_font": {
                "color": "black"
            }
        },
        yaxis={
            "title": "Actual Class",
            "tickfont": {
                "color": "black"
            },
            "title_font": {
                "color": "black"
            }
        },
        height=450
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    st.subheader(
        "Classification Report"
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df = report_df.round(3)

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Actual vs predicted distribution
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Predicted Distribution"
    )

    comparison_df = pd.DataFrame(
        {
            "Class": [
                "Not Dropout",
                "Dropout"
            ],
            "Actual": [
                int(
                    (y_test == 0).sum()
                ),
                int(
                    (y_test == 1).sum()
                )
            ],
            "Predicted": [
                int(
                    (y_pred == 0).sum()
                ),
                int(
                    (y_pred == 1).sum()
                )
            ]
        }
    )

    comparison_long = comparison_df.melt(
        id_vars="Class",
        value_vars=[
            "Actual",
            "Predicted"
        ],
        var_name="Type",
        value_name="Students"
    )

    comparison_fig = px.bar(
        comparison_long,
        x="Class",
        y="Students",
        color="Type",
        barmode="group",
        text="Students"
    )

    comparison_fig.update_traces(
        textfont_color="black",
        textposition="outside"
    )

    comparison_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={
            "color": "black"
        },
        xaxis={
            "title": "Student Class",
            "tickfont": {
                "color": "black"
            },
            "title_font": {
                "color": "black"
            }
        },
        yaxis={
            "title": "Number of Students",
            "tickfont": {
                "color": "black"
            },
            "title_font": {
                "color": "black"
            }
        },
        legend={
            "font": {
                "color": "black"
            }
        }
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Model explanation
    # --------------------------------------------------------

    st.subheader(
        "How the Model Works"
    )

    st.markdown(
        """
        <div class="info-card">

        <h3>Machine Learning Pipeline</h3>

        <p>
        <b>1. Dataset:</b>
        Student academic, demographic and enrollment
        information is loaded from the dataset.
        </p>

        <p>
        <b>2. Target Encoding:</b>
        Dropout is represented by 1, while Graduate and
        Enrolled are represented as Not Dropout (0).
        </p>

        <p>
        <b>3. Train-Test Split:</b>
        80% of the data is used for training and 20% is
        used for testing.
        </p>

        <p>
        <b>4. Standardization:</b>
        StandardScaler transforms the numerical features
        before training.
        </p>

        <p>
        <b>5. Logistic Regression:</b>
        The model learns patterns associated with student
        dropout.
        </p>

        <p>
        <b>6. Prediction:</b>
        For a new student, the model calculates the
        probability of dropout.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>Student Dropout Prediction System</b>

        <br>

        Built with Python, Streamlit,
        Scikit-learn and Plotly

        <br><br>

        Logistic Regression • Student Risk Analysis

    </div>
    """,
    unsafe_allow_html=True
)
