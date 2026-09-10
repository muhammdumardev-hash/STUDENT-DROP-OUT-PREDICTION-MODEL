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
# PAGE CONFIG
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

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }


    /* ======================================================
       MAIN HEADER
    ====================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );

        padding: 35px 40px;
        border-radius: 20px;
        margin-bottom: 28px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.12);
    }

    .hero-title {
        color: white;
        font-size: 40px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #d1d5db;
        font-size: 16px;
        margin-top: 8px;
    }


    /* ======================================================
       SECTION TITLE
    ====================================================== */

    .section-title {
        color: #111827;
        font-size: 27px;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }


    /* ======================================================
       METRIC CARDS
    ====================================================== */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        text-align: center;

        box-shadow:
            0 5px 18px rgba(0, 0, 0, 0.06);
    }

    .metric-label {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }


    /* ======================================================
       RESULT CARD
    ====================================================== */

    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 30px;

        box-shadow:
            0 8px 28px rgba(0, 0, 0, 0.08);

        margin-top: 20px;
    }


    /* ======================================================
       INFO CARD
    ====================================================== */

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #111827;

        border-radius: 12px;

        padding: 18px;
        margin-bottom: 15px;
    }

    .info-title {
        color: #111827;
        font-weight: 750;
        font-size: 16px;
    }

    .info-text {
        color: #6b7280;
        font-size: 14px;
        margin-top: 5px;
    }


    /* ======================================================
       BUTTONS
    ====================================================== */

    .stButton > button {
        height: 48px;
        border-radius: 10px;
        font-weight: 700;
    }


    /* ======================================================
       INPUTS
    ====================================================== */

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    input {
        border-radius: 9px !important;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;

        margin-top: 50px;
        padding-top: 20px;

        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv("datasett.csv")


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = load_dataset().copy()

    # --------------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------------

    df["target"] = df["target"].replace(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0
        }
    )

    df["target"] = df["target"].astype(int)


    # --------------------------------------------------------
    # FEATURES / TARGET
    # --------------------------------------------------------

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_scaled
    )

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
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
        "y_probability": y_probability,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "cm": cm,
        "report": report
    }


# ============================================================
# LOAD MODEL
# ============================================================

try:

    data = train_model()

except FileNotFoundError:

    st.error(
        "datasett.csv was not found. "
        "Make sure datasett.csv is in the same GitHub repository "
        "as app.py."
    )

    st.stop()


df = data["df"]
X = data["X"]
y = data["y"]

model = data["model"]
scaler = data["scaler"]

X_train = data["X_train"]
X_test = data["X_test"]

y_test = data["y_test"]
y_pred = data["y_pred"]

accuracy = data["accuracy"]
roc_auc = data["roc_auc"]

cm = data["cm"]
report = data["report"]

y_probability = data["y_probability"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🎓 Student Dropout Prediction
        </div>

        <div class="hero-subtitle">
            Machine Learning based Student Risk Assessment System
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:10px;
    ">

        <div style="
            font-size:42px;
        ">
            🎓
        </div>

        <h2>
            Student Risk AI
        </h2>

        <p style="
            color:#9ca3af;
        ">
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
    <div style="
        text-align:center;
        color:#9ca3af;
        font-size:12px;
        line-height:1.6;
    ">

        <b>Machine Learning Model</b><br>

        Logistic Regression<br>

        StandardScaler<br>

        Binary Classification

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">System Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Overview of the dataset, machine learning model,
            and prediction system.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Total Students
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

                <div class="metric-label">
                    Input Features
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

                <div class="metric-label">
                    Model Accuracy
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

                <div class="metric-label">
                    ROC-AUC
                </div>

                <div class="metric-value">
                    {roc_auc:.3f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    # --------------------------------------------------------
    # TARGET CHART
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### Student Outcome Distribution"
        )

        target_chart = (
            df["target"]
            .map(
                {
                    0: "Not Dropout",
                    1: "Dropout"
                }
            )
            .value_counts()
            .reset_index()
        )

        target_chart.columns = [
            "Status",
            "Students"
        ]


        fig = px.bar(
            target_chart,
            x="Status",
            y="Students",
            text="Students",
            title="Dropout vs Not Dropout"
        )


        fig.update_traces(
            textposition="outside"
        )


        fig.update_layout(
            template="plotly_white",
            height=420,
            font=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            ),
            xaxis=dict(
                title_font=dict(
                    color="black"
                ),
                tickfont=dict(
                    color="black"
                )
            ),
            yaxis=dict(
                title_font=dict(
                    color="black"
                ),
                tickfont=dict(
                    color="black"
                )
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### Model Information"
        )

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🤖 Algorithm
                </div>

                <div class="info-text">
                    Logistic Regression
                </div>

            </div>

            <div class="info-card">

                <div class="info-title">
                    📏 Feature Scaling
                </div>

                <div class="info-text">
                    StandardScaler
                </div>

            </div>

            <div class="info-card">

                <div class="info-title">
                    📚 Training Data
                </div>

                <div class="info-text">
                    80% of the dataset
                </div>

            </div>

            <div class="info-card">

                <div class="info-title">
                    🧪 Testing Data
                </div>

                <div class="info-text">
                    20% of the dataset
                </div>

            </div>

            <div class="info-card">

                <div class="info-title">
                    🎯 Prediction
                </div>

                <div class="info-text">
                    Dropout / Not Dropout
                </div>

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
        <div class="section-description">
            Enter the student's information and the trained model
            will estimate the probability of dropout.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # INPUT DATA
    # ========================================================

    input_data = {}


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "👤 Personal & Demographic Information",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)


        with col1:

            input_data["Marital status"] = st.selectbox(
                "Marital Status",
                sorted(
                    df["Marital status"].unique()
                )
            )


        with col2:

            input_data["Application mode"] = st.selectbox(
                "Application Mode",
                sorted(
                    df["Application mode"].unique()
                )
            )


        with col3:

            input_data["Application order"] = st.number_input(
                "Application Order",
                min_value=int(
                    df["Application order"].min()
                ),
                max_value=int(
                    df["Application order"].max()
                ),
                value=int(
                    df["Application order"].median()
                )
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            input_data["Course"] = st.selectbox(
                "Course",
                sorted(
                    df["Course"].unique()
                )
            )


        with col2:

            input_data[
                "Daytime/evening attendance"
            ] = st.selectbox(
                "Attendance",
                sorted(
                    df[
                        "Daytime/evening attendance"
                    ].unique()
                )
            )


        with col3:

            input_data[
                "Previous qualification"
            ] = st.selectbox(
                "Previous Qualification",
                sorted(
                    df[
                        "Previous qualification"
                    ].unique()
                )
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            input_data[
                "Previous qualification (grade)"
            ] = st.number_input(
                "Previous Qualification Grade",
                min_value=float(
                    df[
                        "Previous qualification (grade)"
                    ].min()
                ),
                max_value=float(
                    df[
                        "Previous qualification (grade)"
                    ].max()
                ),
                value=float(
                    df[
                        "Previous qualification (grade)"
                    ].median()
                )
            )


        with col2:

            input_data[
                "Nacionality"
            ] = st.selectbox(
                "Nationality",
                sorted(
                    df["Nacionality"].unique()
                )
            )


        with col3:

            input_data[
                "Mother's qualification"
            ] = st.selectbox(
                "Mother's Qualification",
                sorted(
                    df[
                        "Mother's qualification"
                    ].unique()
                )
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            input_data[
                "Father's qualification"
            ] = st.selectbox(
                "Father's Qualification",
                sorted(
                    df[
                        "Father's qualification"
                    ].unique()
                )
            )


        with col2:

            input_data[
                "Mother's occupation"
            ] = st.selectbox(
                "Mother's Occupation",
                sorted(
                    df[
                        "Mother's occupation"
                    ].unique()
                )
            )


        with col3:

            input_data[
                "Father's occupation"
            ] = st.selectbox(
                "Father's Occupation",
                sorted(
                    df[
                        "Father's occupation"
                    ].unique()
                )
            )


    # --------------------------------------------------------
    # FINANCIAL / SOCIAL
    # --------------------------------------------------------

    with st.expander(
        "💰 Financial & Social Information",
        expanded=True
    ):

        col1, col2, col3 = st.columns(3)


        with col1:

            input_data["Displaced"] = st.selectbox(
                "Displaced",
                sorted(
                    df["Displaced"].unique()
                )
            )


        with col2:

            input_data[
                "Educational special needs"
            ] = st.selectbox(
                "Educational Special Needs",
                sorted(
                    df[
                        "Educational special needs"
                    ].unique()
                )
            )


        with col3:

            input_data["Debtor"] = st.selectbox(
                "Debtor",
                sorted(
                    df["Debtor"].unique()
                )
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            input_data[
                "Tuition fees up to date"
            ] = st.selectbox(
                "Tuition Fees Up To Date",
                sorted(
                    df[
                        "Tuition fees up to date"
                    ].unique()
                )
            )


        with col2:

            input_data["Gender"] = st.selectbox(
                "Gender",
                sorted(
                    df["Gender"].unique()
                )
            )


        with col3:

            input_data[
                "Scholarship holder"
            ] = st.selectbox(
                "Scholarship Holder",
                sorted(
                    df[
                        "Scholarship holder"
                    ].unique()
                )
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            input_data[
                "Age at enrollment"
            ] = st.number_input(
                "Age at Enrollment",
                min_value=int(
                    df[
                        "Age at enrollment"
                    ].min()
                ),
                max_value=int(
                    df[
                        "Age at enrollment"
                    ].max()
                ),
                value=int(
                    df[
                        "Age at enrollment"
                    ].median()
                )
            )


        with col2:

            input_data[
                "International"
            ] = st.selectbox(
                "International Student",
                sorted(
                    df["International"].unique()
                )
            )


        with col3:

            input_data["GDP"] = st.number_input(
                "GDP",
                min_value=float(
                    df["GDP"].min()
                ),
                max_value=float(
                    df["GDP"].max()
                ),
                value=float(
                    df["GDP"].median()
                )
            )


    # --------------------------------------------------------
    # FIRST SEMESTER
    # --------------------------------------------------------

    with st.expander(
        "📚 First Semester Academic Performance",
        expanded=True
    ):

        first_sem_features = [
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (without evaluations)"
        ]


        cols = st.columns(3)


        for i, feature in enumerate(
            first_sem_features
        ):

            with cols[i % 3]:

                input_data[feature] = st.number_input(
                    feature,
                    min_value=float(
                        df[feature].min()
                    ),
                    max_value=float(
                        df[feature].max()
                    ),
                    value=float(
                        df[feature].median()
                    )
                )


    # --------------------------------------------------------
    # SECOND SEMESTER
    # --------------------------------------------------------

    with st.expander(
        "📖 Second Semester Academic Performance",
        expanded=True
    ):

        second_sem_features = [
            "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (without evaluations)"
        ]


        cols = st.columns(3)


        for i, feature in enumerate(
            second_sem_features
        ):

            with cols[i % 3]:

                input_data[feature] = st.number_input(
                    feature,
                    min_value=float(
                        df[feature].min()
                    ),
                    max_value=float(
                        df[feature].max()
                    ),
                    value=float(
                        df[feature].median()
                    )
                )


    # --------------------------------------------------------
    # REMAINING FEATURES
    # --------------------------------------------------------

    remaining_features = [
        feature
        for feature in X.columns
        if feature not in input_data
    ]


    if remaining_features:

        with st.expander(
            "🏫 Additional Student Information",
            expanded=False
        ):

            cols = st.columns(3)


            for i, feature in enumerate(
                remaining_features
            ):

                with cols[i % 3]:

                    if pd.api.types.is_numeric_dtype(
                        df[feature]
                    ):

                        input_data[feature] = st.number_input(
                            feature,
                            min_value=float(
                                df[feature].min()
                            ),
                            max_value=float(
                                df[feature].max()
                            ),
                            value=float(
                                df[feature].median()
                            )
                        )

                    else:

                        input_data[feature] = st.selectbox(
                            feature,
                            sorted(
                                df[feature].unique()
                            )
                        )


    # ========================================================
    # PREDICT
    # ========================================================

    st.markdown("")

    predict = st.button(
        "🔮 Analyze Student Risk",
        type="primary",
        use_container_width=True
    )


    if predict:

        input_df = pd.DataFrame(
            [input_data]
        )


        # Exact feature order
        input_df = input_df[
            X.columns
        ]


        try:

            # ------------------------------------------------
            # SCALE
            # ------------------------------------------------

            input_scaled = scaler.transform(
                input_df
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                input_scaled
            )[0]


            probability = model.predict_proba(
                input_scaled
            )[0][1]


            dropout_percentage = (
                probability * 100
            )


            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            if probability < 0.30:

                risk = "LOW RISK"
                risk_description = (
                    "The model estimates a relatively low "
                    "probability of dropout."
                )

            elif probability < 0.60:

                risk = "MEDIUM RISK"
                risk_description = (
                    "The model estimates a moderate probability "
                    "of dropout. The student's progress should "
                    "be monitored."
                )

            else:

                risk = "HIGH RISK"
                risk_description = (
                    "The model estimates a high probability "
                    "of dropout. Early academic support may "
                    "be appropriate."
                )


            predicted_class = (
                "Dropout"
                if prediction == 1
                else "Not Dropout"
            )


            # =================================================
            # RESULT HEADER
            # =================================================

            st.markdown(
                """
                <div class="result-card">

                    <h2 style="
                        color:#111827;
                        margin-top:0;
                    ">
                        🎯 Student Risk Assessment
                    </h2>

                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------------------------------
            # RESULT METRICS
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Dropout Probability",
                    f"{dropout_percentage:.2f}%"
                )


            with col2:

                st.metric(
                    "Predicted Outcome",
                    predicted_class
                )


            with col3:

                st.metric(
                    "Risk Category",
                    risk
                )


            # -------------------------------------------------
            # GAUGE
            # -------------------------------------------------

            st.markdown(
                "### 📊 Dropout Probability"
            )


            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=dropout_percentage,
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
                            "tickfont": {
                                "color": "black"
                            }
                        },

                        "bar": {
                            "color": "#111827"
                        },

                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#e5e7eb"
                            },
                            {
                                "range": [30, 60],
                                "color": "#d1d5db"
                            },
                            {
                                "range": [60, 100],
                                "color": "#9ca3af"
                            }
                        ],

                        "threshold": {
                            "line": {
                                "color": "black",
                                "width": 4
                            },
                            "thickness": 0.75,
                            "value": dropout_percentage
                        }
                    }
                )
            )


            gauge.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20
                ),
                paper_bgcolor="white",
                font=dict(
                    color="black"
                )
            )


            st.plotly_chart(
                gauge,
                use_container_width=True
            )


            # -------------------------------------------------
            # RISK INTERPRETATION
            # -------------------------------------------------

            if probability < 0.30:

                st.success(
                    f"🟢 {risk}: {risk_description}"
                )

            elif probability < 0.60:

                st.warning(
                    f"🟡 {risk}: {risk_description}"
                )

            else:

                st.error(
                    f"🔴 {risk}: {risk_description}"
                )


            # -------------------------------------------------
            # PROBABILITY COMPARISON
            # -------------------------------------------------

            st.markdown(
                "### 📈 Prediction Probability Breakdown"
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
                text="Probability",
                title="Model Prediction Probabilities"
            )


            probability_fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )


            probability_fig.update_layout(
                template="plotly_white",
                height=400,
                yaxis=dict(
                    range=[
                        0,
                        100
                    ],
                    title="Probability (%)",
                    title_font=dict(
                        color="black"
                    ),
                    tickfont=dict(
                        color="black"
                    )
                ),
                xaxis=dict(
                    title="Outcome",
                    title_font=dict(
                        color="black"
                    ),
                    tickfont=dict(
                        color="black"
                    )
                ),
                font=dict(
                    color="black"
                ),
                title_font=dict(
                    color="black"
                )
            )


            st.plotly_chart(
                probability_fig,
                use_container_width=True
            )


            # -------------------------------------------------
            # IMPORTANT NOTE
            # -------------------------------------------------

            st.info(
                "This prediction is an estimated risk indicator "
                "generated by the Logistic Regression model. "
                "It should not be treated as a definitive "
                "decision about a student's future."
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
        <div class="section-description">
            Explore the dataset used to train the student
            dropout prediction model.
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Features",
            X.shape[1]
        )


    with col4:

        st.metric(
            "Duplicates",
            df.duplicated().sum()
        )


    st.markdown(
        "### Dataset Preview"
    )


    preview_df = df.copy()

    preview_df["target"] = (
        preview_df["target"]
        .replace(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
    )


    st.dataframe(
        preview_df.head(20),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.markdown(
        "### Missing Values"
    )


    missing_df = pd.DataFrame(
        {
            "Feature": df.columns,
            "Missing Values": [
                df[column].isnull().sum()
                for column in df.columns
            ]
        }
    )


    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    st.markdown(
        "### Data Types"
    )


    dtype_df = pd.DataFrame(
        {
            "Feature": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ]
        }
    )


    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPLORATORY ANALYSIS
# ============================================================

elif page == "📈 Exploratory Analysis":

    st.markdown(
        '<div class="section-title">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Interactive visual analysis of important student
            characteristics associated with dropout.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    st.markdown(
        "### 1. Student Outcome Distribution"
    )


    target_df = (
        df["target"]
        .map(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
        .value_counts()
        .reset_index()
    )


    target_df.columns = [
        "Status",
        "Students"
    ]


    fig = px.pie(
        target_df,
        names="Status",
        values="Students",
        title="Student Outcome Distribution",
        hole=0.45
    )


    fig.update_layout(
        template="plotly_white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # FIRST SEMESTER GRADE
    # ========================================================

    st.markdown(
        "### 2. First Semester Grade vs Dropout"
    )


    grade_df = df.copy()

    grade_df["Status"] = (
        grade_df["target"]
        .map(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
    )


    fig = px.box(
        grade_df,
        x="Status",
        y="Curricular units 1st sem (grade)",
        color="Status",
        title="First Semester Grade Distribution"
    )


    fig.update_layout(
        template="plotly_white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            title="Student Status",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        ),
        yaxis=dict(
            title="First Semester Grade",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        ),
        showlegend=False
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # TUITION FEES
    # ========================================================

    st.markdown(
        "### 3. Tuition Fees Status vs Dropout"
    )


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


    tuition_df[
        "Tuition Status"
    ] = tuition_df[
        "Tuition fees up to date"
    ].map(
        {
            0: "Not Up to Date",
            1: "Up to Date"
        }
    )


    fig = px.bar(
        tuition_df,
        x="Tuition Status",
        y="Dropout Rate",
        text="Dropout Rate",
        title="Dropout Rate by Tuition Fee Status"
    )


    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig.update_layout(
        template="plotly_white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        yaxis=dict(
            title="Dropout Rate (%)",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        ),
        xaxis=dict(
            title="Tuition Status",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # SECOND SEMESTER
    # ========================================================

    st.markdown(
        "### 4. Second Semester Grade vs Dropout"
    )


    fig = px.box(
        grade_df,
        x="Status",
        y="Curricular units 2nd sem (grade)",
        color="Status",
        title="Second Semester Grade Distribution"
    )


    fig.update_layout(
        template="plotly_white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            title="Student Status",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        ),
        yaxis=dict(
            title="Second Semester Grade",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        ),
        showlegend=False
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
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Evaluation of the Logistic Regression model using
            unseen test data.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )


    with col3:

        st.metric(
            "Training Samples",
            len(X_train)
        )


    with col4:

        st.metric(
            "Testing Samples",
            len(X_test)
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown(
        "### Confusion Matrix"
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual: Not Dropout",
            "Actual: Dropout"
        ],
        columns=[
            "Predicted: Not Dropout",
            "Predicted: Dropout"
        ]
    )


    fig = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix",
        aspect="auto"
    )


    fig.update_layout(
        template="plotly_white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown(
        "### Classification Report"
    )


    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
    )


    report_df = report_df.rename(
        columns={
            "index": "Class"
        }
    )


    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}"
            },
            na_rep="-"
        ),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # TEST PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        "### Test Set Prediction Distribution"
    )


    prediction_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": y_pred
        }
    )


    prediction_df["Actual"] = (
        prediction_df["Actual"]
        .map(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
    )


    prediction_df["Predicted"] = (
        prediction_df["Predicted"]
        .map(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
    )


    comparison_df = (
        prediction_df
        .value_counts()
        .reset_index(
            name="Students"
        )
    )


    fig = px.bar(
        comparison_df,
        x="Actual",
        y="Students",
        color="Predicted",
        barmode="group",
        text="Students",
        title="Actual vs Predicted Student Outcomes"
    )


    fig.update_traces(
        textposition="outside"
    )


    fig.update_layout(
        template="plotly_white",
        font=dict(
            color="black"
        ),
        title_font=dict(
            color="black"
        ),
        xaxis=dict(
            title="Actual Outcome",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        ),
        yaxis=dict(
            title="Number of Students",
            tickfont=dict(
                color="black"
            ),
            title_font=dict(
                color="black"
            )
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>Student Dropout Prediction System</b><br>

        Logistic Regression • StandardScaler • Plotly • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
