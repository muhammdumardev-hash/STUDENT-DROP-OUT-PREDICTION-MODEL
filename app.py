import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    /* --------------------------------------------------------
       MAIN APPLICATION
    -------------------------------------------------------- */

    .stApp {
        background-color: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }


    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .main-header {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 50%,
            #111827 100%
        );

        padding: 32px 35px;
        border-radius: 18px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .main-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .main-subtitle {
        color: #d1d5db;
        font-size: 16px;
        margin-bottom: 0;
    }


    /* --------------------------------------------------------
       SECTION HEADINGS
    -------------------------------------------------------- */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 8px;
    }

    .section-description {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }


    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    .metric-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        text-align: center;
    }

    .metric-title {
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


    /* --------------------------------------------------------
       PREDICTION RESULT
    -------------------------------------------------------- */

    .prediction-card {
        background-color: white;
        border-radius: 18px;
        padding: 30px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 7px 25px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .prediction-title {
        color: #111827;
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .prediction-probability {
        color: #111827;
        font-size: 48px;
        font-weight: 900;
    }

    .risk-text {
        color: #111827;
        font-size: 24px;
        font-weight: 800;
        margin-top: 8px;
    }


    /* --------------------------------------------------------
       INFO BOX
    -------------------------------------------------------- */

    .info-box {
        background-color: white;
        border-left: 5px solid #111827;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        border-top: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
    }

    .info-box-title {
        color: #111827;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .info-box-text {
        color: #4b5563;
        font-size: 14px;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-weight: 700;
        border: none;
    }


    /* --------------------------------------------------------
       INPUT LABELS
    -------------------------------------------------------- */

    label {
        font-weight: 600 !important;
    }


    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
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

    df = pd.read_csv("datasett.csv")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

@st.cache_resource
def train_model():

    df = load_dataset().copy()

    # Encode target
    df["target"] = df["target"].replace({
        "Dropout": 1,
        "Graduate": 0,
        "Enrolled": 0
    })

    df["target"] = df["target"].astype(int)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
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

    return (
        df,
        X,
        y,
        model,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_prob,
        accuracy,
        roc_auc,
        cm,
        report
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    (
        df,
        X,
        y,
        model,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_prob,
        accuracy,
        roc_auc,
        cm,
        report
    ) = train_model()

except FileNotFoundError:

    st.error(
        "datasett.csv was not found. "
        "Please place datasett.csv in the same folder as app.py."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <div class="main-title">
            🎓 Student Dropout Prediction
        </div>

        <div class="main-subtitle">
            Machine Learning Application for Student Risk Assessment
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
    <h2 style="text-align:center;">
        🎓 Student Risk AI
    </h2>

    <p style="text-align:center;color:#d1d5db;">
        Logistic Regression
    </p>
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
    <div style="text-align:center;color:#9ca3af;font-size:12px;">
        Student Dropout Prediction<br>
        Internship Phase 7
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
        <div class="section-description">
            Overview of the student dropout prediction machine learning system.
        </div>
        """,
        unsafe_allow_html=True
    )


    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
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
                <div class="metric-title">
                    Features
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
                    Accuracy
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


    st.markdown("")


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            '<div class="section-title">Target Distribution</div>',
            unsafe_allow_html=True
        )

        target_counts = df["target"].replace({
            0: "Not Dropout",
            1: "Dropout"
        }).value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        target_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Dropout vs Not Dropout",
            color="black"
        )

        ax.set_xlabel(
            "Student Status",
            color="black"
        )

        ax.set_ylabel(
            "Number of Students",
            color="black"
        )

        ax.tick_params(
            axis="both",
            colors="black"
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    with col2:

        st.markdown(
            '<div class="section-title">Model Information</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box">

                <div class="info-box-title">
                    Algorithm
                </div>

                <div class="info-box-text">
                    Logistic Regression
                </div>

            </div>

            <div class="info-box">

                <div class="info-box-title">
                    Feature Scaling
                </div>

                <div class="info-box-text">
                    StandardScaler
                </div>

            </div>

            <div class="info-box">

                <div class="info-box-title">
                    Training / Testing Split
                </div>

                <div class="info-box-text">
                    80% Training / 20% Testing
                </div>

            </div>

            <div class="info-box">

                <div class="info-box-title">
                    Prediction Type
                </div>

                <div class="info-box-text">
                    Binary Classification
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
            Enter student information below to estimate the probability
            of student dropout.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = {}


    # ========================================================
    # PERSONAL / DEMOGRAPHIC INFORMATION
    # ========================================================

    st.markdown(
        "### 👤 Personal & Demographic Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Marital status"] = st.selectbox(
            "Marital Status",
            options=sorted(
                df["Marital status"].unique()
            )
        )


    with col2:

        input_data["Application mode"] = st.selectbox(
            "Application Mode",
            options=sorted(
                df["Application mode"].unique()
            )
        )


    with col3:

        input_data["Application order"] = st.number_input(
            "Application Order",
            min_value=int(df["Application order"].min()),
            max_value=int(df["Application order"].max()),
            value=int(df["Application order"].median()),
            step=1
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Course"] = st.selectbox(
            "Course",
            options=sorted(
                df["Course"].unique()
            )
        )


    with col2:

        input_data["Daytime/evening attendance"] = st.selectbox(
            "Attendance",
            options=sorted(
                df["Daytime/evening attendance"].unique()
            )
        )


    with col3:

        input_data["Previous qualification"] = st.selectbox(
            "Previous Qualification",
            options=sorted(
                df["Previous qualification"].unique()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Previous qualification (grade)"] = st.number_input(
            "Previous Qualification Grade",
            min_value=float(
                df["Previous qualification (grade)"].min()
            ),
            max_value=float(
                df["Previous qualification (grade)"].max()
            ),
            value=float(
                df["Previous qualification (grade)"].median()
            )
        )


    with col2:

        input_data["Nacionality"] = st.selectbox(
            "Nationality",
            options=sorted(
                df["Nacionality"].unique()
            )
        )


    with col3:

        input_data["Mother's qualification"] = st.selectbox(
            "Mother's Qualification",
            options=sorted(
                df["Mother's qualification"].unique()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Father's qualification"] = st.selectbox(
            "Father's Qualification",
            options=sorted(
                df["Father's qualification"].unique()
            )
        )


    with col2:

        input_data["Mother's occupation"] = st.selectbox(
            "Mother's Occupation",
            options=sorted(
                df["Mother's occupation"].unique()
            )
        )


    with col3:

        input_data["Father's occupation"] = st.selectbox(
            "Father's Occupation",
            options=sorted(
                df["Father's occupation"].unique()
            )
        )


    # ========================================================
    # FINANCIAL / SOCIAL INFORMATION
    # ========================================================

    st.markdown(
        "### 💰 Financial & Social Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Displaced"] = st.selectbox(
            "Displaced",
            options=sorted(
                df["Displaced"].unique()
            )
        )


    with col2:

        input_data["Educational special needs"] = st.selectbox(
            "Educational Special Needs",
            options=sorted(
                df["Educational special needs"].unique()
            )
        )


    with col3:

        input_data["Debtor"] = st.selectbox(
            "Debtor",
            options=sorted(
                df["Debtor"].unique()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Tuition fees up to date"] = st.selectbox(
            "Tuition Fees Up To Date",
            options=sorted(
                df["Tuition fees up to date"].unique()
            )
        )


    with col2:

        input_data["Gender"] = st.selectbox(
            "Gender",
            options=sorted(
                df["Gender"].unique()
            )
        )


    with col3:

        input_data["Scholarship holder"] = st.selectbox(
            "Scholarship Holder",
            options=sorted(
                df["Scholarship holder"].unique()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data["Age at enrollment"] = st.number_input(
            "Age at Enrollment",
            min_value=int(
                df["Age at enrollment"].min()
            ),
            max_value=int(
                df["Age at enrollment"].max()
            ),
            value=int(
                df["Age at enrollment"].median()
            )
        )


    with col2:

        input_data["International"] = st.selectbox(
            "International Student",
            options=sorted(
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


    # ========================================================
    # FIRST SEMESTER
    # ========================================================

    st.markdown(
        "### 📚 First Semester Academic Performance"
    )


    first_sem_features = [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)"
    ]


    cols = st.columns(3)


    for i, feature in enumerate(first_sem_features):

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


    # ========================================================
    # SECOND SEMESTER
    # ========================================================

    st.markdown(
        "### 📖 Second Semester Academic Performance"
    )


    second_sem_features = [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)"
    ]


    cols = st.columns(3)


    for i, feature in enumerate(second_sem_features):

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


    # ========================================================
    # OTHER INFORMATION
    # ========================================================

    st.markdown(
        "### 🏫 Additional Information"
    )


    remaining_features = [
        feature
        for feature in X.columns
        if feature not in input_data
    ]


    if len(remaining_features) > 0:

        cols = st.columns(3)

        for i, feature in enumerate(remaining_features):

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
                        options=sorted(
                            df[feature].unique()
                        )
                    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.markdown("")

    predict_button = st.button(
        "🔮 Predict Student Risk",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # Ensure exact feature order
        input_df = input_df[
            X.columns
        ]


        try:

            # ------------------------------------------------
            # SCALE INPUT
            # ------------------------------------------------

            input_scaled = scaler.transform(
                input_df
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                input_scaled
            )[0]


            probability = model.predict_proba(
                input_scaled
            )[0][1]


            probability_percentage = (
                probability * 100
            )


            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if probability < 0.30:

                risk_level = "LOW RISK"

                risk_message = (
                    "The student currently shows a low "
                    "estimated probability of dropout."
                )

            elif probability < 0.60:

                risk_level = "MEDIUM RISK"

                risk_message = (
                    "The student shows a moderate estimated "
                    "probability of dropout. Academic progress "
                    "should be monitored."
                )

            else:

                risk_level = "HIGH RISK"

                risk_message = (
                    "The student shows a high estimated "
                    "probability of dropout. Early academic "
                    "support and intervention may be beneficial."
                )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                """
                <div class="prediction-card">
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                "## 🎯 Prediction Result"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Dropout Probability",
                    f"{probability_percentage:.2f}%"
                )


            with col2:

                if prediction == 1:

                    prediction_text = "Dropout"

                else:

                    prediction_text = "Not Dropout"


                st.metric(
                    "Predicted Class",
                    prediction_text
                )


            with col3:

                st.metric(
                    "Risk Level",
                    risk_level
                )


            # Probability bar

            st.markdown(
                "### Dropout Probability"
            )

            st.progress(
                float(probability)
            )


            st.markdown(
                f"""
                <div class="prediction-probability">
                    {probability_percentage:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="risk-text">
                    {risk_level}
                </div>
                """,
                unsafe_allow_html=True
            )


            st.info(
                risk_message
            )


            # ------------------------------------------------
            # PROBABILITY BREAKDOWN
            # ------------------------------------------------

            st.markdown(
                "### 📊 Probability Breakdown"
            )


            probability_df = pd.DataFrame(
                {
                    "Outcome": [
                        "Not Dropout",
                        "Dropout"
                    ],
                    "Probability": [
                        1 - probability,
                        probability
                    ]
                }
            )


            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            )


            st.dataframe(
                probability_df[
                    [
                        "Outcome",
                        "Probability (%)"
                    ]
                ].style.format(
                    {
                        "Probability (%)": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # MODEL INTERPRETATION
            # ------------------------------------------------

            st.markdown(
                "### 🧠 Interpretation"
            )


            if probability < 0.30:

                st.success(
                    "Low Risk: The model estimates that this "
                    "student has a relatively low likelihood of "
                    "dropping out."
                )

            elif probability < 0.60:

                st.warning(
                    "Medium Risk: The model identifies this "
                    "student as requiring monitoring because "
                    "the estimated dropout probability is moderate."
                )

            else:

                st.error(
                    "High Risk: The model estimates a relatively "
                    "high dropout probability. Academic monitoring "
                    "and early intervention should be considered."
                )


            st.markdown(
                """
                <div class="info-box">

                    <div class="info-box-title">
                        Important Note
                    </div>

                    <div class="info-box-text">
                        This prediction is generated by a machine
                        learning model and should be treated as an
                        estimated risk indicator rather than a
                        definitive decision about a student.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
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
        <div class="section-description">
            Explore the dataset used for training the dropout
            prediction model.
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Features",
            X.shape[1]
        )


    with col3:

        st.metric(
            "Target Classes",
            2
        )


    st.markdown(
        "### Dataset Preview"
    )

    display_df = df.copy()

    display_df["target"] = display_df[
        "target"
    ].replace(
        {
            0: "Not Dropout",
            1: "Dropout"
        }
    )

    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### Dataset Shape"
    )

    st.write(
        f"Rows: **{df.shape[0]}**"
    )

    st.write(
        f"Columns: **{df.shape[1]}**"
    )


    st.markdown(
        "### Missing Values"
    )

    missing_df = pd.DataFrame(
        {
            "Column": df.columns,
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


    st.markdown(
        "### Duplicate Records"
    )

    st.write(
        f"Total duplicate records: "
        f"**{df.duplicated().sum()}**"
    )


    st.markdown(
        "### Data Types"
    )

    dtype_df = pd.DataFrame(
        {
            "Column": df.columns,
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
            Visual analysis of important factors related to student dropout.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        "### 1. Dropout Distribution"
    )


    target_counts = df["target"].value_counts()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    target_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Dropout vs Not Dropout",
        color="black"
    )

    ax.set_xlabel(
        "Student Status",
        color="black"
    )

    ax.set_ylabel(
        "Number of Students",
        color="black"
    )

    ax.set_xticklabels(
        [
            "Not Dropout",
            "Dropout"
        ],
        rotation=0
    )

    ax.tick_params(
        axis="both",
        colors="black"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # FIRST SEMESTER GRADE
    # --------------------------------------------------------

    st.markdown(
        "### 2. First Semester Grade vs Dropout"
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    df.boxplot(
        column="Curricular units 1st sem (grade)",
        by="target",
        ax=ax
    )


    ax.set_title(
        "1st Semester Grade vs Dropout",
        color="black"
    )

    ax.set_xlabel(
        "Student Status",
        color="black"
    )

    ax.set_ylabel(
        "1st Semester Grade",
        color="black"
    )

    ax.set_xticklabels(
        [
            "Not Dropout",
            "Dropout"
        ]
    )

    ax.tick_params(
        axis="both",
        colors="black"
    )


    plt.suptitle("")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # TUITION FEES
    # --------------------------------------------------------

    st.markdown(
        "### 3. Tuition Fees Status vs Dropout"
    )


    tuition_rate = df.groupby(
        "Tuition fees up to date"
    )["target"].mean()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    tuition_rate.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Tuition Fees Status vs Dropout",
        color="black"
    )

    ax.set_xlabel(
        "Tuition Fees Up to Date",
        color="black"
    )

    ax.set_ylabel(
        "Dropout Rate",
        color="black"
    )

    ax.tick_params(
        axis="both",
        colors="black"
    )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # SECOND SEMESTER
    # --------------------------------------------------------

    st.markdown(
        "### 4. Second Semester Grade vs Dropout"
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    df.boxplot(
        column="Curricular units 2nd sem (grade)",
        by="target",
        ax=ax
    )


    ax.set_title(
        "2nd Semester Grade vs Dropout",
        color="black"
    )

    ax.set_xlabel(
        "Student Status",
        color="black"
    )

    ax.set_ylabel(
        "2nd Semester Grade",
        color="black"
    )

    ax.set_xticklabels(
        [
            "Not Dropout",
            "Dropout"
        ]
    )

    ax.tick_params(
        axis="both",
        colors="black"
    )


    plt.suptitle("")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


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
            Evaluation results of the Logistic Regression model
            on unseen test data.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PERFORMANCE METRICS
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


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.imshow(
        cm
    )


    ax.set_title(
        "Confusion Matrix",
        color="black"
    )


    ax.set_xlabel(
        "Predicted Label",
        color="black"
    )


    ax.set_ylabel(
        "Actual Label",
        color="black"
    )


    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )


    ax.set_xticklabels(
        [
            "Not Dropout",
            "Dropout"
        ],
        color="black"
    )


    ax.set_yticklabels(
        [
            "Not Dropout",
            "Dropout"
        ],
        color="black"
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold"
            )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown(
        "### Classification Report"
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    st.dataframe(
        report_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PREDICTION SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "### Test Set Prediction Summary"
    )


    prediction_summary = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": y_pred,
            "Dropout Probability": y_prob
        }
    )


    prediction_summary["Actual"] = (
        prediction_summary["Actual"]
        .replace(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
    )


    prediction_summary["Predicted"] = (
        prediction_summary["Predicted"]
        .replace(
            {
                0: "Not Dropout",
                1: "Dropout"
            }
        )
    )


    prediction_summary["Dropout Probability"] = (
        prediction_summary["Dropout Probability"] * 100
    )


    st.dataframe(
        prediction_summary.head(20).style.format(
            {
                "Dropout Probability": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#6b7280;
        padding:20px;
        border-top:1px solid #e5e7eb;
    ">

        <strong>Student Dropout Prediction System</strong><br>

        Logistic Regression • StandardScaler • Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
