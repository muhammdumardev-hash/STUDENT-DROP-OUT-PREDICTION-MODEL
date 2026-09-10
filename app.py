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

    .section-box {
        background: #111827 !important;
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 15px;
    }

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

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label p {
        color: #ffffff !important;
    }

    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #e0e0e0 !important;
        font-size: 0.90rem !important;
    }

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

    button {
        font-weight: 700 !important;
    }

    button p {
        font-weight: 700 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        overflow: hidden;
        background: #ffffff !important;
    }

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

    hr {
        border-color: #334155 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #334155 !important;
        border-radius: 14px !important;
    }

    div[data-baseweb="input"] {
        background: #ffffff !important;
        border-radius: 8px !important;
    }

    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: #ffffff !important;
    }

    [data-testid="stCheckbox"] label p,
    [data-testid="stRadio"] label p {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #ffffff !important;
    }

    .risk-info-card {
        border-radius: 12px;
        padding: 16px;
        min-height: 125px;
    }

    .risk-info-card h4 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.10rem;
        font-weight: 800;
    }

    .risk-info-card p {
        margin: 0;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    .risk-info-low {
        background: #052e16 !important;
        border: 1px solid #166534;
    }

    .risk-info-medium {
        background: #422006 !important;
        border: 1px solid #a16207;
    }

    .risk-info-high {
        background: #450a0a !important;
        border: 1px solid #991b1b;
    }

    .risk-info-low h4 {
        color: #86efac !important;
    }

    .risk-info-low p {
        color: #dcfce7 !important;
    }

    .risk-info-medium h4 {
        color: #fde047 !important;
    }

    .risk-info-medium p {
        color: #fef3c7 !important;
    }

    .risk-info-high h4 {
        color: #fca5a5 !important;
    }

    .risk-info-high p {
        color: #fee2e2 !important;
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
# CATEGORICAL CODE → HUMAN-READABLE LABEL MAPPINGS
#
# Source: Official UCI "Predict Students' Dropout and Academic
# Success" dataset documentation (Realinho et al., 2021).
#
# These dictionaries are used ONLY to control what is DISPLAYED
# in the Predict Risk dropdowns. The numeric code the user picks
# is always what gets stored in `inputs` and passed to the model.
#
#   Nursing (displayed)  →  9500 (stored + sent to model)
# ============================================================

MARITAL_STATUS_LABELS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated",
}

APPLICATION_MODE_LABELS = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

COURSE_LABELS = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}

DAYTIME_EVENING_LABELS = {
    1: "Daytime",
    0: "Evening",
}

PREVIOUS_QUALIFICATION_LABELS = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
}

NACIONALITY_LABELS = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

MOTHER_QUALIFICATION_LABELS = {
    1: "Secondary Education - 12th Year or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year - Not Completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)",
}

FATHER_QUALIFICATION_LABELS = {
    1: "Secondary Education - 12th Year or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    13: "2nd year complementary high school course",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    20: "Complementary High School Course",
    22: "Technical-professional course",
    25: "Complementary High School Course - not concluded",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year - Not Completed",
    30: "8th year of schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)",
}

MOTHER_OCCUPATION_LABELS = {
    0: "Student",
    1: "Legislative/Executive Directors & Managers",
    2: "Intellectual & Scientific Specialists",
    3: "Intermediate Level Technicians",
    4: "Administrative staff",
    5: "Personal Services, Security & Sellers",
    6: "Farmers & Skilled Agriculture/Fishery Workers",
    7: "Skilled Industry/Construction Workers",
    8: "Installation & Machine Operators",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "(blank)",
    122: "Health professionals",
    123: "Teachers",
    125: "ICT Specialists",
    131: "Science/Engineering Technicians (Intermediate)",
    132: "Health Technicians (Intermediate)",
    134: "Legal/Social/Sports/Cultural Technicians (Intermediate)",
    141: "Office Workers & Secretaries",
    143: "Data/Accounting/Finance Operators",
    144: "Other Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers",
    171: "Skilled Construction Workers",
    173: "Skilled Printing/Precision/Jewelry Workers",
    175: "Food/Wood/Clothing Industry Workers",
    191: "Cleaning Workers",
    192: "Unskilled Agriculture/Fishery Workers",
    193: "Unskilled Construction/Manufacturing Workers",
    194: "Meal Preparation Assistants",
}

FATHER_OCCUPATION_LABELS = {
    0: "Student",
    1: "Legislative/Executive Directors & Managers",
    2: "Intellectual & Scientific Specialists",
    3: "Intermediate Level Technicians",
    4: "Administrative staff",
    5: "Personal Services, Security & Sellers",
    6: "Farmers & Skilled Agriculture/Fishery Workers",
    7: "Skilled Industry/Construction Workers",
    8: "Installation & Machine Operators",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "(blank)",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces Personnel",
    112: "Admin. & Commercial Services Directors",
    114: "Hotel/Catering/Trade Services Directors",
    121: "Physical Sciences/Engineering Specialists",
    122: "Health professionals",
    123: "Teachers",
    124: "Finance/Accounting/Admin Specialists",
    131: "Science/Engineering Technicians (Intermediate)",
    132: "Health Technicians (Intermediate)",
    134: "Legal/Social/Sports/Cultural Technicians (Intermediate)",
    135: "ICT Technicians",
    141: "Office Workers & Secretaries",
    143: "Data/Accounting/Finance Operators",
    144: "Other Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers",
    154: "Protection & Security Personnel",
    161: "Market-oriented Farmers & Skilled Agriculture Workers",
    163: "Subsistence Farmers/Fishermen/Hunters",
    171: "Skilled Construction Workers",
    172: "Skilled Metallurgy/Metalworking Workers",
    174: "Skilled Electricity/Electronics Workers",
    175: "Food/Wood/Clothing Industry Workers",
    181: "Fixed Plant & Machine Operators",
    182: "Assembly Workers",
    183: "Vehicle Drivers & Mobile Equipment Operators",
    192: "Unskilled Agriculture/Fishery Workers",
    193: "Unskilled Construction/Manufacturing Workers",
    194: "Meal Preparation Assistants",
    195: "Street Vendors & Street Service Providers",
}

GENDER_LABELS = {
    1: "Male",
    0: "Female",
}

# Maps a feature column name to its code → label dictionary.
# Any column NOT in this dict falls back to the existing
# behaviour (raw numeric input / plain selectbox).
CATEGORY_LABELS = {
    "Marital status": MARITAL_STATUS_LABELS,
    "Application mode": APPLICATION_MODE_LABELS,
    "Course": COURSE_LABELS,
    "Daytime/evening attendance": DAYTIME_EVENING_LABELS,
    "Previous qualification": PREVIOUS_QUALIFICATION_LABELS,
    "Nacionality": NACIONALITY_LABELS,
    "Mother's qualification": MOTHER_QUALIFICATION_LABELS,
    "Father's qualification": FATHER_QUALIFICATION_LABELS,
    "Mother's occupation": MOTHER_OCCUPATION_LABELS,
    "Father's occupation": FATHER_OCCUPATION_LABELS,
    "Gender": GENDER_LABELS,
}


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
        '<p class="main-title">🎓 Student Dropout Prediction</p>',
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
        '<p class="main-title">📊 Exploratory Data Analysis</p>',
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

    col1, col2 = st.columns(2)

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
            than **Dropout** students.

            This difference is important because the model needs
            to correctly identify the smaller but important
            dropout group.
            """
        )

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

    col3, col4 = st.columns(2)

    with col3:

        if "Curricular units 1st sem (grade)" in df.columns:

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

    with col4:

        if "Curricular units 2nd sem (grade)" in df.columns:

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

    st.write("")

    with st.container(border=True):

        st.subheader(
            "💡 Key Observations"
        )

        st.markdown(
            """
            **1. Student Distribution**

            Most students in the dataset are classified as
            **Not Dropout**, while a smaller group is classified
            as **Dropout**.

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
        '<p class="main-title">🤖 Model Performance</p>',
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
        '<p class="main-title">🔮 Student Risk Prediction</p>',
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

    y_test_raw = results[
        "y_test"
    ]

    # ========================================================
    # IMPORTANT:
    # X_test_raw contains the exact feature values from the
    # cleaned dataset, while raw_df contains the original
    # values directly loaded from datasett.csv.
    #
    # The original pandas index is preserved during
    # train_test_split, so sample_index can be used to retrieve
    # the exact corresponding dataset row from raw_df.
    # ========================================================

    if "sample_index" not in st.session_state:
        st.session_state.sample_index = None

    if "loaded_sample_index" not in st.session_state:
        st.session_state.loaded_sample_index = None

    # ========================================================
    # QUICK TEST CASE
    # ========================================================

    with st.container(border=True):

        st.subheader(
            "🎲 Quick Test Case"
        )

        st.caption(
            "Load a student from the held-out test set. "
            "The model receives only the 36 feature values. "
            "The student's target is kept completely separate "
            "and is used only after prediction for comparison."
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

            st.session_state.loaded_sample_index = None

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

            st.rerun()

    with st.expander(
        "🔐 How the Quick Test works"
    ):

        st.markdown(
            """
            **Dataset**

            `datasett.csv`

            ↓

            **Features (X)**  
            36 student feature columns

            **Target (y)**  
            Dropout / Not Dropout

            ↓

            **Train/Test Split**

            - 80% → Training data
            - 20% → Held-out test data

            ↓

            **Model**

            The Logistic Regression model receives only the
            36 feature values.

            **The `target` column is NOT included in the model input.**

            ↓

            **Prediction**

            The model calculates the dropout probability from
            the student's feature values.

            ↓

            **Actual Outcome**

            For a test-set student, the original target from
            `y_test` is shown separately only after prediction
            so the prediction can be evaluated.

            **Therefore, the model does not simply read the answer
            from the dataset.**
            """
        )

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
            "The model does NOT receive the student's target "
            "during prediction."
        )

    if sample_index is not None:

        selected_student = (
            X_test_raw.loc[
                sample_index
            ]
        )

    else:

        selected_student = None

    # ========================================================
    # LOAD TEST STUDENT VALUES INTO INPUT FIELDS
    # ========================================================

    if (
        selected_student is not None
        and st.session_state.get(
            "loaded_sample_index"
        ) != sample_index
    ):

        for col_name in feature_names:

            value = selected_student[
                col_name
            ]

            if isinstance(
                value,
                np.generic
            ):

                value = value.item()

            st.session_state[
                f"input_{col_name}"
            ] = value

        st.session_state.loaded_sample_index = (
            sample_index
        )

    if selected_student is not None:
        defaults = selected_student
    else:
        defaults = None

    # ========================================================
    # VERIFY LOADED TEST STUDENT VALUES
    #
    # IMPORTANT FIX:
    #
    # Original Dataset Value is retrieved directly from
    # raw_df, which was loaded from datasett.csv.
    #
    # It is NOT retrieved from the input fields.
    #
    # Loaded Input Value is retrieved from session_state,
    # which represents the current value inside the form.
    #
    # This allows us to detect if a user changes any value
    # after loading a test student.
    # ========================================================

    if selected_student is not None:

        with st.expander(
            "🔎 Verify Loaded Test Student Values"
        ):

            verification_rows = []

            # ------------------------------------------------
            # GET THE EXACT ORIGINAL DATASET ROW
            # ------------------------------------------------
            #
            # train_test_split preserves the original pandas
            # index. Therefore sample_index points to the same
            # record in raw_df that came from datasett.csv.
            #
            # We explicitly remove target because this section
            # verifies feature values only.
            # ------------------------------------------------

            original_dataset_row = (
                raw_df.loc[
                    sample_index,
                    feature_names
                ]
            )

            for col_name in feature_names:

                # --------------------------------------------
                # ORIGINAL VALUE
                # --------------------------------------------
                #
                # This comes directly from datasett.csv.
                # --------------------------------------------

                original_value = (
                    original_dataset_row[
                        col_name
                    ]
                )

                # --------------------------------------------
                # CURRENT INPUT VALUE
                # --------------------------------------------
                #
                # This comes from the Streamlit form/session
                # state.
                # --------------------------------------------

                loaded_value = (
                    st.session_state.get(
                        f"input_{col_name}",
                        selected_student[col_name]
                    )
                )

                # --------------------------------------------
                # Convert NumPy scalar values to normal
                # Python values for clean dataframe display.
                # --------------------------------------------

                if isinstance(
                    original_value,
                    np.generic
                ):

                    original_value = (
                        original_value.item()
                    )

                if isinstance(
                    loaded_value,
                    np.generic
                ):

                    loaded_value = (
                        loaded_value.item()
                    )

                # --------------------------------------------
                # COMPARE VALUES
                # --------------------------------------------

                try:

                    original_numeric = float(
                        original_value
                    )

                    loaded_numeric = float(
                        loaded_value
                    )

                    is_match = np.isclose(
                        original_numeric,
                        loaded_numeric,
                        rtol=1e-5,
                        atol=1e-8
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    is_match = (
                        str(original_value).strip()
                        ==
                        str(loaded_value).strip()
                    )

                verification_rows.append(
                    {
                        "Feature": col_name,
                        "Original Dataset Value": original_value,
                        "Loaded Input Value": loaded_value,
                        "Match": (
                            "✅"
                            if is_match
                            else "❌"
                        ),
                    }
                )

            verification_df = pd.DataFrame(
                verification_rows
            )

            st.dataframe(
                verification_df,
                use_container_width=True,
                hide_index=True
            )

            mismatch_count = int(
                (
                    verification_df["Match"]
                    == "❌"
                ).sum()
            )

            # ------------------------------------------------
            # ALL VALUES MATCH
            # ------------------------------------------------

            if mismatch_count == 0:

                st.success(
                    f"✅ All {len(feature_names)} "
                    "test-student feature values "
                    "exactly match the original "
                    "dataset record."
                )

            # ------------------------------------------------
            # SOME VALUES CHANGED
            # ------------------------------------------------

            else:

                st.warning(
                    f"⚠️ {mismatch_count} feature "
                    "value(s) have been changed "
                    "from the original dataset values."
                )

                st.caption(
                    "Original Dataset Value comes directly "
                    "from datasett.csv. Loaded Input Value "
                    "shows the current value in the form."
                )

    # ========================================================
    # PREDICTION FORM
    # ========================================================

    with st.form(
        "prediction_form"
    ):

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

                series = df[
                    col_name
                ]

                clean_series = (
                    series
                    .dropna()
                )

                if defaults is not None:

                    default_value = defaults[
                        col_name
                    ]

                else:

                    # Codes (Course, Marital status, etc.) are
                    # stored as numbers, but a median of codes
                    # is meaningless — use the most common code
                    # (mode) instead, same as other categoricals.
                    if (
                        col_name in CATEGORY_LABELS
                        or not pd.api.types.is_numeric_dtype(
                            series
                        )
                    ):

                        mode_values = (
                            series
                            .mode()
                        )

                        if len(mode_values) > 0:

                            default_value = (
                                mode_values.iloc[0]
                            )

                        else:

                            default_value = (
                                clean_series.iloc[0]
                            )

                    else:

                        default_value = (
                            series.median()
                        )

                if isinstance(
                    default_value,
                    np.generic
                ):

                    default_value = (
                        default_value.item()
                    )

                with input_cols[
                    i % 3
                ]:

                    binary_columns = [
                        "Debtor",
                        "Tuition fees up to date",
                        "Scholarship holder",
                        "Displaced",
                        "Educational special needs",
                        "International",
                    ]

                    if col_name in binary_columns:

                        try:

                            default_binary = int(
                                float(
                                    default_value
                                )
                            )

                        except Exception:

                            default_binary = 0

                        default_binary = (
                            1
                            if default_binary == 1
                            else 0
                        )

                        value = st.selectbox(
                            col_name,
                            options=[
                                0,
                                1
                            ],
                            index=default_binary,
                            format_func=lambda x:
                                "Yes (1)"
                                if x == 1
                                else "No (0)",
                            key=f"input_{col_name}",
                            help="0 = No, 1 = Yes",
                        )

                    elif col_name in CATEGORY_LABELS:

                        # --------------------------------------------
                        # HUMAN-READABLE CATEGORICAL DROPDOWN
                        #
                        # The dropdown DISPLAYS the label (e.g.
                        # "Nursing"), but the value stored in
                        # `inputs` — and therefore sent to the
                        # model — is always the original numeric
                        # code (e.g. 9500).
                        # --------------------------------------------

                        label_map = CATEGORY_LABELS[
                            col_name
                        ]

                        available_codes = sorted(
                            clean_series
                            .unique()
                            .tolist()
                        )

                        available_codes = [
                            int(code)
                            for code in available_codes
                        ]

                        try:

                            default_code = int(
                                float(
                                    default_value
                                )
                            )

                        except Exception:

                            default_code = (
                                available_codes[0]
                                if available_codes
                                else 0
                            )

                        if default_code in available_codes:

                            selected_index = (
                                available_codes.index(
                                    default_code
                                )
                            )

                        else:

                            selected_index = 0

                        def format_category_code(
                            code,
                            _label_map=label_map
                        ):

                            return _label_map.get(
                                int(code),
                                f"Unknown code ({code})"
                            )

                        value = st.selectbox(
                            col_name,
                            options=available_codes,
                            index=selected_index,
                            format_func=format_category_code,
                            key=f"input_{col_name}",
                            help=(
                                "Shown by name here, but the "
                                "original dataset code is what "
                                "is sent to the model."
                            ),
                        )

                    elif pd.api.types.is_numeric_dtype(
                        series
                    ):

                        min_value = float(
                            clean_series.min()
                        )

                        max_value = float(
                            clean_series.max()
                        )

                        try:

                            default_number = float(
                                default_value
                            )

                        except Exception:

                            default_number = float(
                                clean_series.median()
                            )

                        default_number = max(
                            min_value,
                            min(
                                default_number,
                                max_value
                            )
                        )

                        if pd.api.types.is_integer_dtype(
                            series
                        ):

                            value = st.number_input(
                                col_name,
                                min_value=int(
                                    np.floor(
                                        min_value
                                    )
                                ),
                                max_value=int(
                                    np.ceil(
                                        max_value
                                    )
                                ),
                                value=int(
                                    round(
                                        default_number
                                    )
                                ),
                                step=1,
                                format="%d",
                                key=f"input_{col_name}",
                            )

                        else:

                            value = st.number_input(
                                col_name,
                                min_value=min_value,
                                max_value=max_value,
                                value=default_number,
                                step=0.01,
                                format="%.2f",
                                key=f"input_{col_name}",
                            )

                    else:

                        unique_values = (
                            clean_series
                            .unique()
                            .tolist()
                        )

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

                        value = st.selectbox(
                            col_name,
                            options=unique_values,
                            index=selected_index,
                            key=f"input_{col_name}",
                            help=(
                                "Value/code from the "
                                "complete dataset."
                            ),
                        )

                    inputs[
                        col_name
                    ] = value

        # ====================================================
        # RENDER INPUTS
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

        input_df = pd.DataFrame(
            [inputs]
        )

        input_df = input_df[
            feature_names
        ]

        input_df = input_df.astype(float)

        # ====================================================
        # TARGET LEAKAGE SECURITY CHECK
        # ====================================================

        if "target" in input_df.columns:

            st.error(
                "❌ Security check failed: target "
                "was included in model input."
            )

            st.stop()

        # ====================================================
        # SCALE INPUT
        # ====================================================

        input_scaled = (
            results["scaler"]
            .transform(
                input_df
            )
        )

        # ====================================================
        # PREDICT PROBABILITY
        # ====================================================

        probability = (
            results["model"]
            .predict_proba(
                input_scaled
            )[0, 1]
        )

        # ====================================================
        # PREDICT CLASS
        # ====================================================

        predicted_class = int(
            results["model"]
            .predict(
                input_scaled
            )[0]
        )

        st.session_state.last_prediction = (
            predicted_class
        )

        st.session_state.last_probability = (
            float(probability)
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
        # RESULT SECTION
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
                        "text": "Dropout Probability",
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
                                "color": "#dcfce7"
                            },

                            {
                                "range": [
                                    30,
                                    60
                                ],
                                "color": "#fef3c7"
                            },

                            {
                                "range": [
                                    60,
                                    100
                                ],
                                "color": "#fee2e2"
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

                # ------------------------------------------------
                # RISK BADGE
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="risk-badge {risk_css}">
                        {risk_label}
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

                # ------------------------------------------------
                # PROBABILITY CARD
                # ------------------------------------------------

                probability_background = {
                    "result-low": "#f0fdf4",
                    "result-medium": "#fffbeb",
                    "result-high": "#fef2f2",
                }[result_css]

                probability_border = {
                    "result-low": "#86efac",
                    "result-medium": "#fcd34d",
                    "result-high": "#fca5a5",
                }[result_css]

                probability_text_color = {
                    "result-low": "#15803d",
                    "result-medium": "#b45309",
                    "result-high": "#b91c1c",
                }[result_css]

                st.html(
                    f"""
                    <div style="
                        background:{probability_background};
                        border:2px solid {probability_border};
                        border-radius:14px;
                        padding:22px;
                        margin-top:5px;
                        margin-bottom:10px;
                        width:100%;
                        box-sizing:border-box;
                    ">
                        <div style="
                            color:#1e1e1e;
                            font-size:1.05rem;
                            font-weight:750;
                        ">
                            Dropout Probability
                        </div>

                        <div style="
                            color:{probability_text_color};
                            font-size:1.65rem;
                            font-weight:850;
                            margin-top:5px;
                        ">
                            {probability * 100:.2f}%
                        </div>
                    </div>
                    """
                )

                # ------------------------------------------------
                # CLASS COLORS
                # ------------------------------------------------

                if predicted_class == 1:

                    class_color = "#dc2626"
                    class_background = "#fef2f2"
                    class_border = "#fca5a5"

                else:

                    class_color = "#16a34a"
                    class_background = "#f0fdf4"
                    class_border = "#86efac"

                # ------------------------------------------------
                # PREDICTED CLASS
                # ------------------------------------------------

                st.html(
                    f"""
                    <div style="
                        background:{class_background};
                        border:2px solid {class_border};
                        border-radius:14px;
                        padding:18px;
                        margin-bottom:12px;
                        width:100%;
                        box-sizing:border-box;
                    ">

                        <div style="
                            color:#374151;
                            font-size:1rem;
                            font-weight:700;
                        ">
                            Predicted Class
                        </div>

                        <div style="
                            color:{class_color};
                            font-size:1.55rem;
                            font-weight:800;
                            margin-top:5px;
                        ">
                            {predicted_label}
                        </div>

                    </div>
                    """
                )

                # ------------------------------------------------
                # PREDICTION MESSAGE
                # ------------------------------------------------

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
        # ACTUAL DATASET OUTCOME
        # ====================================================

        if sample_index is not None:

            # ------------------------------------------------
            # IMPORTANT:
            # Compare the model input with the ORIGINAL
            # feature values from datasett.csv.
            #
            # raw_df is used here rather than input fields.
            # ------------------------------------------------

            original_values = (
                raw_df
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

            # =================================================
            # UNCHANGED TEST STUDENT
            # =================================================

            if same_as_original:

                actual_outcome = int(
                    y_test_raw.loc[
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
                        "📌 Dataset Actual Outcome"
                    )

                    st.caption(
                        "This is the original target value "
                        "stored in datasett.csv for this test "
                        "student. It was NOT given to the model "
                        "during prediction."
                    )

                    # ------------------------------------------------
                    # ACTUAL OUTCOME CARD
                    # ------------------------------------------------

                    if actual_outcome == 1:

                        st.html(
                            """
                            <div style="
                                background:#fef2f2;
                                border:2px solid #fca5a5;
                                border-radius:14px;
                                padding:22px;
                                margin-top:5px;
                                margin-bottom:10px;
                                width:100%;
                                box-sizing:border-box;
                            ">

                                <div style="
                                    color:#1e1e1e;
                                    font-size:1.05rem;
                                    font-weight:750;
                                ">
                                    Actual Student Outcome
                                </div>

                                <div style="
                                    color:#b91c1c;
                                    font-size:1.65rem;
                                    font-weight:850;
                                    margin-top:5px;
                                ">
                                    🔴 Dropout
                                </div>

                            </div>
                            """
                        )

                    else:

                        st.html(
                            """
                            <div style="
                                background:#f0fdf4;
                                border:2px solid #86efac;
                                border-radius:14px;
                                padding:22px;
                                margin-top:5px;
                                margin-bottom:10px;
                                width:100%;
                                box-sizing:border-box;
                            ">

                                <div style="
                                    color:#1e1e1e;
                                    font-size:1.05rem;
                                    font-weight:750;
                                ">
                                    Actual Student Outcome
                                </div>

                                <div style="
                                    color:#15803d;
                                    font-size:1.65rem;
                                    font-weight:850;
                                    margin-top:5px;
                                ">
                                    🟢 Not Dropout
                                </div>

                            </div>
                            """
                        )

                    # ------------------------------------------------
                    # PREDICTION VS ACTUAL
                    # ------------------------------------------------

                    if (
                        predicted_class
                        == actual_outcome
                    ):

                        st.success(
                            "✅ Model Prediction = Dataset "
                            "Actual Outcome"
                        )

                    else:

                        st.warning(
                            "⚠️ Model Prediction ≠ Dataset "
                            "Actual Outcome. This is a valid "
                            "model error on the held-out "
                            "test student."
                        )

                    # ------------------------------------------------
                    # COMPARISON CARDS
                    # ------------------------------------------------

                    comparison_col1, comparison_col2 = (
                        st.columns(2)
                    )

                    with comparison_col1:

                        st.html(
                            f"""
                            <div style="
                                background:#eff6ff;
                                border:2px solid #93c5fd;
                                border-radius:14px;
                                padding:18px;
                                width:100%;
                                box-sizing:border-box;
                            ">

                                <div style="
                                    color:#1e3a8a;
                                    font-size:0.95rem;
                                    font-weight:700;
                                ">
                                    Model Prediction
                                </div>

                                <div style="
                                    color:#1d4ed8;
                                    font-size:1.45rem;
                                    font-weight:800;
                                    margin-top:5px;
                                ">
                                    {predicted_label}
                                </div>

                            </div>
                            """
                        )

                    with comparison_col2:

                        actual_display_color = (
                            "#dc2626"
                            if actual_outcome == 1
                            else "#16a34a"
                        )

                        actual_display_bg = (
                            "#fef2f2"
                            if actual_outcome == 1
                            else "#f0fdf4"
                        )

                        actual_display_border = (
                            "#fca5a5"
                            if actual_outcome == 1
                            else "#86efac"
                        )

                        st.html(
                            f"""
                            <div style="
                                background:{actual_display_bg};
                                border:2px solid {actual_display_border};
                                border-radius:14px;
                                padding:18px;
                                width:100%;
                                box-sizing:border-box;
                            ">

                                <div style="
                                    color:#374151;
                                    font-size:0.95rem;
                                    font-weight:700;
                                ">
                                    Dataset Actual Target
                                </div>

                                <div style="
                                    color:{actual_display_color};
                                    font-size:1.45rem;
                                    font-weight:800;
                                    margin-top:5px;
                                ">
                                    {actual_label}
                                </div>

                            </div>
                            """
                        )

            # =================================================
            # MODIFIED TEST STUDENT
            # =================================================

            else:

                st.info(
                    "ℹ️ This was originally a test-set student, "
                    "but one or more feature values were changed. "
                    "Therefore, the original dataset target is "
                    "not shown for this modified prediction."
                )

        # ====================================================
        # MANUAL INPUT
        # ====================================================

        else:

            st.info(
                "💡 This is a manually entered prediction. "
                "Because no original test-set student was loaded, "
                "there is no dataset actual outcome to compare."
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

            # ------------------------------------------------
            # LOW RISK
            # ------------------------------------------------

            with r1:

                st.html(
                    """
                    <div style="
                        background:#052e16;
                        border:1px solid #166534;
                        border-radius:12px;
                        padding:16px;
                        min-height:125px;
                        box-sizing:border-box;
                    ">

                        <h4 style="
                            color:#86efac;
                            margin-top:0;
                            margin-bottom:8px;
                            font-size:1.10rem;
                            font-weight:800;
                        ">
                            🟢 Low Risk
                        </h4>

                        <p style="
                            color:#dcfce7;
                            margin:0;
                            font-size:0.96rem;
                            line-height:1.55;
                        ">
                            Dropout probability below 30%.
                        </p>

                    </div>
                    """
                )

            # ------------------------------------------------
            # MEDIUM RISK
            # ------------------------------------------------

            with r2:

                st.html(
                    """
                    <div style="
                        background:#422006;
                        border:1px solid #a16207;
                        border-radius:12px;
                        padding:16px;
                        min-height:125px;
                        box-sizing:border-box;
                    ">

                        <h4 style="
                            color:#fde047;
                            margin-top:0;
                            margin-bottom:8px;
                            font-size:1.10rem;
                            font-weight:800;
                        ">
                            🟡 Medium Risk
                        </h4>

                        <p style="
                            color:#fef3c7;
                            margin:0;
                            font-size:0.96rem;
                            line-height:1.55;
                        ">
                            Probability from 30% to below 60%.
                        </p>

                    </div>
                    """
                )

            # ------------------------------------------------
            # HIGH RISK
            # ------------------------------------------------

            with r3:

                st.html(
                    """
                    <div style="
                        background:#450a0a;
                        border:1px solid #991b1b;
                        border-radius:12px;
                        padding:16px;
                        min-height:125px;
                        box-sizing:border-box;
                    ">

                        <h4 style="
                            color:#fca5a5;
                            margin-top:0;
                            margin-bottom:8px;
                            font-size:1.10rem;
                            font-weight:800;
                        ">
                            🔴 High Risk
                        </h4>

                        <p style="
                            color:#fee2e2;
                            margin:0;
                            font-size:0.96rem;
                            line-height:1.55;
                        ">
                            Probability of 60% or higher.
                        </p>

                    </div>
                    """
                )

            st.caption(
                "These risk bands are project-level thresholds "
                "for interpreting model probability and are "
                "not official institutional policies."
            )
