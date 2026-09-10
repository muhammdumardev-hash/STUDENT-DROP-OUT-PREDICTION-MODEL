"""
Student Dropout Prediction — Streamlit App
Covers: EDA (from your notebook), Model Training/Evaluation, and
Phase 7 — Student Risk Prediction Application (probability + risk level).

Run locally:
    streamlit run app.py

Deploy: push this file + requirements.txt + datasett.csv to GitHub,
then deploy on Streamlit Community Cloud.
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

# ----------------------------------------------------------------------
# PAGE CONFIG + STYLING
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
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
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .risk-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    .risk-low { background: #dcfce7; color: #15803d; }
    .risk-medium { background: #fef9c3; color: #a16207; }
    .risk-high { background: #fee2e2; color: #b91c1c; }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
DEFAULT_FILE = "datasett.csv"


@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):
    df = df.copy()
    if df["target"].dtype == object:
        df["target"] = df["target"].replace(
            {"Dropout": 1, "Graduate": 0, "Enrolled": 0}
        )
    df["target"] = df["target"].astype(int)
    return df


@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

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
        "tpr": tpr,
    }


# ----------------------------------------------------------------------
# SIDEBAR — DATA SOURCE + NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🎓 Dropout Predictor")
st.sidebar.caption("Logistic Regression • Internship Project")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

data_source = None
try:
    if uploaded is not None:
        data_source = uploaded
    else:
        data_source = DEFAULT_FILE
    raw_df = load_data(data_source)
except FileNotFoundError:
    st.sidebar.error(f"Couldn't find '{DEFAULT_FILE}'. Upload your CSV above.")
    st.title("🎓 Student Dropout Prediction")
    st.warning(
        f"No dataset found. Please upload the CSV file (e.g. `{DEFAULT_FILE}`) "
        "using the sidebar to continue, or place it next to `app.py` before deploying."
    )
    st.stop()

df = clean_data(raw_df)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🔮 Predict Risk"],
)

st.sidebar.markdown("---")
st.sidebar.metric("Total Students", f"{df.shape[0]:,}")
st.sidebar.metric("Dropout Rate", f"{df['target'].mean()*100:.1f}%")

results = train_model(df)

# ----------------------------------------------------------------------
# PAGE 1 — OVERVIEW
# ----------------------------------------------------------------------
if page == "🏠 Overview":
    st.markdown('<p class="main-title">🎓 Student Dropout Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Logistic Regression model to identify at-risk students early.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><b>Rows</b><h3>{df.shape[0]:,}</h3></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><b>Columns</b><h3>{df.shape[1]}</h3></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><b>Missing Values</b><h3>{int(df.isnull().sum().sum())}</h3></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><b>Duplicate Rows</b><h3>{int(df.duplicated().sum())}</h3></div>', unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Preview", "Data Types", "Target Distribution"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        dtypes_df = pd.DataFrame({"Column": df.dtypes.index, "Dtype": df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True, height=350)

    with tab3:
        counts = df["target"].value_counts().rename({0: "Not Dropout", 1: "Dropout"})
        st.dataframe(counts.rename("Count"), use_container_width=True)

# ----------------------------------------------------------------------
# PAGE 2 — EDA (the graphs from your script, interactive)
# ----------------------------------------------------------------------
elif page == "📊 Exploratory Analysis":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visual patterns behind student dropout.</p>', unsafe_allow_html=True)

    plot_df = df.copy()
    plot_df["Status"] = plot_df["target"].map({0: "Not Dropout", 1: "Dropout"})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dropout vs Not Dropout")
        counts = plot_df["Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.bar(
            counts, x="Status", y="Count", color="Status",
            color_discrete_map={"Not Dropout": "#3b82f6", "Dropout": "#ef4444"},
            text="Count",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Tuition fees up to date" in df.columns:
            st.subheader("Tuition Fees Status vs Dropout Rate")
            rate = (
                plot_df.groupby("Tuition fees up to date")["target"]
                .mean()
                .reset_index()
            )
            rate["Tuition fees up to date"] = rate["Tuition fees up to date"].map(
                {0: "Not Up to Date", 1: "Up to Date"}
            )
            fig = px.bar(
                rate, x="Tuition fees up to date", y="target",
                color="Tuition fees up to date",
                color_discrete_sequence=["#f59e0b", "#10b981"],
                labels={"target": "Dropout Rate"},
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("'Tuition fees up to date' column not found in this dataset.")

    col3, col4 = st.columns(2)

    with col3:
        if "Curricular units 1st sem (grade)" in df.columns:
            st.subheader("1st Semester Grade vs Dropout")
            fig = px.box(
                plot_df, x="Status", y="Curricular units 1st sem (grade)",
                color="Status",
                color_discrete_map={"Not Dropout": "#3b82f6", "Dropout": "#ef4444"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "Curricular units 2nd sem (grade)" in df.columns:
            st.subheader("2nd Semester Grade vs Dropout")
            fig = px.box(
                plot_df, x="Status", y="Curricular units 2nd sem (grade)",
                color="Status",
                color_discrete_map={"Not Dropout": "#3b82f6", "Dropout": "#ef4444"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.write("")
    with st.expander("🔎 Constant columns (zero variance)"):
        const_cols = df.nunique()[df.nunique() == 1]
        if len(const_cols) == 0:
            st.write("None found.")
        else:
            st.dataframe(const_cols.rename("Unique Values"))

# ----------------------------------------------------------------------
# PAGE 3 — MODEL PERFORMANCE
# ----------------------------------------------------------------------
elif page == "🤖 Model Performance":
    st.markdown('<p class="main-title">🤖 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Logistic Regression evaluated on the held-out test set.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><b>Accuracy</b><h3>{results["accuracy"]*100:.2f}%</h3></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><b>ROC-AUC</b><h3>{results["roc_auc"]:.3f}</h3></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><b>Test Samples</b><h3>{results["X_test"].shape[0]}</h3></div>', unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = results["cm"]
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Dropout", "Dropout"], y=["Not Dropout", "Dropout"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["fpr"], y=results["tpr"], mode="lines", name="ROC curve", line=dict(color="#3b82f6", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(results["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE 4 — PHASE 7: STUDENT RISK PREDICTION APPLICATION
# ----------------------------------------------------------------------
elif page == "🔮 Predict Risk":
    st.markdown('<p class="main-title">🔮 Student Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Enter a student\'s details to get a dropout probability and risk category.</p>',
        unsafe_allow_html=True,
    )

    feature_names = results["feature_names"]
    X_train_raw = results["X_train"]

    # ---- load a real test case as a quick-fill option ----
    X_test_raw = df.loc[results["X_test"].index]
    if "sample_row" not in st.session_state:
        st.session_state.sample_row = None

    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.caption("Tip: load a real student record from the test set to try the model quickly, then tweak values.")
    with top_c2:
        if st.button("🎲 Load random test case", use_container_width=True):
            st.session_state.sample_row = X_test_raw.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]

    defaults = st.session_state.sample_row if st.session_state.sample_row is not None else None

    with st.form("prediction_form"):
        st.markdown("#### Student Information")
        inputs = {}
        cols = st.columns(3)
        for i, col_name in enumerate(feature_names):
            target_col = cols[i % 3]
            series = X_train_raw[col_name]
            uniq_vals = sorted(series.unique().tolist())
            default_val = defaults[col_name] if defaults is not None else series.median()

            with target_col:
                if len(uniq_vals) <= 6:
                    # categorical / binary-like feature
                    idx = uniq_vals.index(default_val) if default_val in uniq_vals else 0
                    val = st.selectbox(col_name, options=uniq_vals, index=idx, key=f"in_{col_name}")
                else:
                    val = st.number_input(
                        col_name,
                        min_value=float(series.min()),
                        max_value=float(series.max()),
                        value=float(default_val),
                        key=f"in_{col_name}",
                    )
            inputs[col_name] = val

        submitted = st.form_submit_button("🔍 Predict Dropout Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([inputs])[feature_names]
        input_scaled = results["scaler"].transform(input_df)

        prob = results["model"].predict_proba(input_scaled)[0, 1]
        pred_class = results["model"].predict(input_scaled)[0]

        if prob < 0.30:
            risk_label, risk_css = "Low Risk", "risk-low"
        elif prob < 0.60:
            risk_label, risk_css = "Medium Risk", "risk-medium"
        else:
            risk_label, risk_css = "High Risk", "risk-high"

        st.write("---")
        st.markdown("### Prediction Result")

        r1, r2 = st.columns([1, 1])

        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "Dropout Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f2937"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                },
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown(f'<div class="risk-badge {risk_css}">{risk_label}</div>', unsafe_allow_html=True)
            st.write("")
            st.metric("Predicted Class", "Dropout" if pred_class == 1 else "Not Dropout")
            st.metric("Dropout Probability", f"{prob*100:.2f}%")

            if defaults is not None:
                actual = df.loc[X_test_raw[X_test_raw.eq(defaults).all(axis=1)].index[0], "target"] \
                    if not X_test_raw[X_test_raw.eq(defaults).all(axis=1)].empty else None
                if actual is not None:
                    st.metric("Actual Outcome (loaded test case)", "Dropout" if actual == 1 else "Not Dropout")

        st.info(
            "Risk bands used: **Low** < 30%, **Medium** 30–60%, **High** ≥ 60%. "
            "Adjust these thresholds in the code to match your institution's policy."
        )
