# ============================================================
# PREDICTION RESULT
# ============================================================

class_text = "Dropout" if predicted_class == 1 else "Not Dropout"

# ------------------------------------------------------------
# Risk Level
# ------------------------------------------------------------

if probability < 0.30:
    risk_level = "Low Risk"
    risk_icon = "🟢"
    risk_background = "#f0fdf4"
    risk_border = "#86efac"
    risk_color = "#15803d"

elif probability < 0.60:
    risk_level = "Medium Risk"
    risk_icon = "🟡"
    risk_background = "#fffbeb"
    risk_border = "#fcd34d"
    risk_color = "#b45309"

else:
    risk_level = "High Risk"
    risk_icon = "🔴"
    risk_background = "#fef2f2"
    risk_border = "#fca5a5"
    risk_color = "#dc2626"


# ------------------------------------------------------------
# Result Header
# ------------------------------------------------------------

st.markdown(
    """
    <div style="
        margin-top:25px;
        margin-bottom:15px;
    ">
        <h3 style="
            color:#ffffff !important;
            margin:0;
            font-size:1.45rem;
            font-weight:800;
        ">
            🎯 Prediction Result
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Risk Badge
# ------------------------------------------------------------

st.markdown(
    f"""
    <div style="
        background:{risk_background};
        border:2px solid {risk_border};
        border-radius:14px;
        padding:14px 18px;
        margin-bottom:15px;
        text-align:center;
    ">
        <div style="
            color:{risk_color} !important;
            font-size:1.15rem;
            font-weight:800;
        ">
            {risk_icon} {risk_level}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Probability + Predicted Class Cards
# ------------------------------------------------------------

result_col1, result_col2 = st.columns(2)


# ============================================================
# DROPOUT PROBABILITY CARD
# ============================================================

with result_col1:

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:2px solid #e5e7eb;
            border-radius:14px;
            padding:20px;
            min-height:125px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
        ">

            <div style="
                color:#374151 !important;
                font-size:1rem;
                font-weight:700;
                margin-bottom:7px;
            ">
                Dropout Probability
            </div>

            <div style="
                color:{risk_color} !important;
                font-size:1.9rem;
                font-weight:800;
                line-height:1.2;
            ">
                {probability * 100:.2f}%
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTED CLASS CARD
# ============================================================

with result_col2:

    if predicted_class == 1:
        class_color = "#dc2626"
    else:
        class_color = "#16a34a"

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:2px solid #e5e7eb;
            border-radius:14px;
            padding:20px;
            min-height:125px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
        ">

            <div style="
                color:#374151 !important;
                font-size:1rem;
                font-weight:700;
                margin-bottom:7px;
            ">
                Predicted Class
            </div>

            <div style="
                color:{class_color} !important;
                font-size:1.55rem;
                font-weight:800;
                margin-top:5px;
            ">
                {class_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Prediction Message
# ------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if predicted_class == 1:

    st.warning(
        "⚠️ The model predicts that this student has a higher "
        "likelihood of dropping out."
    )

else:

    st.success(
        "✅ The model predicts that this student has a lower "
        "likelihood of dropping out."
    )


# ============================================================
# ACTUAL DATASET OUTCOME
# ============================================================

if sample_index is not None:

    # --------------------------------------------------------
    # Check whether the user changed the Quick Test student
    # --------------------------------------------------------

    original_test_row = X_test_raw.loc[sample_index]

    current_features = np.array(
        [inputs[col] for col in feature_names],
        dtype=float
    )

    original_features = np.array(
        [original_test_row[col] for col in feature_names],
        dtype=float
    )

    features_match = np.allclose(
        current_features,
        original_features,
        rtol=1e-5,
        atol=1e-5
    )

    # --------------------------------------------------------
    # Original Test Student
    # --------------------------------------------------------

    if features_match:

        actual_outcome = int(y_test_raw.loc[sample_index])

        actual_text = (
            "Dropout"
            if actual_outcome == 1
            else "Not Dropout"
        )

        actual_color = (
            "#dc2626"
            if actual_outcome == 1
            else "#16a34a"
        )

        st.markdown(
            f"""
            <div style="
                background:#f8fafc;
                border:1px solid #cbd5e1;
                border-radius:14px;
                padding:18px;
                margin-top:18px;
            ">

                <div style="
                    color:#334155 !important;
                    font-size:1rem;
                    font-weight:800;
                    margin-bottom:8px;
                ">
                    📌 Actual Dataset Outcome
                </div>

                <div style="
                    color:{actual_color} !important;
                    font-size:1.3rem;
                    font-weight:800;
                ">
                    {actual_text}
                </div>

                <div style="
                    color:#64748b !important;
                    font-size:0.9rem;
                    margin-top:7px;
                ">
                    This value comes from the held-out test set and was
                    not provided to the model during prediction.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Prediction vs Actual
        # ----------------------------------------------------

        prediction_correct = (
            predicted_class == actual_outcome
        )

        if prediction_correct:

            st.success(
                f"✅ Prediction matches the actual test-set outcome: "
                f"**{actual_text}**."
            )

        else:

            st.error(
                f"❌ Prediction does not match the actual test-set "
                f"outcome. Model predicted **{class_text}**, while "
                f"the actual outcome was **{actual_text}**."
            )

    # --------------------------------------------------------
    # Modified Quick Test Student
    # --------------------------------------------------------

    else:

        st.info(
            "ℹ️ This was originally a test-set student, but one or more "
            "feature values were changed. Therefore, the original dataset "
            "target is not shown for this modified prediction."
        )


# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        margin-top:10px;
        margin-bottom:15px;
    ">
        <h3 style="
            color:#ffffff !important;
            margin:0;
            font-size:1.35rem;
            font-weight:800;
        ">
            📖 Risk Interpretation
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RISK INTERPRETATION CARDS
# ============================================================

risk_col1, risk_col2, risk_col3 = st.columns(3)


# ------------------------------------------------------------
# LOW RISK
# ------------------------------------------------------------

with risk_col1:

    st.markdown(
        """
        <div style="
            background:#0f172a;
            border:1px solid #334155;
            border-radius:14px;
            padding:20px;
            min-height:135px;
            box-shadow:0 2px 8px rgba(0,0,0,0.20);
        ">

            <h4 style="
                color:#22c55e !important;
                margin:0 0 10px 0;
                font-size:1.1rem;
                font-weight:800;
            ">
                🟢 Low Risk
            </h4>

            <p style="
                color:#e2e8f0 !important;
                margin:0;
                font-size:0.95rem;
                line-height:1.5;
            ">
                Dropout probability below 30%.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# MEDIUM RISK
# ------------------------------------------------------------

with risk_col2:

    st.markdown(
        """
        <div style="
            background:#0f172a;
            border:1px solid #334155;
            border-radius:14px;
            padding:20px;
            min-height:135px;
            box-shadow:0 2px 8px rgba(0,0,0,0.20);
        ">

            <h4 style="
                color:#facc15 !important;
                margin:0 0 10px 0;
                font-size:1.1rem;
                font-weight:800;
            ">
                🟡 Medium Risk
            </h4>

            <p style="
                color:#e2e8f0 !important;
                margin:0;
                font-size:0.95rem;
                line-height:1.5;
            ">
                Probability from 30% to below 60%.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# HIGH RISK
# ------------------------------------------------------------

with risk_col3:

    st.markdown(
        """
        <div style="
            background:#0f172a;
            border:1px solid #334155;
            border-radius:14px;
            padding:20px;
            min-height:135px;
            box-shadow:0 2px 8px rgba(0,0,0,0.20);
        ">

            <h4 style="
                color:#f87171 !important;
                margin:0 0 10px 0;
                font-size:1.1rem;
                font-weight:800;
            ">
                🔴 High Risk
            </h4>

            <p style="
                color:#e2e8f0 !important;
                margin:0;
                font-size:0.95rem;
                line-height:1.5;
            ">
                Probability of 60% or higher.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div style="
        background:#1e293b;
        border:1px solid #475569;
        border-radius:12px;
        padding:14px 16px;
        margin-top:18px;
    ">

        <div style="
            color:#cbd5e1 !important;
            font-size:0.88rem;
            line-height:1.5;
        ">
            ℹ️ These risk thresholds are project-level interpretation
            thresholds for this application. They are not official
            institutional or medical policies.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
