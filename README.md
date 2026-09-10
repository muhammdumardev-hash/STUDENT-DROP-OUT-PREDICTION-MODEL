

# 🎓 Student Dropout Prediction System

A Machine Learning web application that predicts whether a student is likely to **Dropout** or **Not Dropout** based on demographic, enrollment, academic, financial, and socioeconomic information.

The project uses **Logistic Regression** and provides an interactive Streamlit dashboard for data analysis, model evaluation, and individual student risk prediction.

The system can be used as an **educational early-warning system** to identify students who may require additional academic or financial support.

---

## 🚀 Live Application

The trained machine learning model is deployed using **Streamlit Community Cloud**.

Users can enter student information through the web interface and receive:

* Dropout Probability
* Risk Level
* Predicted Class
* Model-based interpretation

---

# 📊 Dataset Features

The dataset contains **37 input features** describing different aspects of a student's academic and personal situation.

## 👤 1. Demographic Information

### 1. Nationality

Represents the student's nationality or nationality category.

### 2. Gender

Represents the student's gender.

* `0` = Female
* `1` = Male

### 3. Age at Enrollment

The student's age when they enrolled in the educational program.

This feature can help identify whether age at enrollment is associated with dropout risk.

### 4. International

Indicates whether the student is an international student.

* `0` = No
* `1` = Yes

### 5. Displaced

Indicates whether the student has displaced status.

* `0` = No
* `1` = Yes

### 6. Educational Special Needs

Indicates whether the student has educational special needs.

* `0` = No
* `1` = Yes

---

# 📝 2. Enrollment Information

### 7. Application Mode

Represents the method or type through which the student applied for admission.

Different application modes represent different admission pathways.

### 8. Application Order

Represents the order in which the student selected the course/application preference.

### 9. Course

Represents the academic course or program selected by the student.

### 10. Daytime/Evening Attendance

Indicates the student's attendance schedule.

* `1` = Daytime
* `0` = Evening

### 11. Previous Qualification

Represents the qualification obtained by the student before entering the current program.

### 12. Mother's Qualification

Represents the highest educational qualification of the student's mother.

### 13. Father's Qualification

Represents the highest educational qualification of the student's father.

### 14. Mother's Occupation

Represents the occupation category of the student's mother.

### 15. Father's Occupation

Represents the occupation category of the student's father.

---

# 📚 3. Academic Information

These features describe the student's previous academic performance and first- and second-semester performance.

### 16. Previous Qualification (Grade)

The grade obtained by the student in their previous qualification.

### 17. Admission Grade

The grade obtained by the student during the admission process.

### 18. Curricular Units 1st Sem (Credited)

Number of first-semester curricular units that were credited from previous studies.

### 19. Curricular Units 1st Sem (Enrolled)

Number of curricular units in which the student enrolled during the first semester.

### 20. Curricular Units 1st Sem (Evaluations)

Number of first-semester curricular units in which the student received an evaluation.

### 21. Curricular Units 1st Sem (Approved)

Number of first-semester curricular units successfully passed/approved by the student.

### 22. Curricular Units 1st Sem (Grade)

The student's average/overall grade for first-semester curricular units.

### 23. Curricular Units 1st Sem (Without Evaluations)

Number of first-semester curricular units for which the student did not receive an evaluation.

### 24. Curricular Units 2nd Sem (Credited)

Number of second-semester curricular units that were credited from previous studies.

### 25. Curricular Units 2nd Sem (Enrolled)

Number of curricular units in which the student enrolled during the second semester.

### 26. Curricular Units 2nd Sem (Evaluations)

Number of second-semester curricular units in which the student received an evaluation.

### 27. Curricular Units 2nd Sem (Approved)

Number of second-semester curricular units successfully passed/approved by the student.

### 28. Curricular Units 2nd Sem (Grade)

The student's average/overall grade for second-semester curricular units.

### 29. Curricular Units 2nd Sem (Without Evaluations)

Number of second-semester curricular units for which the student did not receive an evaluation.

---

# 💰 4. Financial & Socioeconomic Information

### 30. Debtor

Indicates whether the student has outstanding financial obligations.

* `0` = No
* `1` = Yes

### 31. Tuition Fees Up to Date

Indicates whether the student's tuition fees are up to date.

* `0` = No
* `1` = Yes

### 32. Scholarship Holder

Indicates whether the student receives a scholarship.

* `0` = No
* `1` = Yes

### 33. Unemployment Rate

Represents the unemployment rate associated with the economic conditions of the dataset's reference period.

### 34. Inflation Rate

Represents the inflation rate associated with the economic conditions of the dataset's reference period.

### 35. GDP

Represents the Gross Domestic Product (GDP) indicator associated with the economic conditions of the reference period.

---

# 👨‍👩‍👦 5. Additional Information

### 36. Marital Status

Represents the student's marital status category.

### 37. Target

The target variable represents the student's final educational outcome.

The original outcome contains:

* `Dropout`
* `Enrolled`
* `Graduate`

For this project, the target was converted into a **binary classification problem**:

| Original Outcome | Model Class |
| ---------------- | ----------- |
| Dropout          | `1`         |
| Enrolled         | `0`         |
| Graduate         | `0`         |

Therefore, the model predicts:

**1 → Dropout**

**0 → Not Dropout**

`Graduate` and `Enrolled` are grouped into **Not Dropout** because the main objective of this project is to identify students who are at risk of dropping out.

---

# 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

### Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Student Risk Prediction
```

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

The model learns from the training data and is evaluated using the unseen testing data.

---

# 📈 Model Evaluation

The application evaluates the model using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve
* Classification Report

These metrics help measure how effectively the model distinguishes between **Dropout** and **Not Dropout** students.

---

# ⚠️ Student Risk Levels

The application converts the predicted dropout probability into three risk categories.

| Dropout Probability | Risk Level     |
| ------------------: | -------------- |
|             `< 30%` | 🟢 Low Risk    |
|        `30% – <60%` | 🟡 Medium Risk |
|             `≥ 60%` | 🔴 High Risk   |

For example:

```text
12.79% → Low Risk
40.61% → Medium Risk
96.28% → High Risk
```

**Important:** Risk level and predicted class are not exactly the same thing.

For example:

```text
40.61% → Medium Risk
Predicted Class → Not Dropout
```

This is valid because the student has a moderate estimated risk but the model's binary prediction is still `Not Dropout`.

---

# 🎯 Educational Early-Warning System

This project can potentially be used as an **educational early-warning system**.

Educational institutions could use student information to identify students who show higher estimated dropout risk.

For example, a university could process student records and identify:

* Students with high dropout probability
* Students with declining academic performance
* Students with low numbers of approved courses
* Students with financial difficulties
* Students who may require additional academic support

The purpose would not be to automatically label or punish students. Instead, the predictions could help institutions identify students who may benefit from:

* Academic counseling
* Financial assistance
* Mentoring
* Additional academic support
* Student success programs

The model should therefore be treated as a **decision-support tool**, not as a final decision-maker.

---

# 🖥️ Streamlit Application

The application provides several sections:

### 📊 Dataset Overview

Displays information about the dataset and its features.

### 🔍 Exploratory Data Analysis

Provides visual analysis of the dataset and target distribution.

### 🤖 Model Performance

Displays machine learning evaluation metrics and model performance.

### 🎯 Predict Risk

Allows users to enter student information and receive:

* Dropout probability
* Risk level
* Predicted class
* Prediction interpretation

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

# 📁 Project Structure

```text
student-drop-out-prediction-model/
│
├── app.py
├── datasett.csv
├── requirements.txt
└── README.md
```

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

The project can be reproduced by:

1. Cloning the GitHub repository.
2. Installing the required Python packages.
3. Running the Streamlit application.
4. Deploying the repository through Streamlit Community Cloud.

---

# 🧪 Example Predictions

The application was tested with different student profiles.

### 🟢 Low Risk

```text
Dropout Probability: 3.83%
Risk Level: Low Risk
Predicted Class: Not Dropout
```

### 🟡 Medium Risk

```text
Dropout Probability: 40.61%
Risk Level: Medium Risk
Predicted Class: Not Dropout
```

### 🔴 High Risk

```text
Dropout Probability: 96.28%
Risk Level: High Risk
Predicted Class: Dropout
```

These examples demonstrate that the application can produce different risk levels based on the provided student information.

---

# ⚠️ Limitations

This project is intended for educational and demonstration purposes.

The predicted probability does not guarantee that a student will actually drop out. Real-world outcomes can be influenced by many factors that may not be included in the dataset.

The model should therefore be used as a **supporting analytical tool**, rather than as the sole basis for decisions about students.

---

# 👨‍💻 Conclusion

The Student Dropout Prediction project demonstrates a complete end-to-end Machine Learning workflow, including:

* Dataset processing
* Exploratory data analysis
* Feature preparation
* Logistic Regression
* Model evaluation
* Risk prediction
* Streamlit development
* GitHub documentation
* Cloud deployment

The project demonstrates how Machine Learning can potentially support educational institutions in identifying students who may need additional support and intervention.
