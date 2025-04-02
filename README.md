
# 🏦 Loan Approval Prediction Model

This project uses supervised machine learning algorithms to predict whether a loan application will be approved based on applicant information such as income, credit history, education, and employment status.

## 🔍 Problem Statement

Financial institutions need efficient ways to assess loan eligibility to reduce risk and improve customer satisfaction. This model assists in predicting loan approvals using historical data and classification techniques.

---

## 📊 Dataset

- Source: [Kaggle Loan Prediction Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)
- Records: 614 loan applicants
- Features include:
  - Gender, Marital Status, Dependents
  - Education, Self-Employed
  - Applicant/Coapplicant Income
  - Loan Amount & Term
  - Credit History, Property Area
  - Loan Status (Target)

---

## 🛠 Technologies Used

- **Language:** Python
- **Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`
- **Algorithms:** Random Forest, Decision Tree, K-Nearest Neighbors, Naive Bayes

---

## ⚙️ Preprocessing

- Handled missing values using mode/mean imputation
- Applied log transformation to skewed features (`LoanAmount`, `TotalIncome`)
- Label-encoded categorical variables
- Created derived features: `TotalIncome`, `LoanAmount_log`

---

## 🧠 Model Training & Evaluation

| Algorithm         | Accuracy (%) |
|------------------|--------------|
| Random Forest     | 77.23        |
| Naive Bayes       | 82.92        |
| Decision Tree     | 74.80        |
| KNN               | 79.67        |

- Split data into train (80%) and test (20%)
- Evaluated using accuracy, confusion matrix, and classification report

---

## 📈 Visualizations

- Distribution of loan approvals by:
  - Gender
  - Marital Status
  - Credit History
- Histogram of income and loan amounts
- Correlation heatmap of key features

---

## 📌 Results & Insights

- Credit history and total income are strong predictors of loan approval.
- Naive Bayes performed best with ~83% accuracy.
- Preprocessing and feature engineering significantly improved performance.

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Use cross-validation for robust evaluation
- Deploy model with Streamlit or Flask for user interaction

---

## 📁 Folder Structure

