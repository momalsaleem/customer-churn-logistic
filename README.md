# Customer Churn Prediction
# Customer Churn Prediction

A machine learning project using a **Logistic Regression** model to predict telecom customer churn.

## Dataset
- **File**: `customer.csv`
- **Size**: 7,043 telecom customers × 21 features
- **Target**: Churn (Yes / No)

## Notebook
`ml.ipynb` contains the full machine learning pipeline:
*EDA → Preprocessing → Training → Evaluation*

## Pipeline
`customer.csv`
1. **EDA** (shape, nulls, distributions, correlation heatmap)
2. **Encode Churn label** (Yes=1, No=0)
3. **One-hot encode categorical features** (`pd.get_dummies`)
4. **StandardScaler** (Scaling numerical features)
5. **train_test_split** (80/20, `random_state=42`)
6. **LogisticRegression** (`max_iter=1000`)

---

## Results

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 81.3% |
| **Macro F1** | 0.76 |
| **Weighted F1** | 0.81 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **No Churn (0)** | 0.88 | 0.86 | 0.87 |
| **Churn (1)** | 0.64 | 0.68 | 0.66 |

### Confusion Matrix

| | Predicted: No Churn | Predicted: Churn |
| :--- | :--- | :--- |
| **Actual: No Churn** | 893 ✅ | 143 |
| **Actual: Churn** | 121 | 252 ✅ |

---

## Potential Improvements

- Address class imbalance (~73% No Churn) with `class_weight='balanced'` or SMOTE.
- Try **XGBoost** or **RandomForest** for better churn recall.
- Feature selection to reduce noise from one-hot encoded columns.

## Setup & Execution

**Install requirements:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Run the pipeline:**
```bash
jupyter notebook ml.ipynb
