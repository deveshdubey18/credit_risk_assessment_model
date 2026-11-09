# Credit Risk Assessment Model

A machine learning classification project that predicts whether an account is marked as **at risk** using historical trade and credit-behaviour features. The repository includes exploratory data analysis (EDA), data cleaning, preprocessing, class-imbalance handling, model training, evaluation, and experiments with automated machine learning tools.

The main training pipeline currently uses a **Random Forest Classifier** to predict the `is_at_risk` target.

## Project Overview

The target column is `is_at_risk`:

- `0` — Not marked as at risk
- `1` — Marked as at risk

The processed dataset contains **2,502 records**, with **19 input features** and one target column.

## Model Performance

The current Random Forest pipeline achieved **70% accuracy** on a held-out test set of 751 records.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|:---:|---:|---:|---:|---:|
| 0 — Not at risk | 0.60 | 0.54 | 0.57 | 274 |
| 1 — At risk | 0.75 | 0.79 | 0.77 | 477 |
| **Accuracy** |  |  | **0.70** | **751** |
| **Macro average** | **0.67** | **0.67** | **0.67** | **751** |
| **Weighted average** | **0.70** | **0.70** | **0.70** | **751** |

**Result interpretation:** The model identifies the at-risk class (class 1) with 79% recall and a 0.77 F1-score. Recall for class 0 is 54%, meaning some accounts in that class are incorrectly classified. In credit-risk use cases, false positives and false negatives can have different business costs, so accuracy should not be considered in isolation.

## Project Workflow

1. **Data ingestion** — Load the processed CSV dataset.
2. **Exploratory data analysis** — Inspect data types, summary statistics, missing values, duplicates, and outliers.
3. **Data cleaning** — Remove rows containing negative values treated as invalid/sentinel values in the raw data, remove selected columns, and apply winsorization to limit extreme values.
4. **Train-test split** — Separate the features and target using a 70/30 split.
5. **Feature scaling** — Fit `StandardScaler` on the training data and use it to transform the test data.
6. **Class-imbalance handling** — Apply SMOTE to the training data only.
7. **Model training** — Train a `RandomForestClassifier`.
8. **Evaluation and export** — Generate a classification report and save the trained classifier as `models/model.pkl`.

## Exploratory Analysis and Experiments

The repository also contains notebooks for exploring and comparing modelling approaches:

- **EDA** — Data inspection, visualization, missing/invalid value checks, outlier analysis, winsorization, and model experiments.
- **Sweetviz report** — Automated exploratory data analysis.
- **FLAML AutoML** — An experiment using automated learner selection for classification.
- **AutoGluon** — An experiment using AutoGluon's tabular classification workflow.
- **Random Forest tuning** — The EDA notebook includes a `RandomizedSearchCV` experiment.

These notebooks are exploratory. The classification report above describes the current Random Forest pipeline implemented in `src/credit_risk_assessment_ml_model/`.

## Dataset Features

The processed dataset uses the following 19 predictor columns:

- `months_since_first_trade`
- `months_since_last_trade`
- `average_duration_of_resolution`
- `number_of_satisfactory_trades`
- `nr_trades_insolvent_for_over_60_days`
- `percentage_of_legal_trades`
- `months_since_last_illegal_trade`
- `maximum_illegal_trades_over_last_year`
- `nr_trades_initiated_in_last_year`
- `percentage_of_installment_trades`
- `months_since_last_inquiry_not_recent`
- `nr_inquiries_in_last_6_months`
- `nr_inquiries_in_last_6_months_not_recent`
- `net_fraction_of_revolving_burden`
- `net_fraction_of_installment_burden`
- `nr_revolving_trades_with_balance`
- `nr_installment_trades_with_balance`
- `nr_banks_with_high_ratio`
- `percentage_trades_with_balance`

The target column is `is_at_risk`.

## Repository Structure

```text
credit_risk_assessment_model/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   └── processed/
│       └── processedData.csv
│
├── eda/
│   ├── EDA.ipynb
│   ├── data_visualize_using_SweetViz.ipynb
│   └── sweet_report.html
│
├── experiments/
│   ├── Experiment_1_automl.ipynb
│   └── Experiment_2_autogluon.ipynb
│
├── logs/
│   └── model.log
│
├── models/
│   └── model.pkl
│
├── src/
│   └── credit_risk_assessment_ml_model/
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       └── model_build.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Technologies Used

- **Python 3.13** — Main programming language
- **Pandas and NumPy** — Data loading and manipulation
- **Matplotlib and Seaborn** — Data visualization
- **Sweetviz and AutoViz** — Exploratory data analysis
- **Scikit-learn** — Data splitting, feature scaling, Random Forest, and evaluation
- **Imbalanced-learn** — SMOTE for class-imbalance handling
- **FLAML and AutoGluon** — Automated machine learning experiments
- **Pickle** — Saving the trained model
- **uv** — Python project and dependency management

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/deveshdubey18/credit_risk_assessment_model.git
cd credit_risk_assessment_model
```

### 2. Set up the environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it if needed:

```bash
pip install uv
```

Install the project dependencies:

```bash
uv sync
```

### 3. Configure the dataset path

Before running the pipeline, open:

`src/credit_risk_assessment_ml_model/data_ingestion.py`

The current implementation uses a machine-specific Windows absolute path. Update it to point to your local processed dataset. For example, when running commands from the repository root:

```python
df = pd.read_csv("data/processed/processedData.csv")
```

### 4. Run the model

From the repository root, run:

```bash
uv run python main.py
```

The script loads the processed data, preprocesses it, trains the Random Forest model, prints the classification report, and saves the trained classifier to:

```text
models/model.pkl
```

## Notes and Limitations

- The reported metrics come from one train-test split; performance may vary with different data splits or future data.
- SMOTE is applied after the train-test split and only to the training data, keeping the held-out test set untouched by resampling.
- The current training code saves the classifier but does **not** save the fitted `StandardScaler`. For consistent predictions on new records, save the fitted scaler alongside the model and apply the same preprocessing steps at inference time.
- This is a learning and experimentation project, not a validated production credit-decision system. Real-world use would require further validation, monitoring, fairness assessment, threshold analysis, and review of the costs of false positives and false negatives.


---

## 👨‍💻 Author

**Devesh Dubey**
- **GitHub:** [deveshdubey18](https://github.com/deveshdubey18)

---
If you find this project useful, feel free to explore the notebooks and experiment with other classification models and evaluation methods.

---

*If you find this project useful, consider starring the repository and sharing your feedback.*
