# Heart Disease Risk Modeling with Bootstrap Uncertainty

## Overview

This project builds an interpretable cardiovascular risk model using the UCI Heart Disease dataset.

The focus is not complex machine learning.
The focus is:

- Proper preprocessing of mixed-type data
- Clean train/test separation
- Logistic regression for interpretability
- Bootstrap-based uncertainty estimation
- Reproducible pipelines using scikit-learn

This project demonstrates applied statistical reasoning in a healthcare context.

---

## Dataset

Source: UCI Machine Learning Repository – Heart Disease dataset

The original label `num` encodes disease severity from 0 to 4.

We convert it into a binary outcome:

    df["target"] = (df["num"] > 0).astype(int)

Meaning:

- 0 → No heart disease
- 1 → Presence of heart disease

We also remove non-feature columns such as:

- `num` (original label)
- `id` (identifier)
- `dataset` (constant in this subset)

---

## Project Structure

The script contains the following key components:

- Data loading
- Target engineering
- Diagnostic utilities
- Preprocessing + modeling pipeline
- Train/test evaluation
- Bootstrap AUC estimation
- Bootstrap odds ratio estimation
- CSV artifact export

---

## Data Loading

The dataset is loaded with explicit handling of boolean strings:

    pd.read_csv(
        path,
        true_values=["TRUE", "True", "true"],
        false_values=["FALSE", "False", "false"]
    )

This ensures boolean columns are parsed correctly.

---

## Diagnostics

Several helper functions provide basic inspection:

- Dataset shape
- Column types
- Missing values
- Target distribution
- Statistical summary (numeric + categorical)

These are simple print-based checks to understand the dataset before modeling.

---

## Preprocessing Pipeline

The core of the project is a scikit-learn `Pipeline` combined with a `ColumnTransformer`.

### Column Separation

Columns are split by dtype:

- Boolean columns
- Numeric columns
- Categorical columns

    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

---

### Numeric Pipeline

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

- Median imputation for missing values
- Standardization for stable logistic regression convergence

---

### Boolean Pipeline

    bool_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_int", FunctionTransformer(lambda a: a.astype(int))),
    ])

- Most frequent imputation
- Explicit conversion to integer (0/1)

---

### Categorical Pipeline

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

- Most frequent imputation
- One-hot encoding
- Safe handling of unseen categories

---

### Final Model

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )

Why logistic regression?

- Interpretable coefficients
- Odds ratio interpretation
- Appropriate baseline for medical risk modeling

---

## Train/Test Split

We use stratified splitting:

    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

This ensures:

- No data leakage
- Stable class proportions
- Honest evaluation on unseen data

---

## Model Evaluation

Metrics reported:

- Accuracy
- ROC AUC (primary metric)
- Classification report

Probabilities are converted to labels using a 0.5 threshold:

    preds = (proba >= 0.5).astype(int)

---

## Bootstrap AUC Estimation

To measure performance stability:

1. Resample training data with replacement
2. Refit the full preprocessing + model pipeline
3. Evaluate AUC on the fixed test set
4. Repeat 500 times

Key loop:

    idx = rng.randint(0, n, size=n)
    X_sample = X_train.iloc[idx]
    y_sample = y_train.iloc[idx]
    pipe.fit(X_sample, y_sample)

We compute:

- Mean AUC
- Standard deviation
- 95% confidence interval

This estimates sampling variability.

---

## Bootstrap Odds Ratios

We also estimate uncertainty in model coefficients.

For each bootstrap iteration:

    coefs = pipe.named_steps["model"].coef_[0]
    samples = np.exp(coefs)

We then compute:

- Mean odds ratio
- 95% confidence interval per feature

This allows identification of:

- Stable predictors
- Uncertain predictors
- Features whose effects may not be robust

---

## Output Files

The script exports:

- log_bs_summary_output.csv  
  Distribution of bootstrap ROC AUC values

- bootstrap_odds_ratios.csv  
  Bootstrap odds ratio confidence intervals

---

## Running the Script

From the project directory:

    python analysis_reg_bs.py

Dependencies:

    pandas
    numpy
    scikit-learn

---

## Design Principles

This project emphasizes:

- Reproducibility
- Proper preprocessing
- Separation of training and testing
- Explicit uncertainty estimation
- Interpretability over complexity

It intentionally avoids:

- Deep learning
- Black-box modeling
- Overengineering

The goal is disciplined applied statistical modeling.
