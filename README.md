# Heart Disease Risk Modeling with Bootstrap Uncertainty

## Overview

This repository trains an interpretable heart disease classifier using logistic regression and reports uncertainty using bootstrap resampling.

What you get from this project:

- A leakage-safe preprocessing + modeling pipeline (scikit-learn Pipeline + ColumnTransformer)
- Honest test-set evaluation (Accuracy + ROC AUC + classification report)
- Bootstrap distribution of ROC AUC (mean / std / 95% CI)
- Bootstrap odds ratios with confidence intervals (interpretability + stability)
- Results saved as CSV artifacts for reporting

This is intentionally a “small but rigorous” applied healthcare modeling project.

---

## Repository Layout

Expected structure:

    heart-disease-bootstrap/
    ├── src/
    │   ├── analysis_reg_bs.py
    │   └── make_snapshot.py              (optional; generates metrics_summary.json)
    ├── data/
    │   ├── heart_disease_uci.csv         (NOT committed; you download via link)
    │   └── README.txt                    (explains how to download data)
    ├── results/
    │   ├── log_bs_summary_output.csv
    │   ├── bootstrap_odds_ratios.csv
    │   └── metrics_summary.json          (optional)
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Dataset

The script expects the [UCI Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data):

    data/heart_disease_uci.csv

Download:

    mkdir -p data
    curl -L "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data" -o data/heart_disease_uci.csv

After download, verify:

    ls -la data/
    head -n 5 data/heart_disease_uci.csv

---

## Quickstart

### 1) Create a virtual environment (recommended)

    python3 -m venv .venv
    source .venv/bin/activate

### 2) Install dependencies

    pip install -r requirements.txt

### 3) Download the dataset into data/

    follow previous instructions

### 4) Run the analysis

    python src/analysis_reg_bs.py

---

## What the Code Does

### 1) Load + target engineering

- Loads CSV with parsing of boolean tokens:
  - true_values: TRUE / True / true
  - false_values: FALSE / False / false

- Creates binary target:
  
    target = 1 if num > 0 else 0

- Drops columns that should not be used as features:
  - num (original label)
  - id (identifier)
  - dataset (constant in this subset)

### 2) Preprocessing (ColumnTransformer)

Columns are split by dtype:

- Numeric columns (int64, float64)
  - SimpleImputer(median)
  - StandardScaler

- Boolean columns (bool, boolean)
  - SimpleImputer(most_frequent)
  - Convert True/False to 0/1 (FunctionTransformer)

- Categorical columns (object)
  - SimpleImputer(most_frequent)
  - OneHotEncoder(handle_unknown="ignore")

ColumnTransformer applies each preprocessing pipeline to its columns and concatenates them into a single numeric matrix.

### 3) Model (Logistic Regression)

The model is:

    LogisticRegression(max_iter=3000, class_weight="balanced")

This is a strong baseline for healthcare risk modeling because it is interpretable and stable.

### 4) Evaluation

- Train/test split (80/20) with stratification
- Metrics on the TEST set:
  - Accuracy
  - ROC AUC (primary)
  - Classification report

### 5) Bootstrap Uncertainty Quantification (Performance and Effects)

This project quantifies uncertainty in both (i) predictive performance and (ii) estimated feature effects by nonparametric bootstrap resampling of the training set. The test set is kept fixed throughout to provide a consistent evaluation target across bootstrap refits.

#### 5.1 Bootstrap ROC AUC (Performance Stability)

Objective:
Estimate the sampling variability of ROC AUC under repeated draws from the empirical training distribution.

Method:
For each bootstrap iteration (N = 500 by default):

1. Sample, with replacement, an index set of size n = |X_train| from the training data.
2. Refit the full pipeline on the bootstrap sample:
   - imputers
   - scaler
   - one-hot encoder
   - logistic regression model
3. Compute predicted probabilities on the fixed test set.
4. Compute ROC AUC on the fixed test labels.
5. Store the AUC value.

Summary statistics reported:
- Mean bootstrap AUC
- Standard deviation of bootstrap AUC
- 95% percentile interval (2.5th and 97.5th percentiles)

Artifact:
- results/log_bs_summary_output.csv
  Contains the per-iteration bootstrap AUC values used to compute the summary.

Interpretation:
The bootstrap AUC distribution provides an empirical approximation of performance variability due to sampling. The 95% percentile interval provides a practical uncertainty interval for expected discrimination on data drawn from a similar population.

---

#### 5.2 Bootstrap Odds Ratios (Effect Stability)

Objective:
Estimate uncertainty in logistic regression coefficients and derived odds ratios under repeated draws from the empirical training distribution.

Background:
In logistic regression, each feature j has coefficient β_j. Exponentiating yields the odds ratio:

    OR_j = exp(β_j)

OR_j represents a multiplicative change in the odds of the positive class for a one-unit increase in the feature (for continuous features) or for presence vs absence (for binary/one-hot features), holding other variables constant. Note that for standardized numeric features, “one unit” corresponds to one standard deviation of the original feature.

Method:
For each bootstrap iteration (N = 500 by default):

1. Sample, with replacement, an index set of size n = |X_train|.
2. Refit the full pipeline on the bootstrap sample.
3. Extract the fitted logistic regression coefficient vector from:
      pipe.named_steps["model"].coef_[0]
4. Map coefficients to the post-transformation feature space using:
      pipe.named_steps["preprocess"].get_feature_names_out()
5. Convert coefficients to odds ratios via exp(β).
6. Store the odds ratio for each transformed feature.

Summary statistics reported per feature:
- Mean odds ratio across bootstrap refits
- 95% percentile interval (2.5th and 97.5th percentiles)

Artifact:
- results/bootstrap_odds_ratios.csv

Columns:
- feature: transformed feature name (after preprocessing; e.g., one-hot expanded columns)
- odds_ratio_mean: mean exp(coef) across bootstrap iterations
- ci_low: 2.5th percentile of exp(coef)
- ci_high: 97.5th percentile of exp(coef)

Console output:
The script prints the top rows of the same odds-ratio table (“Top Odds Ratios”), corresponding to the highest mean odds ratios.

Interpretation:
- OR > 1 suggests a positive association with the outcome (increased odds).
- OR < 1 suggests a negative association (decreased odds).
- A confidence interval spanning 1.0 indicates that the direction/magnitude of the association is not stable across bootstrap resamples, given the model specification and dataset size.

Caveats:
- Odds ratios are conditional on the model design (included covariates, encoding, scaling, and regularization).
- One-hot encoded features represent comparisons to an implicit baseline defined by the encoding/regularization configuration.
- Bootstrap intervals here are empirical percentile intervals and should be interpreted as practical uncertainty measures rather than strict parametric confidence intervals.


## License

Educational and portfolio use.
Dataset credit: UCI Machine Learning Repository.
