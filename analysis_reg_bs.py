import pandas as pd
import numpy as np

# Train/test split (avoid evaluating on the same data you trained on)
from sklearn.model_selection import train_test_split

# ColumnTransformer + Pipeline let you:
# - apply different preprocessing to numeric vs categorical columns
# - keep everything reproducible and refit correctly inside bootstrap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# OneHotEncoder converts strings like "Male" / "flat" / "asymptomatic" to numeric columns
# StandardScaler helps logistic regression when numeric features have different scales
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# SimpleImputer fills missing values so the model doesn’t crash
from sklearn.impute import SimpleImputer

# Logistic regression model
from sklearn.linear_model import LogisticRegression

# Bootstrap sampling with replacement
from sklearn.utils import resample

# Metrics: accuracy is OK but ROC AUC is usually better for medical risk
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


DATA_PATH = "heart_disease_uci.csv"


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame.

    Note:
    - If your file contains '?' as missing values (some UCI variants do),
      use: pd.read_csv(path, na_values=["?"])
    """
    return pd.read_csv(path,
        true_values=["TRUE", "True", "true"],
        false_values=["FALSE", "False", "false"])




def add_target_and_drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Create binary target from num > 0
    2) Drop non-feature columns (num itself, id, dataset if present/constant)
    3) Convert TRUE/FALSE strings to 1/0 for boolean columns
    """
    df = df.copy()  # never mutate original df in place in a pipeline script

    # ---- target engineering ----
    # "num" is the UCI severity label; we binarize it:
    df["target"] = (df["num"] > 0).astype(int)

    # ---- drop columns we shouldn't model ----
    drop_cols = ["num"]  # remove original multi-class target
    if "id" in df.columns:
        drop_cols.append("id")  # identifier, not a feature
    if "dataset" in df.columns:
        # in your sample it's always "Cleveland", so it carries no information
        drop_cols.append("dataset")

    df = df.drop(columns=drop_cols)

    return df


def basic_overview(df: pd.DataFrame) -> None:
    """Print size + column types (diagnostics)."""
    print("=== DATASET SHAPE ===")
    print(df.shape)
    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)


def missing_analysis(df: pd.DataFrame) -> None:
    """Show missing values per column (diagnostics)."""
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())


def outcome_distribution(df: pd.DataFrame) -> None:
    """Check class balance after target creation."""
    print("\n=== OUTCOME DISTRIBUTION (target) ===")
    print(df["target"].value_counts(normalize=True))


def statistical_summary(df: pd.DataFrame) -> None:
    """
    Summary including categorical columns too.
    include="all" helps you see category counts, uniques, etc.
    """
    print("\n=== SUMMARY (NUMERIC + CATEGORICAL) ===")
    print(df.describe(include="all"))


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build a preprocessing + model pipeline:
    - numeric: impute median + scale
    - categorical: impute most frequent + one-hot encode
    - logistic regression on top

    This is the key fix: it turns your mixed-type DataFrame into a numeric matrix.
    """
    # Identify boolean vs numeric vs categorical columns from the DataFrame dtype
    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Numeric preprocessing:
    # - fill missing values with median
    # - standardize features (mean 0, std 1)
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Boolean preprocessing:
    # - impute with most frequent (median makes no sense for all-missing bools)
    # - convert to int (True/False -> 1/0) after imputation
    bool_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_int", FunctionTransformer(lambda a: a.astype(int))),
    ])
    


    # Categorical preprocessing:
    # - fill missing values with most frequent category
    # - one-hot encode (creates 0/1 columns for each category)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ColumnTransformer applies different preprocessors to different column sets
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("bool", bool_pipe, bool_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )

    # Logistic regression:
    # - class_weight balanced helps if target classes are not 50/50
    # - max_iter increased to help convergence after one-hot
    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    # Full pipeline
    return Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])


def fit_and_eval(df: pd.DataFrame):
    """
    Train/test split + fit the pipeline + evaluate on the TEST set.
    Returns everything needed for bootstrap.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    # Fixed split for honest evaluation and reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keeps same class proportion in train/test
    )

    # Build and fit pipeline (preprocess + logistic regression)
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # Predict probabilities (needed for AUC)
    proba = pipe.predict_proba(X_test)[:, 1]

    # Convert probabilities to class labels using a 0.5 threshold (simple baseline)
    preds = (proba >= 0.5).astype(int)

    # Print test metrics (NOT training metrics)
    print("\n=== TEST METRICS ===")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("ROC AUC :", round(roc_auc_score(y_test, proba), 4))
    print("\n" + classification_report(y_test, preds, digits=3))

    return pipe, X_train, y_train, X_test, y_test


def bootstrap_auc(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iterations: int = 500,
    random_state: int = 42
) -> np.ndarray:
    """
    Bootstrap AUC:
    - resample TRAIN set with replacement
    - refit pipeline on each bootstrap sample (important!)
    - evaluate AUC on the SAME fixed test set (honest comparison)
    - return distribution of AUCs so you can compute CI
    """
    rng = np.random.RandomState(random_state)
    aucs = []

    n = len(X_train)

    for _ in range(n_iterations):
        # Sample indices with replacement (size n)
        idx = rng.randint(0, n, size=n)

        # Subset rows; .iloc keeps alignment between X and y
        X_sample = X_train.iloc[idx]
        y_sample = y_train.iloc[idx]

        # Fit on bootstrap sample (refit preprocessors + model!)
        pipe.fit(X_sample, y_sample)

        # Evaluate on fixed test set
        proba = pipe.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))

    aucs = np.array(aucs)

    # Summary stats + CI
    mean = float(aucs.mean())
    std = float(aucs.std(ddof=1))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])

    print("\n=== BOOTSTRAP ROC AUC ===")
    print("Mean :", round(mean, 4))
    print("Std  :", round(std, 4))
    print("95% CI:", (round(ci_low, 4), round(ci_high, 4)))

    return aucs

def bootstrap_odds_ratios(
    pipe,
    X_train,
    y_train,
    n_iterations=500,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    n = len(X_train)

    coef_samples = []

    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        X_sample = X_train.iloc[idx]
        y_sample = y_train.iloc[idx]

        pipe.fit(X_sample, y_sample)

        coefs = pipe.named_steps["model"].coef_[0]
        coef_samples.append(coefs)

    coef_samples = np.array(coef_samples)

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()

    results = []

    for i, name in enumerate(feature_names):
        samples = np.exp(coef_samples[:, i])  # odds ratios
        mean = samples.mean()
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

        results.append({
            "feature": name,
            "odds_ratio_mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values("odds_ratio_mean", ascending=False)

def main():
    # Load raw data
    df = load_data(DATA_PATH)

    # Create target, drop id/dataset/num, map TRUE/FALSE -> 1/0
    df = add_target_and_drop_cols(df)

    # Dataset diagnostics
    basic_overview(df)
    missing_analysis(df)
    statistical_summary(df)
    outcome_distribution(df)

    # Fit + evaluate on test split
    pipe, X_train, y_train, X_test, y_test = fit_and_eval(df)

    # Bootstrap uncertainty on AUC
    aucs = bootstrap_auc(pipe, X_train, y_train, X_test, y_test, n_iterations=500)

    or_df = bootstrap_odds_ratios(pipe, X_train, y_train)
    or_df.to_csv("bootstrap_odds_ratios.csv", index=False)
    print("\nTop Odds Ratios:")
    print(or_df.head(10))

    # Save bootstrap distribution as an artifact (useful for reports)
    pd.DataFrame({"bootstrap_auc": aucs}).to_csv("log_bs_summary_output.csv", index=False)
    print("\nSaved bootstrap AUCs to log_bs_summary_output.csv")


if __name__ == "__main__":
    main()
