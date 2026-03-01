"""
demo_security_features.py
--------------------------
End-to-end demonstration of three security features built on top of the
IBM ai-privacy-toolkit (Goldsteen et al., 2022).

The script:
  1. Loads the real UCI Adult dataset from the UCI repository URL.
  2. Trains an MLPClassifier target model.
  3. Runs GeneralizeToRepresentative from the toolkit with target_accuracy=0.75.
     This threshold is intentionally set below the tree's initial accuracy
     (~0.80) so the toolkit accepts the initial tree generalisation without
     entering its de-generalisation improvement loop.
  4. Computes accuracy retained and NCP from actual toolkit output.
  5. Runs all three security features and prints a summary.

How to run:
    pip install ai-privacy-toolkit scikit-learn pandas numpy
    python demo_security_features.py

Dataset:
    UCI Adult (Census Income), same dataset used in Goldsteen et al. (2022).
    Downloaded automatically from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    Falls back to synthetic data if the download fails.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sensitivity_weighter import (
    SensitivityProfile,
    compute_weighted_ncp,
    get_removal_priority,
    print_sensitivity_report,
)
from privacy_auditor import print_audit_report
from reidentification_monitor import print_risk_report

try:
    from apt.minimization import GeneralizeToRepresentative
    TOOLKIT = True
    print("ai-privacy-toolkit found — will use GeneralizeToRepresentative.")
except ImportError:
    TOOLKIT = False
    print("ai-privacy-toolkit not installed. Install with: pip install ai-privacy-toolkit")
    print("Falling back to simple quantisation for this run.\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(n_samples=5000):
    """
    Load the UCI Adult (Census Income) dataset from the official UCI URL.

    The Adult dataset is the same one used in Goldsteen et al. (2022), making
    results directly comparable to the paper. It contains demographic features
    and a binary income label (<=50K or >50K per year).

    We keep the nine features the paper uses and drop fnlwgt (census weight),
    education (string form of education-num), relationship, and native-country.

    Categorical features are label-encoded to integers so the neural network
    can process them. The income column sometimes has a trailing period in
    the raw file, which is stripped before mapping to 0/1.

    Args:
        n_samples (int): rows to use. Stratified sampling preserves class ratio.

    Returns:
        pd.DataFrame with columns: age, education_num, hours_per_week,
        capital_gain, capital_loss, marital_status, occupation, race, sex, income.
    """
    UCI_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )

    col_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]

    try:
        print(f"Downloading dataset from {UCI_URL} ...")
        raw = pd.read_csv(
            UCI_URL, header=None, names=col_names,
            na_values=" ?", skipinitialspace=True,
        )
        raw.dropna(inplace=True)
        raw.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(raw)} rows from UCI repository.")
    except Exception as err:
        print(f"Download failed: {err}")
        print("Using synthetic fallback data instead.\n")
        return _synthetic_fallback(n_samples)

    numeric     = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
    categorical = ["marital-status", "occupation", "race", "sex"]

    df = raw[numeric + categorical].copy()

    for col in categorical:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str).str.strip())

    # strip trailing period that some versions of the file have
    df["income"] = (
        raw["income"].astype(str).str.strip().str.rstrip(".")
        .map({"<=50K": 0, ">50K": 1}).fillna(0).astype(int)
    )

    df.rename(columns={
        "education-num":  "education_num",
        "hours-per-week": "hours_per_week",
        "capital-gain":   "capital_gain",
        "capital-loss":   "capital_loss",
        "marital-status": "marital_status",
    }, inplace=True)

    if n_samples < len(df):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
        idx, _ = next(sss.split(df.drop("income", axis=1), df["income"]))
        df = df.iloc[idx].reset_index(drop=True)

    print(f"Using {len(df)} rows after stratified sampling.")
    return df


def _synthetic_fallback(n):
    """
    Generate a synthetic Adult-style dataset used when the real data
    cannot be downloaded (e.g. no internet access).

    The structure mirrors the real dataset: same feature names, similar
    value ranges, and a deterministic income label.

    Args:
        n (int): number of rows to generate.

    Returns:
        pd.DataFrame with the same columns as the real dataset.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "age":            np.random.randint(18, 90, n),
        "education_num":  np.random.randint(1, 16, n),
        "hours_per_week": np.random.randint(15, 65, n),
        "capital_gain":   np.abs(np.random.normal(500, 3000, n)).astype(int),
        "capital_loss":   np.abs(np.random.normal(50, 500, n)).astype(int),
        "marital_status": np.random.randint(0, 7, n),
        "occupation":     np.random.randint(0, 14, n),
        "race":           np.random.randint(0, 5, n),
        "sex":            np.random.randint(0, 2, n),
    })
    df["income"] = (
        ((df["education_num"] > 10) & (df["hours_per_week"] > 40))
        | (df["capital_gain"] > 3000)
    ).astype(int)
    print(f"Generated {n} synthetic rows.")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(df):
    """
    Train a two-layer MLP on the dataset and report baseline accuracy.

    The 70/30 stratified train/test split preserves class balance. We return
    both the numpy arrays and a DataFrame version of the test set so that
    audit modules receive the correct rows (not a misaligned slice of the
    full dataset).

    Args:
        df (pd.DataFrame): loaded dataset with an 'income' column.

    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test, feature_names, df_test)
        where df_test is a DataFrame of the test rows with the income column.
    """
    features = [c for c in df.columns if c != "income"]
    X = df[features].values.astype(float)
    y = df["income"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Keep a DataFrame of the test rows for the audit comparisons.
    # We cannot use df.iloc[:len(X_test)] because train_test_split shuffles
    # rows — the first 1500 rows of df are NOT the same as X_test.
    df_test = pd.DataFrame(X_test, columns=features)
    df_test["income"] = y_test

    print("\nTraining MLPClassifier...")
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    baseline_acc = model.score(X_test, y_test)
    print(f"Baseline accuracy on test set: {baseline_acc:.2%}")

    return model, X_train, X_test, y_train, y_test, features, df_test


# ---------------------------------------------------------------------------
# Generalisation
# ---------------------------------------------------------------------------

def generalise(model, X_train, y_train, X_test, feature_names):
    """
    Generalise X_test using GeneralizeToRepresentative from the toolkit.

    How target_accuracy works in the toolkit:
        GeneralizeToRepresentative first builds a decision tree that groups
        similar records into generalisation cells. It then measures how well
        the generalised data preserves the original model's predictions
        (not ground-truth accuracy).

    NCP is computed from the actual toolkit output using the representative
    mapping method: for each unique representative value in a feature, we find
    all original values that mapped to it and measure their range. That range
    divided by the total domain gives the bin's NCP contribution.

    Args:
        model: trained sklearn classifier.
        X_train, y_train: training data for fitting the generaliser.
        X_test: test data to transform.
        feature_names (list): column names for each feature dimension.

    Returns:
        Tuple of (df_gen DataFrame, per_feature_ncp dict, acc_retained float).
    """
    if TOOLKIT:
        print("\nRunning GeneralizeToRepresentative (ai-privacy-toolkit)...")
        gen = GeneralizeToRepresentative(model, target_accuracy=0.81)
        gen.fit(X_train, y_train)
        X_gen = gen.transform(X_test)
    else:
        print("\nRunning quantisation fallback (toolkit not installed)...")
        X_gen = _quantise(X_test)

    # Check whether the toolkit actually generalised anything.
    # If all features match the original exactly, it means the toolkit
    # removed all generalisations (target_accuracy was set too high).
    features_changed = sum(
        1 for i in range(X_test.shape[1])
        if not np.allclose(X_test[:, i], X_gen[:, i])
    )
    if features_changed == 0:
        print("\nWARNING: No features were generalised (output = original data).")
        print("  This means target_accuracy was set above the tree's initial")
        print("  accuracy, causing the toolkit to remove all generalisations.")
        print("  Consider lowering target_accuracy further.")

    # Accuracy retained = fraction of records where model prediction is
    # identical on generalised vs original data (prediction agreement).
    preds_orig   = model.predict(X_test)
    preds_gen    = model.predict(X_gen)
    acc_retained = float(np.mean(preds_orig == preds_gen))
    print(f"Accuracy retained (prediction agreement): {acc_retained:.2%}")
    print(f"Features actually generalised: {features_changed}/{X_test.shape[1]}")

    # Compute NCP using the representative mapping method
    per_feature_ncp = _compute_ncp_from_representatives(X_test, X_gen, feature_names)

    # Build a DataFrame for the audit modules
    df_gen = pd.DataFrame(X_gen, columns=feature_names)
    df_gen["income"] = preds_orig   # original predictions as the label

    return df_gen, per_feature_ncp, acc_retained


def _quantise(X):
    """
    Simple fallback quantiser used when the toolkit is not available.

    Groups each feature into sqrt(n_unique) equal-width bins and replaces
    each value with its bin centre. Produces coarser generalisation than the
    toolkit's decision-tree approach.

    Args:
        X (np.ndarray): feature matrix to quantise.

    Returns:
        np.ndarray of the same shape with values replaced by bin centres.
    """
    X_gen = X.copy().astype(float)
    for i in range(X.shape[1]):
        col    = X[:, i]
        n_bins = max(3, int(np.sqrt(len(np.unique(col)))))
        bins   = np.linspace(col.min(), col.max(), n_bins + 1)
        idx    = np.clip(np.digitize(col, bins[:-1]) - 1, 0, n_bins - 1)
        X_gen[:, i] = ((bins[:-1] + bins[1:]) / 2.0)[idx]
    return X_gen


def _compute_ncp_from_representatives(X_orig, X_gen, feature_names):
    """
    Compute per-feature NCP using the representative mapping in X_gen.

    The toolkit replaces each original value with a representative value
    for its generalisation cell (e.g. all ages 30-40 become 35). The gap
    between representative values is NOT the bin width, so using it directly
    as NCP would be wrong.

    Correct approach (per Goldsteen et al. 2022 Eq. 3):
        For each unique representative value r in X_gen[:, i]:
            - Find all records whose generalised value is r
            - Look at their ORIGINAL values: orig_min, orig_max
            - bin_ncp = (orig_max - orig_min) / domain
        NCP(feature_i) = weighted average of bin_ncp across all cells
                         (weighted by number of records in each cell)

    A feature left unchanged by the toolkit (all records keep original value)
    gets NCP = 0.

    Args:
        X_orig (np.ndarray): original feature values.
        X_gen  (np.ndarray): generalised values from the toolkit.
        feature_names (list): column name for each feature index.

    Returns:
        dict mapping feature name to NCP value in [0, 1].
    """
    ncp = {}

    for i, feat in enumerate(feature_names):
        col_orig = X_orig[:, i]
        col_gen  = X_gen[:, i]
        domain   = col_orig.max() - col_orig.min()

        # Feature with no variation or left unchanged by the toolkit
        if domain == 0 or np.allclose(col_orig, col_gen):
            ncp[feat] = 0.0
            continue

        total_ncp   = 0.0
        total_count = 0

        # For each unique representative value, measure the range of original
        # values that were mapped to it. That range / domain = bin NCP.
        for rep_val in np.unique(col_gen):
            mask     = col_gen == rep_val
            bin_orig = col_orig[mask]
            bin_width = bin_orig.max() - bin_orig.min()
            bin_ncp   = bin_width / domain

            count       = mask.sum()
            total_ncp   += bin_ncp * count   # weighted by bin population
            total_count += count

        ncp[feat] = round(total_ncp / total_count, 4) if total_count > 0 else 0.0

    return ncp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ai-privacy-toolkit Security Extensions Demo")
    print("  Goldsteen et al. (2022) - Data Minimization for GDPR")
    print("=" * 60)

    # Step 1: load data and train model
    df = load_data(n_samples=10000)
    feature_names = [c for c in df.columns if c != "income"]
    model, X_train, X_test, y_train, y_test, _, df_test= train_model(df)

    # Step 2: generalise using the toolkit (or fallback)
    df_gen, per_feature_ncp, acc_retained = generalise(
        model, X_train, y_train, X_test, feature_names
    )

    # df_test contains the correct original test rows (matched to X_test).
    # We use this as the "before generalisation" baseline in Features 2 and 3.
    # NOTE: df.iloc[:len(X_test)] would be WRONG because train_test_split
    # shuffles rows — the first 1500 rows of df are not the test set.
    df_orig = df_test.copy()

    # -------------------------------------------------------------------
    # Feature 1: Sensitivity-Weighted Generalisation
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("FEATURE 1: Sensitivity-Weighted Generalisation")
    print("-" * 60)
    print("Gap: Section 5.1 of the paper describes a weighted NCP")
    print("variant but never implements it. We add it here so that")
    print("GDPR Article 9 attributes (race, sex) are scored as more")
    print("important than non-sensitive features like hours-per-week.")

    profile = SensitivityProfile.from_tiers({
        "race":           "critical",   # GDPR Article 9 special category
        "sex":            "critical",   # GDPR Article 9 special category
        "age":            "high",       # strong quasi-identifier
        "marital_status": "high",
        "education_num":  "medium",
        "occupation":     "medium",
        "hours_per_week": "low",
        "capital_gain":   "low",
        "capital_loss":   "low",
    })

    std_ncp  = float(np.mean(list(per_feature_ncp.values())))
    w_ncp, _ = compute_weighted_ncp(per_feature_ncp, profile)

    print(f"\nStandard NCP (unweighted average) : {std_ncp:.4f}")
    print(f"Sensitivity-weighted NCP          : {w_ncp:.4f}")
    print()
    print("Interpreting the gap:")
    print("  Standard NCP weights every feature equally — race and hours-per-week")
    print("  contribute the same amount regardless of how sensitive they are.")
    print("  Weighted NCP multiplies each feature's NCP by its sensitivity weight")
    print("  before averaging, so GDPR Article 9 attributes (race, sex, weight=8)")
    print("  pull the score much more than low-sensitivity features (weight=1).")
    print()

    rel_diff = (w_ncp - std_ncp) / std_ncp if std_ncp > 0 else 0.0

    if std_ncp == 0.0:
        print("  INCONCLUSIVE: all NCP values are 0 — no generalisation was applied.")
        print("  Check that target_accuracy is below the toolkit's initial tree accuracy.")
    elif rel_diff > 0.05:
        print(f"  RESULT: w_ncp ({w_ncp:.4f}) > std_ncp ({std_ncp:.4f})")
        print("  Sensitive features received MORE generalisation than average.")
        print("  This is the desired GDPR Article 9 outcome — race and sex are better")
        print("  protected than the unweighted score would suggest.")
        print("  The weighted score confirms compliance that standard NCP cannot show.")
    elif rel_diff < -0.05:
        print(f"  RESULT: w_ncp ({w_ncp:.4f}) < std_ncp ({std_ncp:.4f})")
        print("  Sensitive features received LESS generalisation than average.")
        print("  Standard NCP hides this gap entirely. This is a compliance risk.")
        print("  Recommendation: increase accuracy loss budget so race and sex")
        print("  receive broader generalisation ranges.")
    else:
        print(f"  RESULT: w_ncp ({w_ncp:.4f}) ≈ std_ncp ({std_ncp:.4f})")
        print("  Generalisation was uniform — sensitivity was not a factor.")
        print("  Sensitive features received no extra protection over low-sensitivity ones.")

    # Accuracy gains per feature used for removal priority ordering.
    # In a real deployment these would come from the toolkit's own ILAG
    # calculation; here we use small uniform values to demonstrate ordering.
    rng       = np.random.default_rng(42)
    acc_gains = {f: float(rng.uniform(0.01, 0.04)) for f in per_feature_ncp}
    get_removal_priority(per_feature_ncp, acc_gains, profile)

    print_sensitivity_report(per_feature_ncp, profile, acc_retained)

    # -------------------------------------------------------------------
    # Feature 2: k-Anonymity & Homogeneity Attack Audit
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("FEATURE 2: k-Anonymity & Homogeneity Attack Audit")
    print("-" * 60)
    print("Gap: Section 4.3 measures disclosure risk (Eq. 6) but never")
    print("checks k-anonymity or l-diversity, and does not detect")
    print("homogeneity attacks. We add a formal post-generalisation audit.")

    audit_results = print_audit_report(
        original_df=df_orig,
        generalised_df=df_gen,
        quasi_identifiers=feature_names,
        sensitive_attribute="income",
        target_k=5,
        target_l=2,
    )

    # -------------------------------------------------------------------
    # Feature 3: Per-Record Re-identification Risk Monitor
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("FEATURE 3: Per-Record Re-identification Risk Monitor")
    print("-" * 60)
    print("Gap: Eq. 6 gives one average risk score for the whole dataset.")
    print("An average hides individuals who are still fully identifiable.")
    print("We compute risk per person and flag outliers.")

    risk_results = print_risk_report(
        original_df=df_orig,
        generalised_df=df_gen,
        quasi_identifiers=feature_names,
        threshold=0.2,
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Accuracy retained          : {acc_retained:.2%}")
    print(f"  Standard NCP               : {std_ncp:.4f}")
    print(f"  Weighted NCP               : {w_ncp:.4f}")
    print(f"  k-Anonymity                : {audit_results['k_anonymity']}")
    print(f"  l-Diversity                : {audit_results['l_diversity']}")
    print(f"  Disclosure risk reduction  : {audit_results['risk_reduction_pct']:.1f}%")
    print(f"  Unique records before      : {risk_results['orig_summary']['n_risk_100pct']}")
    print(f"  Unique records after       : {risk_results['gen_summary']['n_risk_100pct']}")
    print(f"  High-risk records flagged  : {risk_results['n_flagged']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
