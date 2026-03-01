"""
reidentification_monitor.py
---------------------------
Feature 3: Per-Record Re-identification Risk Monitor

Gap addressed:
    Section 4.3 of Goldsteen et al. (2022) measures disclosure risk using
    Equation 6 as a single average across the whole dataset. That average
    hides outliers — a dataset where 990 records are safely generalised but
    10 are still fully unique would report a healthy average while those 10
    individuals remain completely re-identifiable. GDPR Article 5(1)(f)
    implies security must be appropriate to individual risk, not just a
    dataset-level statistic.

What this module adds:
    - compute_per_record_risk: applies Eq. 6 per record, not just as an average
    - print_risk_distribution: text histogram showing how risk is spread
    - flag_high_risk_records: filters records above a threshold
    - find_culprit_features: ranks which features cause uniqueness
    - suggest_targeted_generalisation: estimates how many people each fix helps
    - print_risk_report: master function that runs all of the above

Call sequence:
    print_risk_report() -> compute_per_record_risk() -> print_risk_distribution()
    -> flag_high_risk_records() -> find_culprit_features()
    -> suggest_targeted_generalisation()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_per_record_risk(df,quasi_identifiers):
    """
    Compute a re-identification risk score for every individual record.

    This extends Goldsteen et al. (2022) Equation 6 from a dataset average
    to a per-record value. The formula for each record r is:

        risk(r) = 1 / freq(r)

    where freq(r) is the number of records that share the same combination
    of quasi-identifier values as r.

    Interpretation:
        risk = 1.0  -> record is unique, fully re-identifiable
        risk = 0.5  -> shares pattern with one other person (50% chance)
        risk = 0.1  -> ten people share this pattern (relatively safe)
        risk = 0.01 -> hundred people share it (safe)

    Args:
        df (pd.DataFrame): the generalised dataset.
        quasi_identifiers (list): columns used as quasi-identifiers.

    Returns:
        Tuple of (risk_df with added columns, summary stats dict).
    """
    # Count how many records share each unique QI combination
    freq_table = df.groupby(quasi_identifiers).size().reset_index(name="equiv_class_size")

    # Merge the frequency count back so every record knows its group size
    risk_df = df.copy().reset_index(drop=True)
    risk_df = risk_df.merge(freq_table, on=quasi_identifiers, how="left")

    # risk = 1/freq — a larger group means lower individual risk
    risk_df["risk_score"] = 1.0 / risk_df["equiv_class_size"].astype(float)

    scores = risk_df["risk_score"]
    n      = len(risk_df)

    summary = {
        "n_records":          n,
        "mean_risk":          float(scores.mean()),
        "max_risk":           float(scores.max()),
        "min_risk":           float(scores.min()),
        "n_risk_100pct":      int((scores >= 1.0).sum()),
        "n_risk_above_50pct": int((scores >= 0.5).sum()),
        "n_risk_above_20pct": int((scores >= 0.2).sum()),
        "n_safe":             int((scores < 0.1).sum()),
        "pct_unique":         100.0 * (scores >= 1.0).sum() / n,
        "pct_safe":           100.0 * (scores < 0.1).sum()  / n,
    }

    print(f"\n[RiskMonitor] Per-record risk computed for {n} records")
    print(f"  Mean risk  : {summary['mean_risk']:.4f}")
    print(f"  Unique (k=1): {summary['n_risk_100pct']} ({summary['pct_unique']:.1f}%)")
    print(f"  Safe (<0.1) : {summary['n_safe']} ({summary['pct_safe']:.1f}%)")

    return risk_df, summary


def print_risk_distribution(risk_df):
    """
    Print a text histogram showing how risk is distributed across records.

    Rather than a single average, this shows the full picture — e.g. whether
    risk is spread evenly or concentrated in a tail of highly exposed individuals.
    A well-generalised dataset should have most records in the low-risk bands.

    Args:
        risk_df (pd.DataFrame): output of compute_per_record_risk(),
            must contain a 'risk_score' column.
    """
    print(f"\n[RiskMonitor] Risk Distribution")

    # Define risk bands from very safe to fully identifiable
    bands = [
        (0.0,  0.01, "k>100  very safe   "),
        (0.01, 0.05, "k=20-100  safe     "),
        (0.05, 0.10, "k=10-20  moderate  "),
        (0.10, 0.20, "k=5-10   caution   "),
        (0.20, 0.50, "k=2-5    risky     "),
        (0.50, 1.0,  "k=2      high risk "),
        (1.0,  1.01, "k=1      UNIQUE    "),
    ]

    scores = risk_df["risk_score"]
    n      = len(scores)

    for lo, hi, label in bands:
        if lo >= 1.0:
            count = int((scores >= lo).sum())
        else:
            count = int(((scores >= lo) & (scores < hi)).sum())

        pct     = 100.0 * count / n if n > 0 else 0.0
        bar_len = int(pct / 100.0 * 38)
        bar     = "#" * bar_len + "." * (38 - bar_len)
        flag    = " !" if lo >= 0.2 else ""
        print(f"  {label} |{bar}| {count:>5} ({pct:>5.1f}%){flag}")


def flag_high_risk_records(risk_df,threshold):
    """
    Return records whose risk score is at or above the threshold.

    A threshold of 0.2 means: anyone in a group of five or fewer is
    flagged. This aligns with the GDPR best practice of k >= 5.

    Common threshold values:
        1.0 = only fully unique records
        0.5 = groups of two or fewer (k < 2)
        0.2 = groups of five or fewer (k < 5, recommended)
        0.1 = groups of ten or fewer

    Args:
        risk_df (pd.DataFrame): output of compute_per_record_risk().
        threshold (float): risk score cutoff (default 0.2).

    Returns:
        Tuple of (high_risk_df DataFrame, n_flagged int).
    """
    if "risk_score" not in risk_df.columns:
        raise ValueError("risk_df must have a 'risk_score' column. Run compute_per_record_risk first.")

    high_risk_df = risk_df[risk_df["risk_score"] >= threshold].copy()
    n_flagged    = len(high_risk_df)
    pct          = 100.0 * n_flagged / len(risk_df) if len(risk_df) > 0 else 0.0

    print(f"\n[RiskMonitor] High-risk records (threshold={threshold}, k<={int(1/threshold)})")
    print(f"  Flagged: {n_flagged} ({pct:.1f}% of dataset)")

    if n_flagged > 0:
        unique_count = int((high_risk_df["risk_score"] >= 1.0).sum())
        pairs_count  = int(((high_risk_df["risk_score"] >= 0.5) &
                            (high_risk_df["risk_score"] < 1.0)).sum())
        print(f"    Fully unique (k=1) : {unique_count}")
        print(f"    Pairs only   (k=2) : {pairs_count}")
        if unique_count > 0:
            print(f"  WARNING: {unique_count} individuals are fully identifiable")
    else:
        print("  PASS — no records above risk threshold")

    return high_risk_df, n_flagged


def find_culprit_features(full_df,high_risk_df,quasi_identifiers):
    """
    Identify which features are most responsible for high-risk records.

    The culprit score for a feature is:

        culprit_score(f) = distinct_values_in_high_risk(f)
                           / distinct_values_in_full_dataset(f)

    A ratio near 1.0 means high-risk records use almost all of the feature's
    possible values — that feature contributes strongly to their uniqueness.
    A ratio near 0 means the feature is similar across high-risk records and
    is not the problem.

    Args:
        full_df (pd.DataFrame): the complete generalised dataset.
        high_risk_df (pd.DataFrame): the high-risk subset from flag_high_risk_records().
        quasi_identifiers (list): quasi-identifier column names.

    Returns:
        List of (feature_name, culprit_score) tuples, sorted by score descending.
    """
    if len(high_risk_df) == 0:
        print("\n[RiskMonitor] No high-risk records, skipping culprit analysis.")
        return []

    scores = []

    for feat in quasi_identifiers:
        if feat not in full_df.columns:
            continue
        n_full     = full_df[feat].nunique()
        n_highrisk = high_risk_df[feat].nunique()
        if n_full == 0:
            continue
        scores.append((feat, round(n_highrisk / n_full, 4)))

    # Highest ratio first = biggest contributor to uniqueness
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n[RiskMonitor] Culprit Feature Analysis")
    print(f"  (higher score = feature drives more uniqueness among high-risk records)")
    print(f"  {'Feature':<25} {'Score':>7}  {'All':>6}  {'High-risk':>10}")
    print("  " + "-" * 52)

    for feat, score in scores:
        n_full     = full_df[feat].nunique()
        n_highrisk = high_risk_df[feat].nunique()
        flag       = " <- top culprit" if feat == scores[0][0] else ""
        print(f"  {feat:<25} {score:>7.4f}  {n_full:>6}  {n_highrisk:>10}{flag}")

    return scores


def suggest_targeted_generalisation(high_risk_df,culprit_features,top_n: int = 3):
    """
    Recommend which features to generalise further to help the most people.

    For each top culprit feature, this counts how many high-risk records
    share their value with at least one other high-risk record. Those are
    the records most likely to benefit from broader generalisation of that
    feature, since merging nearby values could group them together.

    Args:
        high_risk_df (pd.DataFrame): high-risk records from flag_high_risk_records().
        culprit_features (list): output of find_culprit_features().
        top_n (int): how many features to recommend (default 3).

    Returns:
        List of recommendation strings.
    """
    if not culprit_features or len(high_risk_df) == 0:
        return ["No high-risk records — no targeted action needed."]

    recs = []
    print(f"\n[RiskMonitor] Targeted Generalisation Suggestions")

    for feat, score in culprit_features[:top_n]:
        if feat not in high_risk_df.columns:
            continue

        # Count how many distinct values could be merged (appear more than once)
        val_counts    = high_risk_df[feat].value_counts()
        mergeable     = int((val_counts > 1).sum())
        could_rescue  = int(val_counts[val_counts > 1].sum())
        n_distinct    = high_risk_df[feat].nunique()

        rec = (
            f"Broaden '{feat}' (score={score:.2f}): {n_distinct} distinct values among "
            f"{len(high_risk_df)} high-risk records. Merging {mergeable} value groups "
            f"could rescue up to {could_rescue} individuals into larger classes."
        )
        recs.append(rec)
        print(f"  -> {rec}")

    return recs


def print_risk_report(original_df,generalised_df,quasi_identifiers,threshold: float = 0.2):
    """
    Run a complete per-record risk report and print all results to terminal.

    This is the main entry point for Feature 3. It computes per-record risk
    for both the original and generalised datasets, prints a side-by-side
    comparison, flags who is still at risk, and recommends targeted fixes.

    Args:
        original_df (pd.DataFrame): the original ungeneralised dataset.
        generalised_df (pd.DataFrame): the generalised dataset from the toolkit.
        quasi_identifiers (list): quasi-identifier column names.
        threshold (float): risk score above which a record is flagged (default 0.2).

    Returns:
        dict with all results for use in the summary section.
    """
    print("\n" + "=" * 62)
    print("  PER-RECORD RE-IDENTIFICATION RISK REPORT")
    print(f"  Extends Goldsteen et al. (2022) Section 4.3 / Eq. 6")
    print("=" * 62)
    print(f"  Records    : {len(generalised_df)}")
    print(f"  Threshold  : {threshold}  (flags groups of {int(1/threshold)} or fewer)")

    # --- compute risk on original data for comparison ---
    print("\n-- Before generalisation --")
    orig_risk_df, orig_summary = compute_per_record_risk(original_df, quasi_identifiers)
    print_risk_distribution(orig_risk_df)

    # --- compute risk on generalised data ---
    print("-- After generalisation --")
    gen_risk_df, gen_summary = compute_per_record_risk(generalised_df, quasi_identifiers)
    print_risk_distribution(gen_risk_df)

    # --- flag, analyse, recommend ---
    high_risk_df, n_flagged = flag_high_risk_records(gen_risk_df, threshold)
    culprit_features        = find_culprit_features(generalised_df, high_risk_df, quasi_identifiers)
    recommendations         = suggest_targeted_generalisation(high_risk_df, culprit_features)

    # --- before vs after comparison ---
    unique_before  = orig_summary["n_risk_100pct"]
    unique_after   = gen_summary["n_risk_100pct"]
    unique_reduced = unique_before - unique_after
    pct_reduced    = 100.0 * unique_reduced / unique_before if unique_before > 0 else 0.0

    print(f"\n-- Comparison --")
    print(f"  {'Metric':<35} {'Before':>8}  {'After':>8}  {'Change':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Mean risk score':<35} {orig_summary['mean_risk']:>8.4f}  {gen_summary['mean_risk']:>8.4f}  "
          f"{gen_summary['mean_risk']-orig_summary['mean_risk']:>+8.4f}")
    print(f"  {'Unique records (k=1)':<35} {unique_before:>8}  {unique_after:>8}  "
          f"{unique_after-unique_before:>+8}")
    print(f"  {'Safe records (risk<0.1)':<35} {orig_summary['n_safe']:>8}  "
          f"{gen_summary['n_safe']:>8}  {gen_summary['n_safe']-orig_summary['n_safe']:>+8}")

    if unique_reduced > 0:
        print(f"\n  Generalisation removed {unique_reduced} unique records ({pct_reduced:.1f}% reduction)")
    else:
        print("\n  WARNING: generalisation did not reduce unique records")

    # --- verdict ---
    print("\n-- Verdict --")
    if n_flagged == 0:
        print(f"  PASS — no records above risk threshold {threshold}")
    elif unique_after == 0:
        print(f"  PARTIAL — no unique records, but {n_flagged} in small groups")
    else:
        print(f"  FAIL — {unique_after} records still fully identifiable")
        print("  Apply the targeted generalisation suggestions above")

    print("=" * 62)

    return {
        "orig_summary":     orig_summary,
        "gen_summary":      gen_summary,
        "high_risk_records": high_risk_df,
        "n_flagged":        n_flagged,
        "culprit_features": culprit_features,
        "recommendations":  recommendations,
        "unique_reduced":   unique_reduced,
    }
