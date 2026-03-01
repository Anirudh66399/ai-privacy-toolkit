"""
privacy_auditor.py
------------------
Feature 2: k-Anonymity & Homogeneity Attack Auditor

Gap addressed:
    Section 4.3 of Goldsteen et al. (2022) measures disclosure risk using
    Equation 6 but never checks whether the output satisfies formal privacy
    guarantees. Specifically, it does not verify k-anonymity (Sweeney, 2002),
    l-diversity (Machanavajjhala et al., 2007), or detect homogeneity attacks
    — where all records in a group share the same sensitive value, letting an
    attacker infer it without identifying any individual.

What this module adds:
    - compute_k_anonymity: minimum equivalence class size
    - detect_homogeneity_attacks: finds classes with one sensitive value
    - compute_l_diversity: minimum distinct sensitive values per class
    - compute_disclosure_risk: reproduces Equation 6 from the paper
    - suggest_remediation: actionable fixes based on audit results
    - print_audit_report: master function that runs all checks

Call sequence:
    print_audit_report() -> compute_disclosure_risk(),
    compute_k_anonymity(), detect_homogeneity_attacks(),
    compute_l_diversity(), suggest_remediation()
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple


def compute_k_anonymity(df,quasi_identifiers):
    """
    Compute the k-anonymity level of a generalised dataset.

    k-anonymity (Sweeney 2002) requires every record to be
    indistinguishable from at least k-1 others on quasi-identifier
    columns. A low k means individuals can be re-identified.
    GDPR best practice recommends k >= 5.

    Method: group by all quasi-identifier columns, take the size of
    each group, and return the minimum — that is k.

    Args:
        df (pd.DataFrame): the generalised dataset.
        quasi_identifiers (list): column names used as quasi-identifiers.

    Returns:
        Tuple of (k_value int, stats dict).
    """
    groups = df.groupby(quasi_identifiers).size()

    k_value    = int(groups.min())
    n_unique   = int((groups == 1).sum())

    stats = {
        "k_anonymity":       k_value,
        "max_class_size":    int(groups.max()),
        "mean_class_size":   float(groups.mean()),
        "num_classes":       len(groups),
        "num_unique":        n_unique,
        "pct_unique":        100.0 * n_unique / len(df),
    }

    print(f"\n[Auditor] k-Anonymity Check")
    print(f"  k = {k_value}  (min equivalence class size)")
    print(f"  Unique records (k=1): {n_unique} ({stats['pct_unique']:.1f}%)")
    print(f"  Total classes: {stats['num_classes']}")

    if k_value < 2:
        print("  WARNING: k=1 — some records are fully re-identifiable!")
    elif k_value < 5:
        print(f"  WARNING: k={k_value} is below the recommended k >= 5")
    else:
        print(f"  PASS: k={k_value} meets the recommended threshold")

    return k_value, stats


def detect_homogeneity_attacks(df,quasi_identifiers,sensitive_attribute):
    """
    Find equivalence classes where all records share the same sensitive value.

    A homogeneity attack (Machanavajjhala et al. 2007) exploits the fact
    that k-anonymity alone does not protect attribute disclosure. If every
    record in a group has income=High, an attacker who knows someone belongs
    to that group learns their income without identifying them.

    Method: for each equivalence class, count distinct values of the
    sensitive attribute using Counter. If only one distinct value exists,
    the class is homogeneous and vulnerable.

    Args:
        df (pd.DataFrame): the generalised dataset.
        quasi_identifiers (list): quasi-identifier column names.
        sensitive_attribute (str): the sensitive column to check.

    Returns:
        Tuple of (list of vulnerable class dicts, vulnerability rate float).
    """
    if sensitive_attribute not in df.columns:
        raise ValueError(f"Column '{sensitive_attribute}' not found in DataFrame.")

    vulnerable = []
    groups     = df.groupby(quasi_identifiers)

    for group_key, group_df in groups:
        counts     = Counter(group_df[sensitive_attribute].tolist())
        class_size = len(group_df)

        # Only one distinct sensitive value — attacker can infer it for free
        if len(counts) == 1:
            sensitive_val = list(counts.keys())[0]
            risk = "CRITICAL" if class_size >= 50 else (
                   "HIGH"     if class_size >= 10 else "MEDIUM")

            vulnerable.append({
                "group_key":       group_key,
                "size":            class_size,
                "sensitive_value": sensitive_val,
                "risk":            risk,
            })

    vuln_rate = len(vulnerable) / len(groups) if len(groups) > 0 else 0.0

    print(f"\n[Auditor] Homogeneity Attack Detection")
    print(f"  Sensitive attribute : '{sensitive_attribute}'")
    print(f"  Vulnerable classes  : {len(vulnerable)} of {len(groups)} ({vuln_rate*100:.1f}%)")

    critical_count = sum(1 for v in vulnerable if v["risk"] == "CRITICAL")
    if critical_count:
        print(f"  CRITICAL classes (>=50 records): {critical_count}")

    if not vulnerable:
        print("  PASS — no homogeneity attacks detected")
    else:
        print("  FAIL — remediation recommended (see suggest_remediation)")

    return vulnerable, vuln_rate


def compute_l_diversity(df,quasi_identifiers,sensitive_attribute):
    """
    Compute the l-diversity level of the generalised dataset.

    l-diversity (Machanavajjhala et al. 2007) requires each equivalence
    class to have at least l distinct sensitive values. l=1 means at
    least one class is fully homogeneous. l >= 2 is the minimum safe level.

    Method: for each group count unique values of the sensitive column
    using nunique(), then take the minimum across all groups.

    Args:
        df (pd.DataFrame): the generalised dataset.
        quasi_identifiers (list): quasi-identifier column names.
        sensitive_attribute (str): the sensitive column.

    Returns:
        Tuple of (l_value int, stats dict).
    """
    diversity_per_class = [
        int(group[sensitive_attribute].nunique())
        for _, group in df.groupby(quasi_identifiers)
    ]

    l_value = min(diversity_per_class) if diversity_per_class else 0

    stats = {
        "l_diversity":     l_value,
        "max_diversity":   max(diversity_per_class) if diversity_per_class else 0,
        "mean_diversity":  float(np.mean(diversity_per_class)) if diversity_per_class else 0.0,
        "num_homogeneous": sum(1 for d in diversity_per_class if d == 1),
    }

    print(f"\n[Auditor] l-Diversity Check")
    print(f"  l = {l_value}  (min distinct sensitive values per class)")
    print(f"  Homogeneous classes: {stats['num_homogeneous']}")

    if l_value >= 3:
        print(f"  PASS: l={l_value} — good diversity")
    elif l_value == 2:
        print("  WARNING: l=2 — minimal diversity, consider entropy l-diversity")
    else:
        print("  FAIL: l=1 — homogeneity present, dataset is vulnerable")

    return l_value, stats


def compute_disclosure_risk(df,quasi_identifiers):
    """
    Compute the disclosure risk score from Goldsteen et al. (2022) Eq. 6.

    Risk = (1/n) * sum_r( 1/freq(r) )

    where freq(r) is the number of records sharing the same quasi-identifier
    combination as record r. A unique record contributes 1.0 (maximum risk).
    A record in a group of 100 contributes 0.01.

    Args:
        df (pd.DataFrame): dataset to evaluate.
        quasi_identifiers (list): quasi-identifier column names.

    Returns:
        float: disclosure risk in [0, 1].
    """
    # Build a frequency table for each unique QI combination
    freq_table = df.groupby(quasi_identifiers).size().reset_index(name="_freq")

    # Merge frequency back onto every record
    merged = df.merge(freq_table, on=quasi_identifiers, how="left")
    risk   = float((1.0 / merged["_freq"]).sum()) / len(df)

    print(f"\n[Auditor] Disclosure Risk (Eq. 6): {risk:.4f}")
    if risk > 0.5:
        print("  HIGH risk — many records have unique QI patterns")
    elif risk > 0.1:
        print("  MODERATE risk")
    else:
        print("  LOW risk")

    return risk


def suggest_remediation(k_value,l_value,vuln_rate,target_k,target_l):
    """
    Generate remediation suggestions based on audit results.

    Args:
        k_value (int): measured k-anonymity level.
        l_value (int): measured l-diversity level.
        vuln_rate (float): fraction of classes that are homogeneous.
        target_k (int): minimum acceptable k (default 5).
        target_l (int): minimum acceptable l (default 2).

    Returns:
        List of human-readable suggestion strings.
    """
    suggestions = []

    if k_value < target_k:
        suggestions.append(
            f"[k-Anonymity] k={k_value} is below target k={target_k}. "
            "Increase the allowed accuracy loss so the minimiser creates "
            "broader ranges and merges small equivalence classes."
        )
    if l_value < target_l:
        suggestions.append(
            f"[l-Diversity] l={l_value} is below target l={target_l}. "
            "Exclude the sensitive attribute from quasi-identifiers, or "
            "reduce decision-tree depth to split clusters differently."
        )
    if vuln_rate > 0.3:
        suggestions.append(
            f"[Homogeneity] {vuln_rate*100:.0f}% of classes are homogeneous. "
            "Consider adding the sensitive attribute as a splitting criterion."
        )
    if not suggestions:
        suggestions.append("No remediation needed — all targets met.")

    print(f"\n[Auditor] Remediation Suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

    return suggestions


def print_audit_report(original_df,generalised_df,quasi_identifiers,sensitive_attribute,target_k,target_l):
    """
    Run a complete privacy audit and print a PASS/FAIL report.

    This is the main entry point for Feature 2. It calls each individual
    check function in sequence and then produces a combined verdict.

    Args:
        original_df (pd.DataFrame): the original ungeneralised dataset.
        generalised_df (pd.DataFrame): the generalised dataset from the toolkit.
        quasi_identifiers (list): quasi-identifier column names.
        sensitive_attribute (str): the sensitive column to audit.
        target_k (int): minimum acceptable k-anonymity level.
        target_l (int): minimum acceptable l-diversity level.

    Returns:
        dict with all audit results for programmatic use.
    """
    print("\n" + "=" * 62)
    print("  PRIVACY AUDIT REPORT")
    print("=" * 62)
    print(f"  Rows            : {len(generalised_df)}")
    print(f"  Quasi-ids       : {quasi_identifiers}")
    print(f"  Sensitive attr  : '{sensitive_attribute}'")

    # Baseline on original data
    print("\n-- Baseline (original data) --")
    orig_risk = compute_disclosure_risk(original_df, quasi_identifiers)

    # Checks on generalised data
    print("\n-- Generalised data --")
    k_val, k_stats          = compute_k_anonymity(generalised_df, quasi_identifiers)
    gen_risk                = compute_disclosure_risk(generalised_df, quasi_identifiers)
    vuln_classes, vuln_rate = detect_homogeneity_attacks(
        generalised_df, quasi_identifiers, sensitive_attribute
    )
    l_val, l_stats          = compute_l_diversity(
        generalised_df, quasi_identifiers, sensitive_attribute
    )

    risk_reduction_pct = 100.0 * (orig_risk - gen_risk) / orig_risk if orig_risk > 0 else 0.0
    print(f"\n  Disclosure risk: {orig_risk:.4f} -> {gen_risk:.4f} "
          f"(reduced by {risk_reduction_pct:.1f}%)")

    suggestions = suggest_remediation(k_val, l_val, vuln_rate, target_k, target_l)

    # Final verdict
    passes_k    = k_val    >= target_k
    passes_l    = l_val    >= target_l
    passes_homo = vuln_rate <= 0.1
    all_pass    = passes_k and passes_l and passes_homo

    print(f"\n-- Verdict --")
    print(f"  k-Anonymity (k>={target_k}): {'PASS' if passes_k else 'FAIL'}  (k={k_val})")
    print(f"  l-Diversity (l>={target_l}): {'PASS' if passes_l else 'FAIL'}  (l={l_val})")
    print(f"  Homogeneity (<=10%): {'PASS' if passes_homo else 'FAIL'}  ({vuln_rate*100:.1f}% vulnerable)")
    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL -- remediation required'}")
    print("=" * 62)

    return {
        "k_anonymity":           k_val,
        "l_diversity":           l_val,
        "disclosure_risk_orig":  orig_risk,
        "disclosure_risk_gen":   gen_risk,
        "risk_reduction_pct":    risk_reduction_pct,
        "vulnerable_classes":    vuln_classes,
        "vulnerability_rate":    vuln_rate,
        "passes_all":            all_pass,
        "remediation":           suggestions,
    }
