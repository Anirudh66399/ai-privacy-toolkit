# ai-privacy-toolkit — Security Extensions

**Paper:** Goldsteen, A., Ezov, G., Shmelkin, R., Moffie, M., & Farkash, A. (2022). Data minimization for GDPR compliance in machine learning models. *AI and Ethics*, 2, 477–491.

**Original repo:** https://github.com/IBM/ai-privacy-toolkit

---

## Overview

This project extends the IBM ai-privacy-toolkit with three security features. The toolkit generalises ML input features at runtime (e.g. replacing an exact age with a 10-year range) so less precise personal data is collected while keeping model accuracy within a chosen threshold, supporting GDPR Article 5(1)(c). Each feature addresses a specific gap in the original paper.

---

## Features and Security Mechanisms

**Feature 1 — Sensitivity-Weighted Generalisation** (`sensitivity_weighter.py`): Section 5.1 describes a weighted NCP variant but never implements it. Without weights, race is treated identically to hours-per-week in the privacy score, violating GDPR Article 9. This feature assigns sensitivity tiers (low/medium/high/critical) to each feature. `compute_weighted_ncp()` multiplies each feature's NCP by its normalised sensitivity weight before averaging, so GDPR Article 9 attributes (race, sex, weight=8) influence the score far more than low-sensitivity features (weight=1). The result is interpreted directionally: if weighted NCP > standard NCP, sensitive features were generalised more aggressively than average — the desired GDPR Article 9 outcome. If weighted NCP < standard NCP, sensitive features are underprotected and standard NCP hides the gap entirely. `get_removal_priority()` reorders de-generalisation using `weight / ILAG(f)`, ensuring critical features are de-generalised last when accuracy needs recovering. Features with NCP=0 are excluded from the removal queue and listed separately as candidates for future generalisation. NCP values are computed from the actual toolkit output using a representative-mapping method: for each unique representative value in a generalised feature, the range of original values that mapped to it is measured, and `(orig_max - orig_min) / domain` gives that bin's NCP contribution, weighted by record count. `print_sensitivity_report()` outputs a full per-feature breakdown with a directional verdict.

Call sequence: `SensitivityProfile.from_tiers()` → `compute_weighted_ncp()` → `get_removal_priority()` → `print_sensitivity_report()`

**Feature 2 — k-Anonymity & Homogeneity Attack Auditor** (`privacy_auditor.py`): Section 4.3 measures disclosure risk with Equation 6 but never checks k-anonymity (Sweeney, 2002) or l-diversity (Machanavajjhala et al., 2007), and does not detect homogeneity attacks — where all records in a group share the same sensitive value, letting an attacker infer it without identifying anyone. `compute_k_anonymity()` finds the smallest equivalence class via `groupby().size().min()`. `detect_homogeneity_attacks()` iterates groups and flags any where `Counter(sensitive_values)` has only one key. `compute_l_diversity()` uses `nunique()` per group. `print_audit_report()` orchestrates all checks and prints a PASS/FAIL verdict. The audit compares a matched original test set against the generalised output — both drawn from the same `train_test_split` indices — so the before/after comparison reflects the same individuals.

Call sequence: `print_audit_report()` → internally calls `compute_k_anonymity()`, `compute_disclosure_risk()`, `detect_homogeneity_attacks()`, `compute_l_diversity()`, `suggest_remediation()`

**Feature 3 — Per-Record Re-identification Risk Monitor** (`reidentification_monitor.py`): Equation 6 gives one average risk score, which can hide fully identifiable individuals. GDPR Article 5(1)(f) implies risk must be assessed at the individual level. `compute_per_record_risk()` applies `1/freq(r)` to every record by merging the `groupby().size()` frequency table back onto each row. `flag_high_risk_records()` filters records above a threshold (default 0.2, meaning groups of five or fewer). `find_culprit_features()` computes `distinct_in_high_risk / distinct_in_full` per feature to rank which features cause uniqueness. `suggest_targeted_generalisation()` estimates how many flagged records each targeted fix would rescue.

Call sequence: `print_risk_report()` → `compute_per_record_risk()` → `print_risk_distribution()` → `flag_high_risk_records()` → `find_culprit_features()` → `suggest_targeted_generalisation()`

---

## How to Run

```bash
# Method 1
git clone https://github.com/IBM/ai-privacy-toolkit
cd ai-privacy-toolkit
# copy sensitivity_weighter.py, privacy_auditor.py,
# reidentification_monitor.py, demo_security_features.py, requirements.txt here and follow the next steps

# Method 2
# Download the updated git, open a terminal in that folder and follow the next steps


# Please note if using windows run the below commands in CMD.
python -m venv .venv
.\.venv\Scripts\Activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install ai-privacy-toolkit
python demo_security_features.py
```

The demo loads the UCI Adult dataset directly from the UCI repository URL. On first run it downloads ~3 MB and requires an internet connection; if offline it falls back to synthetic data automatically so all three features can still be demonstrated.

All output prints to the terminal — no extra configuration is required.

---

## References

- Goldsteen et al. (2022). *AI and Ethics*, 2, 477–491.
- Sweeney (2002). k-anonymity. *IJUFKS*, 10, 557–570.
- Machanavajjhala et al. (2007). l-diversity. *ACM TKDD*, 1(1).
- GDPR Articles 5(1)(c), 5(1)(f), 9.
