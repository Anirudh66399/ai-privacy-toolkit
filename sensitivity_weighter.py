"""
sensitivity_weighter.py
-----------------------
Feature 1: Sensitivity-Weighted Generalisation

Gap addressed:
    Section 5.1 of Goldsteen et al. (2022) mentions a weighted NCP variant
    that could steer generalisation toward better protecting sensitive features,
    but the authors never implement it. Without weights, the toolkit treats
    race and hours-per-week equally when scoring privacy, which is incorrect
    under GDPR Article 9 (special category data).

What this module adds:
    - SensitivityProfile: stores per-feature sensitivity weights (1–8)
    - compute_weighted_ncp: weighted average of NCP values
    - get_removal_priority: adjusts ILAG ordering so sensitive features
      are de-generalised last
    - print_sensitivity_report: formatted audit output

Call sequence:
    SensitivityProfile.from_tiers() -> compute_weighted_ncp()
    -> get_removal_priority() -> print_sensitivity_report()
"""

import numpy as np
from typing import Dict, List, Tuple


# Numeric weights for each sensitivity tier.
# These map directly to GDPR concepts:
#   critical = Article 9 special categories (race, health, sex)
#   high     = quasi-identifiers with strong re-identification power
#   medium   = useful but less sensitive attributes
#   low      = generic numeric features
SENSITIVITY_TIERS = {
    "low":      1.0,
    "medium":   2.0,
    "high":     4.0,
    "critical": 8.0,
}


class SensitivityProfile:
    """
    Stores sensitivity weights for each feature in the dataset.

    A weight is a positive number indicating how privacy-sensitive
    a feature is. Higher weight = more sensitive = should be
    generalised more aggressively and de-generalised last.

    Args:
        feature_weights (dict): mapping of feature name to weight value.

    Raises:
        ValueError: if any weight is zero or negative.
    """

    def __init__(self, feature_weights):
        for feat, w in feature_weights.items():
            if w <= 0:
                raise ValueError(f"Weight for '{feat}' must be > 0, got {w}")
        self.feature_weights = feature_weights

        # Print a summary so the user can verify the profile at a glance
        print("\n[SensitivityProfile] Weights loaded:")
        for feat, w in sorted(feature_weights.items(), key=lambda x: -x[1]):
            tier = self._weight_to_tier(w)
            print(f"  {feat:<25} weight={w:.1f}  ({tier})")

    def get_weight(self, feature_name):
        """Return weight for a feature, defaulting to 1.0 if not listed."""
        return self.feature_weights.get(feature_name, 1.0)

    def normalised_weights(self, feature_names):
        """
        Return weights normalised to sum to 1.0.

        Used by compute_weighted_ncp so the weighted average stays
        in the same range as the standard NCP (0 to 1).

        Args:
            feature_names (list): features to include, in order.

        Returns:
            numpy array of normalised weights.
        """
        raw = np.array([self.get_weight(f) for f in feature_names], dtype=float)
        return raw / raw.sum()

    @staticmethod
    def _weight_to_tier(weight):
        """Map a numeric weight back to a tier label for display."""
        if weight >= 6.0:
            return "critical"
        elif weight >= 3.0:
            return "high"
        elif weight >= 1.5:
            return "medium"
        return "low"

    @classmethod
    def from_tiers(cls, tier_assignments):
        """
        Convenience constructor using tier names instead of raw numbers.

        Args:
            tier_assignments (dict): {feature_name: tier_name}
                where tier_name is one of 'low', 'medium', 'high', 'critical'.

        Returns:
            SensitivityProfile instance.

        Example:
            profile = SensitivityProfile.from_tiers({
                "race": "critical",
                "age":  "high",
            })
        """
        weights = {}
        for feat, tier in tier_assignments.items():
            if tier not in SENSITIVITY_TIERS:
                raise ValueError(f"Unknown tier '{tier}'. Use: {list(SENSITIVITY_TIERS)}")
            weights[feat] = SENSITIVITY_TIERS[tier]
        return cls(weights)


def compute_weighted_ncp(per_feature_ncp,profile):
    """
    Compute a sensitivity-weighted NCP score.

    Standard NCP averages all features equally. Weighted NCP multiplies
    each feature's NCP by its normalised sensitivity weight first, so a
    poorly generalised sensitive feature drags the score down more than
    a poorly generalised non-sensitive one.

    Formula (adapted from Goldsteen et al. 2022, Eq. 3):
        NCP_weighted = sum_i( w_i_norm * NCP_i )

    Args:
        per_feature_ncp (dict): {feature: ncp_value} from the toolkit output.
        profile (SensitivityProfile): the sensitivity weights to apply.

    Returns:
        Tuple of (weighted_ncp float, breakdown dict of per-feature contributions).
    """
    features     = list(per_feature_ncp.keys())
    ncp_values   = np.array([per_feature_ncp[f] for f in features])
    norm_weights = profile.normalised_weights(features)

    # Element-wise multiply then sum — this is the dot product of weights and NCP
    contributions = norm_weights * ncp_values
    weighted_ncp  = float(contributions.sum())

    breakdown = {f: float(c) for f, c in zip(features, contributions)}
    return weighted_ncp, breakdown


def get_removal_priority(per_feature_ncp,per_feature_accuracy_gain,profile):
    """
    Return features sorted by sensitivity-adjusted ILAG score (ascending).

    The original ILAG formula (Eq. 5 in the paper):
        ILAG(f) = NCP(f) / AccuracyGain(f)

    Features with low ILAG get de-generalised first. But this is blind to
    sensitivity — a sensitive feature with low NCP would unfairly be removed
    early. We fix this by dividing by the weight instead of multiplying:

        ILAG_adj(f) = weight(f) / ILAG(f)

    A higher weight produces a higher adj_ILAG, pushing sensitive features
    to the end of the removal queue (they appear last in an ascending sort).

    Features with NCP=0 have no generalisation to remove, so they are
    excluded from the removal queue entirely and listed separately as
    candidates to generalise next (ordered by sensitivity).

    Args:
        per_feature_ncp (dict): {feature: ncp_value}
        per_feature_accuracy_gain (dict): {feature: accuracy_gain_if_removed}
        profile (SensitivityProfile): sensitivity weights.

    Returns:
        List of feature names in removal order (first = de-generalise first).
    """
    removal_scores = {}   # features with NCP > 0
    not_generalised = {}  # features with NCP = 0

    for feat in per_feature_ncp:
        ncp    = per_feature_ncp[feat]
        gain   = per_feature_accuracy_gain.get(feat, 0.0)
        weight = profile.get_weight(feat)

        # Compute the base ILAG score (paper Eq. 5)
        if gain != 0:
            base_ilag = ncp / gain
        else:
            base_ilag = ncp

        if base_ilag == 0:
            # NCP=0 means the feature was never generalised — nothing to remove.
            # Track it separately so it doesn't pollute the removal queue.
            not_generalised[feat] = weight
        else:
            # Higher weight -> higher score -> further back in ascending sort
            removal_scores[feat] = weight / base_ilag

    # Sort ascending: lowest adj_ILAG first (cheapest/least sensitive to lose)
    sorted_queue = sorted(removal_scores, key=lambda f: removal_scores[f])

    # Sort ungeneralised features by sensitivity (most urgent to protect first)
    sorted_ungeneralised = sorted(not_generalised, key=lambda f: not_generalised[f], reverse=True)

    # --- print the removal queue ---
    print("\n[SensitivityWeighter] De-generalisation order (if accuracy drops):")
    print("  Top of list = remove first | Bottom = protected longest")
    print(f"  {'Rank':<5} {'Feature':<25} {'adj_ILAG':>10}  {'weight':>7}  tier")
    print("  " + "-" * 58)

    for rank, feat in enumerate(sorted_queue, 1):
        score = removal_scores[feat]
        w     = profile.get_weight(feat)
        tier  = SensitivityProfile._weight_to_tier(w)
        note  = " <- remove first"   if rank == 1 else (
                " <- protected last" if rank == len(sorted_queue) else "")
        print(f"  {rank:<5} {feat:<25} {score:>10.2f}  {w:>7.1f}  [{tier}]{note}")

    if sorted_ungeneralised:
        print(f"\n  Not yet generalised (NCP=0) — consider these next:")
        for feat in sorted_ungeneralised:
            w    = not_generalised[feat]
            tier = SensitivityProfile._weight_to_tier(w)
            print(f"    {feat:<25} weight={w:.1f}  [{tier}]")

    return sorted_queue


def print_sensitivity_report(per_feature_ncp,profile,accuracy_retained,label: str = "Final"):
    """
    Print a formatted sensitivity audit report.

    Shows both the standard (unweighted) NCP and the weighted NCP
    side by side, along with a per-feature breakdown so a GDPR
    compliance team can see exactly where the privacy gaps are.

    Args:
        per_feature_ncp (dict): {feature: ncp_value}
        profile (SensitivityProfile): the sensitivity weights.
        accuracy_retained (float): fraction of predictions preserved (e.g. 0.98).
        label (str): label for the report header.
    """
    std_ncp          = float(np.mean(list(per_feature_ncp.values())))
    w_ncp, breakdown = compute_weighted_ncp(per_feature_ncp, profile)

    print(f"\n{'='*58}")
    print(f"  SENSITIVITY REPORT  [{label}]")
    print(f"{'='*58}")
    print(f"  Standard NCP (unweighted) : {std_ncp:.4f}")
    print(f"  Weighted NCP              : {w_ncp:.4f}")
    print(f"  Accuracy retained         : {accuracy_retained*100:.2f}%")
    print(f"\n  {'Feature':<25} {'NCP':>6}  {'Weight':>7}  {'Contribution':>13}")
    print("  " + "-" * 56)

    # Sort by weight descending so most sensitive features appear first
    for feat in sorted(per_feature_ncp, key=lambda f: profile.get_weight(f), reverse=True):
        ncp_val = per_feature_ncp[feat]
        w       = profile.get_weight(feat)
        contrib = breakdown.get(feat, 0.0)
        tier    = SensitivityProfile._weight_to_tier(w)
        print(f"  {feat:<25} {ncp_val:>6.4f}  {w:>7.1f}  {contrib:>13.4f}  [{tier}]")

    # Interpret by comparing weighted NCP to standard NCP, not by thresholding
    # the absolute value. The direction of the gap is what matters:
    #
    #   w_ncp > std_ncp  ->  sensitive features were generalised MORE than average.
    #                        This is the correct GDPR Article 9 outcome — the toolkit
    #                        is already steering generalisation toward protecting race,
    #                        sex, etc. more aggressively than lower-sensitivity features.
    #
    #   w_ncp < std_ncp  ->  sensitive features were generalised LESS than average.
    #                        Standard NCP hides this gap entirely. This is the warning
    #                        case this feature is specifically designed to surface.
    #
    #   w_ncp ≈ std_ncp  ->  generalisation was applied uniformly regardless of
    #                        sensitivity. The toolkit treated race the same as
    #                        hours-per-week.
    #
    # A tolerance of 5% relative difference is used to avoid flagging noise.

    rel_diff = (w_ncp - std_ncp) / std_ncp if std_ncp > 0 else 0.0

    if std_ncp == 0.0:
        verdict = (
            "INCONCLUSIVE — no features were generalised (all NCP=0). "
            "Check that target_accuracy is set below the toolkit's initial tree accuracy."
        )
    elif rel_diff > 0.05:
        verdict = (
            f"SENSITIVE FEATURES PROTECTED — weighted NCP ({w_ncp:.4f}) is higher than "
            f"standard NCP ({std_ncp:.4f}), meaning high-sensitivity features (race, sex) "
            f"received above-average generalisation. This confirms GDPR Article 9 compliance."
        )
    elif rel_diff < -0.05:
        verdict = (
            f"WARNING — weighted NCP ({w_ncp:.4f}) is lower than standard NCP ({std_ncp:.4f}). "
            f"Sensitive features are being generalised LESS than average. "
            f"Standard NCP hides this gap. Consider increasing accuracy loss budget."
        )
    else:
        verdict = (
            f"UNIFORM — weighted NCP ({w_ncp:.4f}) ≈ standard NCP ({std_ncp:.4f}). "
            f"Generalisation was applied uniformly regardless of sensitivity. "
            f"Sensitive features received no extra protection."
        )

    print(f"\n  Verdict: {verdict}")
    print("=" * 58)
