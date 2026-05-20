"""Recommendation generator for DrugOracle."""

from typing import Dict, List, Optional


_ADMET_THRESHOLDS = {
    "herg": {"bad": 0.5, "warn": 0.35, "label": "hERG inhibition", "severity": "high",
             "suggestion": "Reduce LogP or remove basic amines",
             "improvement": "Lower cardiotoxicity risk"},
    "ames": {"bad": 0.5, "warn": 0.35, "label": "Ames mutagenicity", "severity": "high",
             "suggestion": "Reduce electrophilicity or remove alerting groups",
             "improvement": "Lower mutagenicity"},
    "dili": {"bad": 0.5, "warn": 0.35, "label": "DILI (liver injury)", "severity": "high",
             "suggestion": "Reduce reactive metabolites or lipophilicity",
             "improvement": "Lower hepatotoxicity"},
    "bioavailability_ma": {"low": 0.5, "label": "Oral bioavailability", "severity": "medium",
                           "suggestion": "Add polar groups or reduce MW",
                           "improvement": "Improve oral exposure"},
    "bbb_martins": {"low": 0.3, "label": "BBB penetration", "severity": "low",
                    "suggestion": "Increase lipophilicity or reduce TPSA if CNS target",
                    "improvement": "Improve CNS exposure"},
    "clearance_hepatocyte_az": {"bad": 80, "warn": 50, "label": "Hepatocyte clearance",
                                "severity": "medium",
                                "suggestion": "Block metabolic soft spots or add steric shielding",
                                "improvement": "Increase metabolic stability"},
    "ppbr_az": {"bad": 95, "warn": 85, "label": "Plasma protein binding", "severity": "low",
                "suggestion": "Reduce lipophilicity to free more unbound drug",
                "improvement": "Increase free fraction"},
}


def generate_recommendations(
    admet_preds: Dict[str, float],
    alerts: List[str],
    prev_admet: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    recs: List[Dict] = []

    for alert in alerts:
        recs.append({
            "type": "Structural Alert",
            "issue": alert,
            "suggestion": "Modify or replace substructure to eliminate the alerting motif",
            "severity": "high",
            "expected_improvement": "Reduce toxicity risk",
        })

    for key, cfg in _ADMET_THRESHOLDS.items():
        val = admet_preds.get(key)
        if val is None:
            continue

        if key == "bioavailability_ma":
            if val < cfg["low"]:
                recs.append({
                    "type": "Bioavailability",
                    "issue": f"Low predicted {cfg['label']} ({val:.0%})",
                    "suggestion": cfg["suggestion"],
                    "severity": cfg["severity"],
                    "expected_improvement": cfg["improvement"],
                })
            elif val >= 0.7:
                recs.append({
                    "type": "Strength",
                    "issue": f"Good {cfg['label']} ({val:.0%})",
                    "suggestion": "Maintain current scaffold features",
                    "severity": "positive",
                    "expected_improvement": "Oral route viable",
                })
            continue

        if key in ("bbb_martins",):
            if val < cfg.get("low", 0):
                recs.append({
                    "type": "ADMET",
                    "issue": f"Low {cfg['label']} ({val:.0%})",
                    "suggestion": cfg["suggestion"],
                    "severity": cfg["severity"],
                    "expected_improvement": cfg["improvement"],
                })
            continue

        if key in ("clearance_hepatocyte_az", "ppbr_az"):
            if val > cfg.get("bad", 999):
                recs.append({
                    "type": "ADMET",
                    "issue": f"High {cfg['label']} ({val:.1f})",
                    "suggestion": cfg["suggestion"],
                    "severity": cfg["severity"],
                    "expected_improvement": cfg["improvement"],
                })
            elif val > cfg.get("warn", 999):
                recs.append({
                    "type": "ADMET",
                    "issue": f"Borderline {cfg['label']} ({val:.1f})",
                    "suggestion": cfg["suggestion"],
                    "severity": "low",
                    "expected_improvement": cfg["improvement"],
                })
            continue

        if val > cfg.get("bad", 999):
            recs.append({
                "type": "Safety",
                "issue": f"{cfg['label']} risk ({val:.0%})",
                "suggestion": cfg["suggestion"],
                "severity": cfg["severity"],
                "expected_improvement": cfg["improvement"],
            })
        elif val > cfg.get("warn", 999):
            recs.append({
                "type": "Safety",
                "issue": f"Borderline {cfg['label']} ({val:.0%})",
                "suggestion": cfg["suggestion"],
                "severity": "medium",
                "expected_improvement": cfg["improvement"],
            })
        elif key in ("herg", "ames", "dili") and val < 0.2:
            recs.append({
                "type": "Strength",
                "issue": f"Low {cfg['label']} risk ({val:.0%})",
                "suggestion": "Maintain current safety profile",
                "severity": "positive",
                "expected_improvement": "Clean toxicity signal",
            })

    if prev_admet:
        improved = []
        regressed = []
        for k in admet_preds:
            if k not in prev_admet:
                continue
            delta = admet_preds[k] - prev_admet[k]
            is_lower_better = k in ("herg", "ames", "dili", "clearance_hepatocyte_az", "ppbr_az")
            if is_lower_better:
                if delta < -0.05:
                    improved.append(k)
                elif delta > 0.05:
                    regressed.append(k)
            else:
                if delta > 0.05:
                    improved.append(k)
                elif delta < -0.05:
                    regressed.append(k)
        if improved:
            recs.append({
                "type": "Progress",
                "issue": f"Improved: {', '.join(improved)}",
                "suggestion": "Continue optimizing in this structural direction",
                "severity": "positive",
                "expected_improvement": "Compound trending toward drug-like space",
            })
        if regressed:
            recs.append({
                "type": "Regression",
                "issue": f"Worsened: {', '.join(regressed)}",
                "suggestion": "Consider reverting recent structural changes for these endpoints",
                "severity": "medium",
                "expected_improvement": "Recover lost ADMET performance",
            })

    if not recs:
        recs.append({
            "type": "Status",
            "issue": "No critical flags detected",
            "suggestion": "Molecule shows acceptable predicted ADMET profile",
            "severity": "positive",
            "expected_improvement": "Proceed to further optimization or synthesis",
        })

    return recs
