"""Reward functions for SafeMolGen RL fine-tuning."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import QED
RDLogger.DisableLog("rdApp.error")


def validity_reward(smiles: str) -> float:
    return 1.0 if Chem.MolFromSmiles(smiles) is not None else 0.0


def qed_reward(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(QED.qed(mol))


def diversity_reward(smiles_list: List[str]) -> float:
    unique = len(set(smiles_list))
    return unique / max(len(smiles_list), 1)


def _oracle_scalar(
    fn: Optional[Callable[[str], Union[float, Dict[str, float], None]]],
    smiles: str,
    phase_weights: Optional[Tuple[float, float, float]] = None,
) -> float:
    out = fn(smiles) if fn else None
    if out is None:
        return 0.0
    if isinstance(out, dict) and phase_weights is not None:
        p1 = out.get("phase1", 0.0)
        p2 = out.get("phase2", 0.0)
        p3 = out.get("phase3", 0.0)
        return phase_weights[0] * p1 + phase_weights[1] * p2 + phase_weights[2] * p3
    return float(out)


def _scalar_from_prediction(
    pred: Any,
    phase_weights: Optional[Tuple[float, float, float]] = None,
) -> float:
    if pred is None:
        return 0.0
    if isinstance(pred, dict):
        if phase_weights is not None:
            p1 = pred.get("phase1_prob", pred.get("phase1", 0.0))
            p2 = pred.get("phase2_prob", pred.get("phase2", 0.0))
            p3 = pred.get("phase3_prob", pred.get("phase3", 0.0))
            return phase_weights[0] * p1 + phase_weights[1] * p2 + phase_weights[2] * p3
        return float(pred.get("overall_prob", 0.0))
    if phase_weights is not None:
        return (
            phase_weights[0] * getattr(pred, "phase1_prob", 0.0)
            + phase_weights[1] * getattr(pred, "phase2_prob", 0.0)
            + phase_weights[2] * getattr(pred, "phase3_prob", 0.0)
        )
    return float(getattr(pred, "overall_prob", 0.0))


def _alert_penalty(pred: Any) -> int:
    if pred is None:
        return 0
    alerts = getattr(pred, "structural_alerts", None) or (pred.get("structural_alerts", []) if isinstance(pred, dict) else [])
    risks = getattr(pred, "risk_factors", None) or (pred.get("risk_factors", []) if isinstance(pred, dict) else [])
    return len(alerts) + len(risks)


def compute_reward_per_smiles(
    smiles: str,
    oracle_score_fn: Optional[Callable[[str], Union[float, Dict[str, float], None]]] = None,
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
    validity_gated_oracle: bool = True,
    phase_weights: Optional[Tuple[float, float, float]] = None,
    oracle_prediction_fn: Optional[Callable[[str], Any]] = None,
    w_alert: float = 0.0,
) -> float:
    validity = validity_reward(smiles)
    qed = qed_reward(smiles)
    if oracle_prediction_fn is not None and w_alert != 0:
        pred = oracle_prediction_fn(smiles)
        oracle_raw = _scalar_from_prediction(pred, phase_weights) - w_alert * _alert_penalty(pred)
    else:
        oracle_raw = _oracle_scalar(oracle_score_fn, smiles, phase_weights) if oracle_score_fn else 0.0
    oracle = oracle_raw * validity if validity_gated_oracle else oracle_raw
    return w_validity * validity + w_qed * qed + w_oracle * oracle


def compute_rewards_per_sample(
    smiles_list: List[str],
    oracle_score_fn: Optional[Callable[[str], Union[float, Dict[str, float], None]]] = None,
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
    w_diversity: float = 0.1,
    validity_gated_oracle: bool = True,
    phase_weights: Optional[Tuple[float, float, float]] = None,
    oracle_scores_override: Optional[List[float]] = None,
    oracle_prediction_fn: Optional[Callable[[str], Any]] = None,
    w_alert: float = 0.0,
) -> List[float]:
    if oracle_scores_override is not None and len(oracle_scores_override) == len(smiles_list):
        oracle_scalars = oracle_scores_override
    elif oracle_prediction_fn is not None:
        oracle_scalars = []
        for s in smiles_list:
            pred = oracle_prediction_fn(s)
            sc = _scalar_from_prediction(pred, phase_weights) - w_alert * _alert_penalty(pred)
            oracle_scalars.append(sc)
    else:
        oracle_scalars = [
            _oracle_scalar(oracle_score_fn, s, phase_weights) if oracle_score_fn else 0.0
            for s in smiles_list
        ]
    base = []
    for i, s in enumerate(smiles_list):
        validity = validity_reward(s)
        qed = qed_reward(s)
        o_raw = oracle_scalars[i] if i < len(oracle_scalars) else 0.0
        oracle = o_raw * validity if validity_gated_oracle else o_raw
        base.append(w_validity * validity + w_qed * qed + w_oracle * oracle)
    diversity = diversity_reward(smiles_list)
    return [b + (w_diversity * diversity) for b in base]


def compute_rewards(
    smiles_list: List[str],
    oracle_score_fn: Optional[Callable[[str], Union[float, Dict[str, float], None]]] = None,
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
    w_diversity: float = 0.1,
    validity_gated_oracle: bool = True,
    phase_weights: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, float]:
    validity = sum(validity_reward(s) for s in smiles_list) / max(len(smiles_list), 1)
    qed = sum(qed_reward(s) for s in smiles_list) / max(len(smiles_list), 1)
    oracle = 0.0
    if oracle_score_fn:
        if validity_gated_oracle:
            oracle = sum(
                _oracle_scalar(oracle_score_fn, s, phase_weights) * validity_reward(s)
                for s in smiles_list
            ) / max(len(smiles_list), 1)
        else:
            oracle = sum(
                _oracle_scalar(oracle_score_fn, s, phase_weights) for s in smiles_list
            ) / max(len(smiles_list), 1)
    diversity = diversity_reward(smiles_list)
    total = w_validity * validity + w_qed * qed + w_oracle * oracle + w_diversity * diversity
    return {
        "validity": validity,
        "qed": qed,
        "oracle": oracle,
        "diversity": diversity,
        "total": total,
    }
