"""Integrated SafeMolGen + DrugOracle design loop (aligned with SafeMolGen-DrugOracle main)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import random

import torch
from rdkit import Chem

from models.generator.safemolgen import SafeMolGen
from models.oracle.drug_oracle import DrugOracle, OraclePrediction
from models.oracle.structural_alerts import STRUCTURAL_ALERTS_DB
from utils.chemistry import (
    calculate_properties,
    generate_mutations,
    tanimoto_similarity,
    validate_smiles,
)
from utils.condition_vector import (
    build_condition_vector_toward_target,
    build_condition_vector_toward_target_admet,
    build_condition_vector_toward_target_phase_aware,
)


@dataclass
class IterationResult:
    iteration: int
    smiles: str
    prediction: OraclePrediction
    improvements: List[str]
    passed_safety: bool
    used_oracle_feedback: bool

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "smiles": self.smiles,
            "phase1_prob": self.prediction.phase1_prob,
            "phase2_prob": self.prediction.phase2_prob,
            "phase3_prob": self.prediction.phase3_prob,
            "overall_prob": self.prediction.overall_prob,
            "improvements": self.improvements,
            "structural_alerts": self.prediction.structural_alerts,
            "recommendations": self.prediction.recommendations,
            "passed_safety": self.passed_safety,
            "used_oracle_feedback": self.used_oracle_feedback,
        }


@dataclass
class DesignResult:
    final_smiles: str
    final_prediction: OraclePrediction
    iteration_history: List[IterationResult]
    target_achieved: bool
    total_iterations: int

    def to_dict(self) -> Dict:
        return {
            "final_smiles": self.final_smiles,
            "final_phase1": self.final_prediction.phase1_prob,
            "final_phase2": self.final_prediction.phase2_prob,
            "final_phase3": self.final_prediction.phase3_prob,
            "final_overall": self.final_prediction.overall_prob,
            "target_achieved": self.target_achieved,
            "total_iterations": self.total_iterations,
            "history": [i.to_dict() for i in self.iteration_history],
            "recommendations": self.final_prediction.recommendations or [],
        }


def _passed_safety(
    prediction: OraclePrediction,
    safety_threshold: float,
    require_no_structural_alerts: bool,
) -> bool:
    if prediction.overall_prob < safety_threshold:
        return False
    if require_no_structural_alerts and prediction.structural_alerts:
        return False
    return True


def _encode_oracle_feedback(
    prediction: OraclePrediction,
    target_success: float = 0.5,
    device: str = "cpu",
    phase_boost: float = 0.0,
    use_phase_aware_steering: bool = True,
) -> Dict:
    target_profile: Dict[str, float] = {}
    admet = prediction.admet_predictions or {}
    if admet.get("herg", 0) > 0.5:
        target_profile["herg_max"] = 0.5
    if admet.get("ames", 0) > 0.5:
        target_profile["ames_max"] = 0.5
    if admet.get("dili", 0) > 0.5:
        target_profile["dili_max"] = 0.5
    if admet.get("bioavailability_ma", 1) < 0.5:
        target_profile["bioavailability_ma_min"] = 0.5
    avoid_smarts: List[str] = []
    for alert_name in prediction.structural_alerts or []:
        for alert in STRUCTURAL_ALERTS_DB.values():
            if alert.name == alert_name:
                avoid_smarts.append(alert.smarts)
                break
    if target_profile:
        condition_vector = build_condition_vector_toward_target_admet(
            admet,
            target_profile,
            prediction.phase1_prob,
            prediction.phase2_prob,
            prediction.phase3_prob,
            target_success=target_success,
            device=device,
            phase_boost=phase_boost,
        )
    elif use_phase_aware_steering:
        condition_vector = build_condition_vector_toward_target_phase_aware(
            admet,
            prediction.phase1_prob,
            prediction.phase2_prob,
            prediction.phase3_prob,
            target_success=target_success,
            device=device,
            phase_boost=phase_boost,
        )
    else:
        condition_vector = build_condition_vector_toward_target(
            admet,
            prediction.phase1_prob,
            prediction.phase2_prob,
            prediction.phase3_prob,
            target_success=target_success,
            device=device,
            phase_boost=phase_boost,
        )
    return {
        "target_profile": target_profile,
        "avoid_smarts": avoid_smarts,
        "condition_vector": condition_vector,
    }


def _filter_by_avoid(smiles_list: List[str], avoid_smarts: List[str]) -> List[str]:
    if not avoid_smarts:
        return smiles_list
    patterns = []
    for sma in avoid_smarts:
        p = Chem.MolFromSmarts(sma)
        if p is not None:
            patterns.append(p)
    if not patterns:
        return smiles_list
    out: List[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if any(mol.HasSubstructMatch(pat) for pat in patterns):
            continue
        out.append(smi)
    return out


def _filter_by_seed_scaffold(smiles_list: List[str], seed_smiles: Optional[str]) -> List[str]:
    if not seed_smiles or not validate_smiles(seed_smiles):
        return smiles_list
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        return smiles_list
    out: List[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.HasSubstructMatch(seed_mol):
            out.append(smi)
    return out if out else smiles_list


def _score_for_oracle_target(pred: OraclePrediction, target_profile: Dict) -> float:
    score = 0.0
    admet = pred.admet_predictions or {}
    if "herg_max" in target_profile:
        if admet.get("herg", 0) <= target_profile["herg_max"]:
            score += 0.1
    if "ames_max" in target_profile:
        if admet.get("ames", 0) <= target_profile["ames_max"]:
            score += 0.1
    if "dili_max" in target_profile:
        if admet.get("dili", 0) <= target_profile["dili_max"]:
            score += 0.1
    if "bioavailability_ma_min" in target_profile:
        if admet.get("bioavailability_ma", 0) >= target_profile["bioavailability_ma_min"]:
            score += 0.1
    return score


def _pareto_front(evaluated: List[Tuple[str, OraclePrediction]]) -> List[Tuple[str, OraclePrediction]]:
    if not evaluated:
        return []
    dominated = [False] * len(evaluated)
    for i, (_, pi) in enumerate(evaluated):
        for j, (_, pj) in enumerate(evaluated):
            if i == j:
                continue
            if (
                pj.phase1_prob >= pi.phase1_prob
                and pj.phase2_prob >= pi.phase2_prob
                and pj.phase3_prob >= pi.phase3_prob
                and (
                    pj.phase1_prob > pi.phase1_prob
                    or pj.phase2_prob > pi.phase2_prob
                    or pj.phase3_prob > pi.phase3_prob
                )
            ):
                dominated[i] = True
                break
    return [evaluated[i] for i in range(len(evaluated)) if not dominated[i]]


def _select_diverse(
    evaluated: List[Tuple[str, OraclePrediction]],
    ref_smiles: Optional[str],
    tanimoto_max: float = 0.7,
) -> Tuple[Optional[str], Optional[OraclePrediction]]:
    if not evaluated:
        return (None, None)
    sorted_eval = sorted(evaluated, key=lambda x: x[1].overall_prob, reverse=True)
    if not ref_smiles:
        return sorted_eval[0]
    for smi, pred in sorted_eval:
        if tanimoto_similarity(smi, ref_smiles) <= tanimoto_max:
            return (smi, pred)
    return sorted_eval[0]


def _score_phase_weighted(pred: OraclePrediction, w1: float = 0.2, w2: float = 0.5, w3: float = 0.3) -> float:
    return w1 * pred.phase1_prob + w2 * pred.phase2_prob + w3 * pred.phase3_prob


def _select_phase_weighted(
    evaluated: List[Tuple[str, OraclePrediction]],
    w1: float = 0.2,
    w2: float = 0.5,
    w3: float = 0.3,
    ref_smiles: Optional[str] = None,
    diversity_tanimoto_max: float = 0.7,
) -> Tuple[Optional[str], Optional[OraclePrediction]]:
    if not evaluated:
        return (None, None)
    ranked = sorted(evaluated, key=lambda x: _score_phase_weighted(x[1], w1, w2, w3), reverse=True)
    if ref_smiles:
        top_k = max(5, len(ranked) // 3)
        for smi, pred in ranked[:top_k]:
            if tanimoto_similarity(smi, ref_smiles) <= diversity_tanimoto_max:
                return (smi, pred)
    return ranked[0]


def _select_bottleneck(evaluated: List[Tuple[str, OraclePrediction]]) -> Tuple[Optional[str], Optional[OraclePrediction]]:
    if not evaluated:
        return (None, None)
    return max(
        evaluated,
        key=lambda x: min(x[1].phase1_prob, x[1].phase2_prob, x[1].phase3_prob),
    )


def _apply_improvement_pacing(
    selection_pool: List[Tuple[str, OraclePrediction]],
    best_score: float,
    target_success: float,
    iteration: int,
    max_iterations: int,
    max_step_per_iteration: float = 0.05,
) -> Optional[Tuple[str, OraclePrediction]]:
    if not selection_pool:
        return None
    remaining = max(1, max_iterations - iteration - 1)
    if best_score <= 0:
        desired_next = min(max_step_per_iteration, target_success / max(1, max_iterations))
        cap = min(1.0, desired_next * 1.5)
        improving = [(s, p) for s, p in selection_pool if 0 < p.overall_prob <= cap]
        if not improving:
            improving = [(s, p) for s, p in selection_pool if p.overall_prob > 0]
        if not improving:
            return None
        return min(improving, key=lambda x: abs(x[1].overall_prob - desired_next))
    step = min(max_step_per_iteration, (target_success - best_score) / remaining)
    cap = best_score + step
    improving_bounded = [(s, p) for s, p in selection_pool if best_score < p.overall_prob <= cap]
    if improving_bounded:
        desired_next = best_score + step
        return min(improving_bounded, key=lambda x: abs(x[1].overall_prob - desired_next))
    improving = [(s, p) for s, p in selection_pool if p.overall_prob > best_score]
    if not improving:
        return None
    return min(improving, key=lambda x: x[1].overall_prob)


def _select_with_target_and_diversity(
    evaluated: List[Tuple[str, OraclePrediction]],
    target_profile: Dict,
    ref_smiles: Optional[str],
    diversity_tanimoto_max: float = 0.7,
    target_bonus_weight: float = 0.15,
) -> Tuple[Optional[str], Optional[OraclePrediction]]:
    if not evaluated:
        return (None, None)
    if not target_profile:
        return max(evaluated, key=lambda x: x[1].overall_prob)

    def meets_any_target(item: Tuple[str, OraclePrediction]) -> int:
        _, pred = item
        return 1 if _score_for_oracle_target(pred, target_profile) > 0 else 0

    def sort_key(item: Tuple[str, OraclePrediction]):
        smi, pred = item
        target_bonus = _score_for_oracle_target(pred, target_profile)
        return (meets_any_target(item), pred.overall_prob + target_bonus_weight * target_bonus)

    sorted_pool = sorted(evaluated, key=sort_key, reverse=True)
    if not ref_smiles:
        return sorted_pool[0]
    top_k = max(5, len(sorted_pool) // 3)
    top_candidates = sorted_pool[:top_k]
    for smi, pred in top_candidates:
        if tanimoto_similarity(smi, ref_smiles) <= diversity_tanimoto_max:
            return (smi, pred)
    return sorted_pool[0]


def _relaxed_property_targets(property_targets: Dict, relax: bool = True) -> Dict:
    if not property_targets or not relax:
        return dict(property_targets) if property_targets else {}
    out = dict(property_targets)
    logp = out.get("logp")
    if logp is not None:
        lo, hi = logp if isinstance(logp, (list, tuple)) else (logp, logp)
        out["logp"] = [max(-2, float(lo) - 0.5), min(10, float(hi) + 0.5)]
    if out.get("mw_min") is not None:
        out["mw_min"] = max(0, float(out["mw_min"]) - 75)
    if out.get("mw") is not None:
        out["mw"] = min(1000, float(out["mw"]) + 75)
    if out.get("qed") is not None:
        out["qed"] = max(0.0, float(out["qed"]) - 0.08)
    if out.get("hbd") is not None:
        out["hbd"] = min(15, float(out["hbd"]) + 2)
    if out.get("hba") is not None:
        out["hba"] = min(20, float(out["hba"]) + 2)
    if out.get("tpsa") is not None:
        out["tpsa"] = min(200, float(out["tpsa"]) + 15)
    return out


def _filter_by_property_targets(smiles_list: List[str], targets: Optional[Dict]) -> List[str]:
    if not targets:
        return smiles_list
    logp_range = targets.get("logp")
    mw_max = targets.get("mw")
    mw_min = targets.get("mw_min")
    qed_min = targets.get("qed")
    hbd_max = targets.get("hbd")
    hba_max = targets.get("hba")
    tpsa_max = targets.get("tpsa")
    out: List[str] = []
    for smi in smiles_list:
        props = calculate_properties(smi)
        if props is None:
            continue
        if logp_range is not None:
            lo, hi = logp_range if isinstance(logp_range, (list, tuple)) else (logp_range, logp_range)
            if not (float(lo) <= props.get("logp", 0) <= float(hi)):
                continue
        if mw_min is not None and props.get("mw", 0) < float(mw_min):
            continue
        if mw_max is not None and props.get("mw", 0) > float(mw_max):
            continue
        if qed_min is not None and props.get("qed", 0) < float(qed_min):
            continue
        if hbd_max is not None and props.get("hbd", 0) > float(hbd_max):
            continue
        if hba_max is not None and props.get("hba", 0) > float(hba_max):
            continue
        if tpsa_max is not None and props.get("tpsa", 0) > float(tpsa_max):
            continue
        out.append(smi)
    return out


MIN_HEAVY_ATOMS_DRUGLIKE = 10
# Anti-grease: cap flexible chains (beyond main repo; prj_demo policy).
DEFAULT_MAX_ROTATABLE_BONDS = 15


def _is_druglike_complexity(smiles: str, min_heavy_atoms: int = MIN_HEAVY_ATOMS_DRUGLIKE) -> bool:
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() >= min_heavy_atoms


def _has_polar_or_aromatic_scaffold(smiles: str) -> bool:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return False
    if any(a.GetIsAromatic() for a in mol.GetAtoms()):
        return True
    return any(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())


def _passes_rotatable_cap(smiles: str, max_rotatable: Optional[int]) -> bool:
    if max_rotatable is None:
        return True
    props = calculate_properties(smiles)
    if props is None:
        return False
    return int(props.get("rotatable_bonds", 0)) <= int(max_rotatable)


def _passes_prj_druglike_filters(
    smiles: str,
    max_rotatable_bonds: Optional[int],
) -> bool:
    return (
        _is_druglike_complexity(smiles)
        and _has_polar_or_aromatic_scaffold(smiles)
        and _passes_rotatable_cap(smiles, max_rotatable_bonds)
    )


def _filter_evaluated_by_admet_targets(
    evaluated: List[Tuple[str, OraclePrediction]], property_targets: Optional[Dict]
) -> List[Tuple[str, OraclePrediction]]:
    if not property_targets or not evaluated:
        return evaluated
    solubility_range = property_targets.get("solubility")
    ppbr_range = property_targets.get("ppbr")
    clearance_max = property_targets.get("clearance_hepatocyte_max")
    if solubility_range is None and ppbr_range is None and clearance_max is None:
        return evaluated
    out: List[Tuple[str, OraclePrediction]] = []
    for smi, pred in evaluated:
        admet = pred.admet_predictions or {}
        if solubility_range is not None:
            val = admet.get("solubility_aqsoldb")
            if val is None:
                continue
            lo, hi = (
                solubility_range
                if isinstance(solubility_range, (list, tuple))
                else (solubility_range, solubility_range)
            )
            if not (float(lo) <= val <= float(hi)):
                continue
        if ppbr_range is not None:
            val = admet.get("ppbr_az")
            if val is None:
                continue
            lo, hi = ppbr_range if isinstance(ppbr_range, (list, tuple)) else (ppbr_range, ppbr_range)
            if not (float(lo) <= val <= float(hi)):
                continue
        if clearance_max is not None:
            val = admet.get("clearance_hepatocyte_az")
            if val is not None and val > float(clearance_max):
                continue
        out.append((smi, pred))
    return out if out else evaluated


class SafeMolGenDrugOracle:
    def __init__(
        self,
        generator: SafeMolGen,
        oracle: DrugOracle,
        device: str = "cpu",
        reranker=None,
        generator_early: Optional[SafeMolGen] = None,
    ):
        self.generator = generator
        self.oracle = oracle
        self.device = device
        self.reranker = reranker
        self.generator_early = generator_early

    @classmethod
    def from_pretrained(
        cls,
        generator_path: str,
        oracle_path: str,
        admet_path: str,
        endpoint_names: List[str],
        endpoint_task_types: Dict[str, str],
        admet_input_dim: int,
        device: str = "cpu",
        reranker_path: Optional[str] = None,
        generator_early: Optional[SafeMolGen] = None,
        **_: Dict,
    ) -> "SafeMolGenDrugOracle":
        generator = SafeMolGen.from_pretrained(generator_path, device=device)
        oracle = DrugOracle.from_pretrained(
            oracle_path=oracle_path,
            admet_path=admet_path,
            endpoint_names=endpoint_names,
            endpoint_task_types=endpoint_task_types,
            input_dim=admet_input_dim,
            device=device,
        )
        reranker = None
        if reranker_path:
            from models.reranker.model import load_reranker
            reranker = load_reranker(reranker_path, generator.tokenizer, device=device)
        return cls(generator, oracle, device=device, reranker=reranker, generator_early=generator_early)

    def evaluate_molecule(self, smiles: str) -> Optional[OraclePrediction]:
        if not validate_smiles(smiles):
            return None
        return self.oracle.predict(smiles)

    def generate_candidates(
        self,
        n: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        condition: Optional[torch.Tensor] = None,
        generator_override: Optional[SafeMolGen] = None,
    ) -> List[str]:
        gen = generator_override or self.generator
        return gen.generate(n=n, temperature=temperature, top_k=top_k, device=self.device, condition=condition)

    def _rerank_candidates(
        self,
        condition: torch.Tensor,
        candidates: List[str],
        top_k: int,
    ) -> List[str]:
        """Score candidates with reranker and return top_k by predicted oracle score."""
        if not self.reranker or not candidates:
            return candidates
        tokenizer = self.generator.tokenizer
        cond_batch = condition.expand(len(candidates), -1)
        ids = [tokenizer.encode(s) for s in candidates]
        max_len = tokenizer.max_length
        pad_id = tokenizer.vocab.get(tokenizer.PAD_TOKEN, 0)
        padded = []
        for seq in ids:
            if len(seq) < max_len:
                seq = seq + [pad_id] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded.append(seq)
        ids_t = torch.tensor(padded, dtype=torch.long, device=self.device)
        cond_batch = cond_batch.to(self.device)
        with torch.no_grad():
            scores = self.reranker(cond_batch, ids_t)
        sorted_idx = scores.cpu().argsort(descending=True)
        return [candidates[i] for i in sorted_idx[:top_k]]

    def save_result(self, result: "DesignResult", path: str) -> None:
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(result.to_dict(), f, indent=2)

    def compare_molecules(self, smiles_list: List[str]) -> List[Dict]:
        rows = []
        for smi in smiles_list:
            pred = self.evaluate_molecule(smi)
            if pred is None:
                continue
            rows.append(
                {
                    "smiles": smi,
                    "prediction": pred.to_dict(),
                    "properties": calculate_properties(smi),
                }
            )
        rows.sort(key=lambda x: x["prediction"]["overall_prob"], reverse=True)
        return rows

    def design_molecule(
        self,
        target_success: float = 0.5,
        max_iterations: int = 10,
        candidates_per_iteration: int = 250,
        top_k: int = 40,
        safety_threshold: float = 0.2,
        require_no_structural_alerts: bool = False,
        use_oracle_feedback: bool = True,
        property_targets: Optional[Dict] = None,
        seed_smiles: Optional[str] = None,
        on_iteration_done: Optional[Callable[[DesignResult], None]] = None,
        use_phase_aware_steering: bool = True,
        use_improvement_pacing: bool = True,
        max_step_per_iteration: float = 0.08,
        first_iteration_temperature: Optional[float] = None,
        selection_mode: str = "phase_weighted",
        diversity_tanimoto_max: float = 0.7,
        exploration_fraction: float = 0.25,
        max_rotatable_bonds: Optional[int] = DEFAULT_MAX_ROTATABLE_BONDS,
        show_progress: bool = False,
        use_reranker: bool = False,
        reranker_top_k: Optional[int] = None,
        **_: Dict,
    ) -> DesignResult:
        iteration_history: List[IterationResult] = []
        best_smiles: Optional[str] = None
        best_pred: Optional[OraclePrediction] = None
        best_score = 0.0
        oracle_feedback_for_next: Optional[Dict] = None
        consecutive_empty_iters = 0
        consecutive_no_improvement = 0

        temp_schedule = [0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72]

        for iteration in range(max_iterations):
            if iteration == 0:
                temperature = (
                    first_iteration_temperature if first_iteration_temperature is not None else 1.4
                )
                generator_override = self.generator_early
            else:
                temperature = temp_schedule[min(iteration, len(temp_schedule) - 1)]
                generator_override = None

            condition: Optional[torch.Tensor] = None
            if oracle_feedback_for_next:
                condition = oracle_feedback_for_next.get("condition_vector")

            evaluated: List[Tuple[str, OraclePrediction]] = []
            cond_dim = getattr(self.generator.model, "cond_dim", 0)
            n_mutations = 100
            if best_smiles and iteration > 0:
                if cond_dim == 0 and oracle_feedback_for_next:
                    n_mutations = 200
                if consecutive_no_improvement >= 1:
                    n_mutations = max(n_mutations, 200)
                for smi in generate_mutations(best_smiles, n=n_mutations, random_seed=iteration + 42):
                    pred = self.evaluate_molecule(smi)
                    if pred is not None:
                        evaluated.append((smi, pred))

            base_mult = 1.5 if property_targets else 1.0
            for attempt in range(2):
                n_cand = int(candidates_per_iteration * base_mult * (2**attempt))
                temps = [temperature]
                if attempt == 0:
                    temps.append(min(0.95, temperature * 1.15))
                    temps.append(1.0)
                candidates: List[str] = []
                for t in temps:
                    if t == temperature:
                        n_here = n_cand
                    elif t < 1.0:
                        n_here = max(50, n_cand // 2)
                    else:
                        n_here = max(40, n_cand // 3)
                    candidates.extend(
                        self.generate_candidates(
                            n=n_here,
                            temperature=t,
                            top_k=top_k,
                            condition=condition,
                            generator_override=generator_override,
                        )
                    )
                eff_exploration = (
                    0.35 if (oracle_feedback_for_next and exploration_fraction > 0) else exploration_fraction
                )
                if consecutive_no_improvement >= 1:
                    eff_exploration = max(eff_exploration, 0.4)
                if attempt == 0 and eff_exploration > 0:
                    n_explore = max(40, int(eff_exploration * n_cand))
                    candidates.extend(
                        self.generate_candidates(
                            n=n_explore,
                            temperature=1.2,
                            top_k=min(top_k + 15, 80),
                            condition=None,
                        )
                    )
                candidates = list(dict.fromkeys(candidates))

                if use_reranker and self.reranker and condition is not None:
                    k = reranker_top_k if reranker_top_k is not None else 200
                    candidates = self._rerank_candidates(condition, candidates, min(k, len(candidates)))

                if property_targets:
                    candidates = _filter_by_property_targets(candidates, property_targets)
                if seed_smiles:
                    candidates = _filter_by_seed_scaffold(candidates, seed_smiles)
                if oracle_feedback_for_next:
                    avoid_smarts = oracle_feedback_for_next.get("avoid_smarts", [])
                    candidates_after_avoid = _filter_by_avoid(candidates, avoid_smarts)
                    min_after_avoid = max(20, int(0.2 * n_cand))
                    if len(candidates_after_avoid) >= min_after_avoid:
                        candidates = candidates_after_avoid

                for smi in candidates:
                    pred = self.evaluate_molecule(smi)
                    if pred is not None:
                        evaluated.append((smi, pred))
                if len(evaluated) >= 1:
                    break

            filters_relaxed_this_round = False
            used_soft_fallback = False
            if not evaluated and property_targets:
                for relax_level, targets in enumerate(
                    [
                        _relaxed_property_targets(property_targets, relax=True),
                        _relaxed_property_targets(
                            _relaxed_property_targets(property_targets, relax=True), relax=True
                        ),
                    ]
                ):
                    if evaluated:
                        break
                    n_relax = int(candidates_per_iteration * base_mult * 3)
                    relax_candidates: List[str] = []
                    for t in [temperature, min(0.95, temperature * 1.15), 1.0]:
                        relax_candidates.extend(
                            self.generate_candidates(
                                n=max(80, n_relax // 3),
                                temperature=t,
                                top_k=top_k,
                                condition=condition,
                                generator_override=generator_override,
                            )
                        )
                    relax_candidates = list(dict.fromkeys(relax_candidates))
                    relax_candidates = _filter_by_property_targets(relax_candidates, targets)
                    if seed_smiles:
                        relax_candidates = _filter_by_seed_scaffold(relax_candidates, seed_smiles)
                    if oracle_feedback_for_next:
                        avoid_smarts = oracle_feedback_for_next.get("avoid_smarts", [])
                        relax_candidates = _filter_by_avoid(relax_candidates, avoid_smarts)
                    for smi in relax_candidates:
                        pred = self.evaluate_molecule(smi)
                        if pred is not None:
                            evaluated.append((smi, pred))
                    if evaluated:
                        filters_relaxed_this_round = True
                        break
                if not evaluated:
                    no_filter_candidates: List[str] = []
                    for t in [temperature, 0.95, 1.0]:
                        no_filter_candidates.extend(
                            self.generate_candidates(
                                n=max(100, int(candidates_per_iteration * base_mult)),
                                temperature=t,
                                top_k=top_k,
                                condition=condition,
                                generator_override=generator_override,
                            )
                        )
                    no_filter_candidates = list(dict.fromkeys(no_filter_candidates))
                    if seed_smiles:
                        no_filter_candidates = _filter_by_seed_scaffold(no_filter_candidates, seed_smiles)
                    if oracle_feedback_for_next:
                        avoid_smarts = oracle_feedback_for_next.get("avoid_smarts", [])
                        no_filter_candidates = _filter_by_avoid(no_filter_candidates, avoid_smarts)
                    for smi in no_filter_candidates:
                        pred = self.evaluate_molecule(smi)
                        if pred is not None and _is_druglike_complexity(smi):
                            evaluated.append((smi, pred))
                    if evaluated:
                        used_soft_fallback = True

            if not evaluated:
                _no_pred = OraclePrediction(
                    phase1_prob=0.0,
                    phase2_prob=0.0,
                    phase3_prob=0.0,
                    overall_prob=0.0,
                    admet_predictions={},
                    risk_factors=[],
                    structural_alerts=[],
                    recommendations=[
                        {
                            "type": "info",
                            "suggestion": "No candidates passed property filters this round. Try relaxing logP, MW, or QED.",
                        },
                    ],
                )
                iteration_history.append(
                    IterationResult(
                        iteration=iteration + 1,
                        smiles="",
                        prediction=_no_pred,
                        improvements=[],
                        passed_safety=False,
                        used_oracle_feedback=oracle_feedback_for_next is not None,
                    )
                )
                if on_iteration_done:
                    on_iteration_done(
                        DesignResult(
                            final_smiles=best_smiles or "",
                            final_prediction=best_pred or _no_pred,
                            iteration_history=iteration_history.copy(),
                            target_achieved=False,
                            total_iterations=iteration + 1,
                        )
                    )
                consecutive_empty_iters += 1
                if consecutive_empty_iters >= 2:
                    break
                continue

            consecutive_empty_iters = 0
            evaluated = list({k[0]: k for k in evaluated}.values())
            evaluated.sort(key=lambda x: x[1].overall_prob, reverse=True)
            if property_targets:
                evaluated = _filter_evaluated_by_admet_targets(evaluated, property_targets)
            if not evaluated:
                continue

            evaluated_druglike = [
                (s, p)
                for s, p in evaluated
                if _passes_prj_druglike_filters(s, max_rotatable_bonds)
            ]
            if not evaluated_druglike:
                if best_smiles is not None and best_pred is not None:
                    iter_best_smi, iter_best_pred = best_smiles, best_pred
                else:
                    continue
            else:
                selection_pool = evaluated_druglike
                target_profile = (
                    oracle_feedback_for_next.get("target_profile", {}) if oracle_feedback_for_next else {}
                )
                if target_profile:
                    iter_best_smi, iter_best_pred = _select_with_target_and_diversity(
                        selection_pool,
                        target_profile,
                        best_smiles,
                        diversity_tanimoto_max,
                        target_bonus_weight=0.2,
                    )
                elif selection_mode == "pareto":
                    front = _pareto_front(selection_pool)
                    iter_best_smi, iter_best_pred = (
                        max(front, key=lambda x: x[1].overall_prob) if front else selection_pool[0]
                    )
                elif selection_mode == "diversity":
                    iter_best_smi, iter_best_pred = _select_diverse(
                        selection_pool, best_smiles, diversity_tanimoto_max
                    )
                elif selection_mode == "phase_weighted":
                    iter_best_smi, iter_best_pred = _select_phase_weighted(
                        selection_pool,
                        ref_smiles=best_smiles,
                        diversity_tanimoto_max=diversity_tanimoto_max,
                    )
                elif selection_mode == "bottleneck":
                    iter_best_smi, iter_best_pred = _select_bottleneck(selection_pool)
                else:
                    iter_best_smi, iter_best_pred = max(
                        selection_pool, key=lambda x: x[1].overall_prob
                    )

                if use_improvement_pacing and selection_pool and (best_pred is not None or iteration == 0):
                    paced = _apply_improvement_pacing(
                        selection_pool,
                        best_score,
                        target_success,
                        iteration,
                        max_iterations,
                        max_step_per_iteration,
                    )
                    if paced is not None:
                        iter_best_smi, iter_best_pred = paced
                # Allow bigger jumps when the model finds genuinely better candidates
                # rather than hard-clamping which kills creative exploration

            if iter_best_smi is None or iter_best_pred is None:
                continue

            passed_safety = _passed_safety(
                iter_best_pred, safety_threshold, require_no_structural_alerts
            )
            used_oracle_feedback = oracle_feedback_for_next is not None

            improvements: List[str] = []
            if used_soft_fallback:
                improvements.append(
                    "No molecule met your property filters; showing best available. Consider relaxing logP, MW, or QED."
                )
            elif filters_relaxed_this_round:
                improvements.append("Filters slightly relaxed this round to get candidates.")
            if best_pred is not None:
                if iter_best_pred.phase1_prob > best_pred.phase1_prob:
                    improvements.append(
                        f"Phase I: +{iter_best_pred.phase1_prob - best_pred.phase1_prob:.1%}"
                    )
                if iter_best_pred.phase2_prob > best_pred.phase2_prob:
                    improvements.append(
                        f"Phase II: +{iter_best_pred.phase2_prob - best_pred.phase2_prob:.1%}"
                    )
                if iter_best_pred.overall_prob > best_pred.overall_prob:
                    improvements.append(
                        f"Overall: +{iter_best_pred.overall_prob - best_pred.overall_prob:.1%}"
                    )

            iteration_history.append(
                IterationResult(
                    iteration=iteration + 1,
                    smiles=iter_best_smi,
                    prediction=iter_best_pred,
                    improvements=improvements,
                    passed_safety=passed_safety,
                    used_oracle_feedback=used_oracle_feedback,
                )
            )

            if (
                _passes_prj_druglike_filters(iter_best_smi, max_rotatable_bonds)
                and iter_best_pred.overall_prob > best_score
            ):
                best_smiles = iter_best_smi
                best_pred = iter_best_pred
                best_score = iter_best_pred.overall_prob
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            source_for_feedback = best_pred if best_pred is not None else iter_best_pred
            has_recommendations = bool(getattr(source_for_feedback, "recommendations", None))
            below_target = source_for_feedback.overall_prob < target_success
            skip_feedback_this_round = consecutive_no_improvement >= 2 and below_target
            if skip_feedback_this_round:
                consecutive_no_improvement = 0
            if (
                use_oracle_feedback
                and not skip_feedback_this_round
                and (below_target or has_recommendations)
            ):
                if target_success >= 0.6:
                    phase_boost = min(0.22, (iteration + 1) * 0.028)
                else:
                    phase_boost = min(0.18, (iteration + 1) * 0.024)
                oracle_feedback_for_next = _encode_oracle_feedback(
                    source_for_feedback,
                    target_success,
                    device=self.device,
                    phase_boost=phase_boost,
                    use_phase_aware_steering=use_phase_aware_steering,
                )
            else:
                oracle_feedback_for_next = None

            if on_iteration_done:
                _cur_smi = best_smiles if best_smiles is not None else iter_best_smi
                _cur_pred = best_pred if best_pred is not None else iter_best_pred
                on_iteration_done(
                    DesignResult(
                        final_smiles=_cur_smi or "",
                        final_prediction=_cur_pred,
                        iteration_history=iteration_history.copy(),
                        target_achieved=best_score >= target_success,
                        total_iterations=iteration + 1,
                    )
                )

        if best_smiles is None and iteration_history:
            druglike_from_history = [
                (h.smiles, h.prediction)
                for h in iteration_history
                if h.smiles and _is_druglike_complexity(h.smiles)
            ]
            if druglike_from_history:
                best_smiles, best_pred = max(
                    druglike_from_history, key=lambda x: x[1].overall_prob
                )
                best_score = best_pred.overall_prob
            else:
                best_smiles = ""
                best_pred = iteration_history[-1].prediction

        if best_pred is None:
            fallback = random.choice(["CCO", "CCN", "c1ccccc1"])
            best_smiles = best_smiles or fallback
            best_pred = self.evaluate_molecule(best_smiles)
            if best_pred is None:
                raise RuntimeError("No valid candidate produced by generator/evaluator loop.")

        return DesignResult(
            final_smiles=best_smiles or "",
            final_prediction=best_pred,
            iteration_history=iteration_history,
            target_achieved=best_pred.overall_prob >= target_success,
            total_iterations=max_iterations,
        )

    def design_molecule_with_restarts(self, n_restarts: int = 5, **kwargs) -> DesignResult:
        best: Optional[DesignResult] = None
        for _ in range(max(1, n_restarts)):
            result = self.design_molecule(**kwargs)
            if best is None or result.final_prediction.overall_prob > best.final_prediction.overall_prob:
                best = result
            if best.target_achieved:
                break
        return best

    def design_molecule_evolutionary(
        self,
        population_size: int = 20,
        generations: int = 8,
        target_success: float = 0.5,
        property_targets: Optional[Dict] = None,
        max_rotatable_bonds: Optional[int] = DEFAULT_MAX_ROTATABLE_BONDS,
        **extra: Any,
    ) -> DesignResult:
        population = self.generate_candidates(n=population_size, temperature=0.95, top_k=40, condition=None)
        population = _filter_by_property_targets(population, property_targets)
        evaluated: List[Tuple[str, OraclePrediction]] = []
        for smi in population:
            pred = self.evaluate_molecule(smi)
            if pred is not None:
                evaluated.append((smi, pred))
        if not evaluated:
            return self.design_molecule(
                target_success=target_success,
                max_iterations=3,
                candidates_per_iteration=120,
                property_targets=property_targets,
                max_rotatable_bonds=max_rotatable_bonds,
                **extra,
            )

        evaluated.sort(key=lambda x: x[1].overall_prob, reverse=True)
        evaluated = [
            (s, p)
            for s, p in evaluated
            if _passes_prj_druglike_filters(s, max_rotatable_bonds)
        ]
        if not evaluated:
            return self.design_molecule(
                target_success=target_success,
                max_iterations=3,
                candidates_per_iteration=120,
                property_targets=property_targets,
                max_rotatable_bonds=max_rotatable_bonds,
                **extra,
            )
        best_smi, best_pred = evaluated[0]
        history = [
            IterationResult(
                iteration=1,
                smiles=best_smi,
                prediction=best_pred,
                improvements=[],
                passed_safety=_passed_safety(best_pred, 0.2, False),
                used_oracle_feedback=False,
            )
        ]
        for gen in range(max(1, generations - 1)):
            candidates: List[str] = []
            for smi, _ in evaluated[:population_size]:
                candidates.extend(generate_mutations(smi, n=15, random_seed=gen + 11))
            candidates.extend(self.generate_candidates(n=80, temperature=0.9, top_k=50))
            candidates = list(dict.fromkeys(candidates))
            candidates = _filter_by_property_targets(candidates, property_targets)
            new_eval: List[Tuple[str, OraclePrediction]] = []
            for smi in candidates:
                pred = self.evaluate_molecule(smi)
                if pred is not None:
                    new_eval.append((smi, pred))
            if not new_eval:
                continue
            new_eval = [
                (s, p)
                for s, p in new_eval
                if _passes_prj_druglike_filters(s, max_rotatable_bonds)
            ]
            if not new_eval:
                continue
            new_eval.sort(key=lambda x: x[1].overall_prob, reverse=True)
            evaluated = new_eval[:population_size]
            cur_smi, cur_pred = evaluated[0]
            if cur_pred.overall_prob > best_pred.overall_prob:
                best_smi, best_pred = cur_smi, cur_pred
            history.append(
                IterationResult(
                    iteration=gen + 2,
                    smiles=cur_smi,
                    prediction=cur_pred,
                    improvements=[],
                    passed_safety=_passed_safety(cur_pred, 0.2, False),
                    used_oracle_feedback=False,
                )
            )
            if best_pred.overall_prob >= target_success:
                break
        return DesignResult(
            final_smiles=best_smi,
            final_prediction=best_pred,
            iteration_history=history,
            target_achieved=best_pred.overall_prob >= target_success,
            total_iterations=len(history),
        )
