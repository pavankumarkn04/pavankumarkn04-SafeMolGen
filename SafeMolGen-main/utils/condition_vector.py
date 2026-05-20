"""Condition vector construction for generator conditioning and steering."""

from typing import Dict, Optional

import torch

from models.generator.transformer import COND_DIM


def get_target_condition_for_rl(device: str = "cpu") -> torch.Tensor:
    return get_target_condition(device=device, phase=0.5)


def get_target_condition(
    device: str = "cpu",
    phase: float = 0.5,
    phase1: Optional[float] = None,
    phase2: Optional[float] = None,
    phase3: Optional[float] = None,
) -> torch.Tensor:
    p1 = phase1 if phase1 is not None else phase
    p2 = phase2 if phase2 is not None else phase
    p3 = phase3 if phase3 is not None else phase
    n_admet = COND_DIM - 3
    vals = [0.5] * n_admet if n_admet > 0 else []
    while len(vals) < n_admet:
        vals.append(0.0)
    vals.extend([p1, p2, p3])
    return torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)


def build_condition_vector(
    admet: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    device: str = "cpu",
) -> torch.Tensor:
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    vals.extend([phase1, phase2, phase3])
    return torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)


def build_condition_vector_toward_target(
    admet: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    target_success: float = 0.5,
    blend: float = 0.92,
    device: str = "cpu",
    phase_boost: float = 0.0,
) -> torch.Tensor:
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    target_phase = min(0.75, max(0.55, target_success * 1.2) + phase_boost)
    p1 = phase1 + blend * max(0.0, target_phase - phase1)
    p2 = phase2 + blend * max(0.0, target_phase - phase2)
    p3 = phase3 + blend * max(0.0, target_phase - phase3)
    vals.extend([p1, p2, p3])
    return torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)


def build_condition_vector_toward_target_phase_aware(
    admet: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    target_success: float = 0.5,
    blend: float = 0.92,
    worst_phase_blend: float = 0.98,
    device: str = "cpu",
    phase_boost: float = 0.0,
) -> torch.Tensor:
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    target_phase = min(0.75, max(0.55, target_success * 1.2) + phase_boost)
    phases = [(phase1, 0), (phase2, 1), (phase3, 2)]
    worst_idx = min(phases, key=lambda x: x[0])[1]
    p1 = phase1 + (worst_phase_blend if worst_idx == 0 else blend) * max(0.0, target_phase - phase1)
    p2 = phase2 + (worst_phase_blend if worst_idx == 1 else blend) * max(0.0, target_phase - phase2)
    p3 = phase3 + (worst_phase_blend if worst_idx == 2 else blend) * max(0.0, target_phase - phase3)
    vals.extend([p1, p2, p3])
    return torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)


def build_condition_vector_toward_target_admet(
    admet: Dict[str, float],
    target_profile: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    target_success: float = 0.5,
    blend: float = 0.92,
    worst_phase_blend: float = 0.98,
    device: str = "cpu",
    phase_boost: float = 0.0,
) -> torch.Tensor:
    keys = sorted(admet.keys())[: COND_DIM - 3]
    admet_target = dict(admet)
    if target_profile.get("herg_max") is not None:
        admet_target["herg"] = float(target_profile["herg_max"])
    if target_profile.get("ames_max") is not None:
        admet_target["ames"] = float(target_profile["ames_max"])
    if target_profile.get("dili_max") is not None:
        admet_target["dili"] = float(target_profile["dili_max"])
    if target_profile.get("bioavailability_ma_min") is not None:
        admet_target["bioavailability_ma"] = float(target_profile["bioavailability_ma_min"])

    vals = [float(admet_target.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)

    target_phase = min(0.75, max(0.55, target_success * 1.2) + phase_boost)
    phases = [(phase1, 0), (phase2, 1), (phase3, 2)]
    worst_idx = min(phases, key=lambda x: x[0])[1]
    p1 = phase1 + (worst_phase_blend if worst_idx == 0 else blend) * max(0.0, target_phase - phase1)
    p2 = phase2 + (worst_phase_blend if worst_idx == 1 else blend) * max(0.0, target_phase - phase2)
    p3 = phase3 + (worst_phase_blend if worst_idx == 2 else blend) * max(0.0, target_phase - phase3)
    vals.extend([p1, p2, p3])
    return torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
