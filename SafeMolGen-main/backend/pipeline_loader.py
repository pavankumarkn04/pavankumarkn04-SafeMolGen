"""Load generator-evaluator integrated pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from models.generator.safemolgen import SafeMolGen
from models.integrated.pipeline import SafeMolGenDrugOracle
from utils.checkpoint_utils import get_admet_node_feature_dim

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model_pt_path(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "model.pt").exists():
        return path
    if (path / "best" / "model.pt").exists():
        return path / "best"
    return None


def _resolve_early_generator() -> Optional[Path]:
    env = os.environ.get("GENERATOR_EARLY_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return _model_pt_path(p)
    config_file = PROJECT_ROOT / "config" / "pipeline.yaml"
    if config_file.exists():
        try:
            cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            gp = cfg.get("generator_early_path")
            if gp:
                p = Path(gp)
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                resolved = _model_pt_path(p)
                if resolved is not None:
                    return resolved
        except Exception:
            pass
    return _model_pt_path(PROJECT_ROOT / "checkpoints" / "generator_early")


def _read_endpoints(path: Path) -> Tuple[List[str], Dict[str, str]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names: List[str] = []
    task_types: Dict[str, str] = {}
    for item in cfg.get("endpoints", []):
        if not item.get("enabled", True):
            continue
        name = item["name"]
        names.append(name)
        task_types[name] = item["task_type"]
    return names, task_types


def load_pipeline(use_rl_model: bool = False) -> Optional[SafeMolGenDrugOracle]:
    generator_path = PROJECT_ROOT / "checkpoints" / ("generator_rl" if use_rl_model else "generator")
    if not generator_path.exists():
        return None
    oracle_path = PROJECT_ROOT / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = PROJECT_ROOT / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = PROJECT_ROOT / "config" / "endpoints.yaml"
    if not (oracle_path.exists() and admet_path.exists() and endpoints_path.exists()):
        return None

    endpoint_names, endpoint_task_types = _read_endpoints(endpoints_path)
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    generator_early = None
    early_path = _resolve_early_generator()
    if early_path is not None:
        try:
            generator_early = SafeMolGen.from_pretrained(str(early_path), device="cpu")
        except Exception:
            generator_early = None

    reranker_path: Optional[str] = None
    reranker_ckpt = PROJECT_ROOT / "checkpoints" / "reranker" / "reranker.pt"
    if reranker_ckpt.exists():
        reranker_path = str(reranker_ckpt)

    return SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(generator_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
        reranker_path=reranker_path,
        generator_early=generator_early,
    )

