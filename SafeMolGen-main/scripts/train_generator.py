"""Train SafeMolGen (Phase 3).

Pretrain trains a CONDITIONED generator (cond_dim=25) so the pipeline can steer
generation with Oracle feedback. Saves to checkpoints/generator by default.

Pretrain expects generator SMILES data at data/processed/generator/smiles.tsv.
If that path is missing, SMILES are aggregated from data/admet_group/*/train_val.csv.

RL stage requires a pretrained generator (--resume checkpoints/generator) and loads
the DrugOracle to use overall_prob as reward.
"""

from pathlib import Path
from typing import Callable, Optional

import argparse
import torch
import yaml

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import CausalTransformerGenerator, COND_DIM
from models.generator.trainer import PretrainConfig, train_pretrain
from models.generator.rl_trainer import RLConfig, train_rl
from models.generator.safemolgen import SafeMolGen
from models.generator.cond_dataset import CondSMILESDataset, TargetCondSMILESDataset
from utils.data_utils import load_and_prepare_smiles, aggregate_admet_smiles, read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from utils.chemistry import validate_smiles
from utils.condition_vector import get_target_condition, get_target_condition_for_rl

_DEFAULT_PRETRAIN_EPOCHS = 30
_DEFAULT_RL_EPOCHS = 5
_DEFAULT_RL_BATCH = 8
_VALIDITY_SAMPLE_SIZE = 200
_PRODUCTION_PRETRAIN_EPOCHS = 30
_PRODUCTION_PRETRAIN_LIMIT = 200_000
_PRODUCTION_PRETRAIN_BATCH = 128
_PRODUCTION_RL_EPOCHS = 15
_PRODUCTION_RL_BATCH = 16
_PRETRAIN_CHECKPOINT_EVERY = 10
_PRETRAIN_CHECKPOINT_EVERY_PRODUCTION = 5


def _make_oracle_for_pretrain(project_root: Path, device: str = "cpu"):
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = project_root / "config" / "endpoints.yaml"
    if not oracle_path.exists() or not admet_path.exists() or not endpoints_path.exists():
        return None
    from models.oracle.drug_oracle import DrugOracle
    endpoints_cfg = yaml.safe_load(endpoints_path.read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    input_dim = get_admet_node_feature_dim(str(admet_path))
    return DrugOracle.from_pretrained(
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        input_dim=input_dim,
        device=device,
    )


def _make_oracle_score_fn(project_root: Path, device: str = "cpu") -> Optional[Callable[[str], float]]:
    oracle = _make_oracle_for_pretrain(project_root, device=device)
    if oracle is None:
        return None

    def score_fn(smiles: str) -> float:
        if not validate_smiles(smiles):
            return 0.0
        pred = oracle.predict(smiles)
        return pred.overall_prob if pred else 0.0

    return score_fn


def _make_oracle_prediction_fn(project_root: Path, device: str = "cpu"):
    oracle = _make_oracle_for_pretrain(project_root, device=device)
    if oracle is None:
        return None
    return oracle.predict


def _make_oracle_phase_fn(project_root: Path, device: str = "cpu") -> Optional[Callable[[str], dict]]:
    oracle = _make_oracle_for_pretrain(project_root, device=device)
    if oracle is None:
        return None

    def phase_fn(smiles: str) -> dict:
        if not validate_smiles(smiles):
            return {"phase1": 0.0, "phase2": 0.0, "phase3": 0.0}
        pred = oracle.predict(smiles)
        if pred is None:
            return {"phase1": 0.0, "phase2": 0.0, "phase3": 0.0}
        return {"phase1": pred.phase1_prob, "phase2": pred.phase2_prob, "phase3": pred.phase3_prob}

    return phase_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "rl"], default="pretrain")
    parser.add_argument("--data", type=str, default="data/processed/generator/smiles.tsv")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--no-canonicalize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-validity", type=float, default=None)
    parser.add_argument("--w-oracle", type=float, default=None)
    parser.add_argument("--w-diversity", type=float, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--phase-weights", type=str, default=None)
    parser.add_argument("--batch-normalize-oracle", action="store_true")
    parser.add_argument("--use-ppo", action="store_true")
    parser.add_argument("--ppo-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--use-value-baseline", action="store_true")
    parser.add_argument("--w-alert", type=float, default=0.0)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--use-target-condition", action="store_true")
    parser.add_argument("--target-phase", type=float, default=0.5)
    args = parser.parse_args()

    production = getattr(args, "production", False)
    model_config_for_save = None

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    if args.stage == "pretrain":
        limit = _PRODUCTION_PRETRAIN_LIMIT if production else args.limit
        if data_path.exists():
            smiles_list = load_and_prepare_smiles(
                data_path, limit=limit, canonicalize=not args.no_canonicalize,
            )
        else:
            admet_base = project_root / "data" / "admet_group"
            if not admet_base.exists():
                raise FileNotFoundError(
                    f"SMILES data not found at {data_path}. "
                    "Run scripts/download_chembl_smiles.py or ensure data/admet_group exists."
                )
            smiles_list = aggregate_admet_smiles(admet_base, limit=limit)
        if not smiles_list:
            raise ValueError("No valid SMILES to train on.")
        resume_pretrain = Path(args.resume) if args.resume else None
        if resume_pretrain is not None and not resume_pretrain.is_absolute():
            resume_pretrain = project_root / resume_pretrain
        if resume_pretrain is not None and (resume_pretrain / "model.pt").exists():
            gen = SafeMolGen.from_pretrained(str(resume_pretrain), device="cpu")
            tokenizer = gen.tokenizer
            model = gen.model
            tokenizer.fit(smiles_list)
            _ckpt = torch.load(resume_pretrain / "model.pt", map_location="cpu", weights_only=False)
            _cfg = _ckpt.get("config", {})
            d_model = _cfg.get("d_model", 256)
            num_layers = _cfg.get("num_layers", 6)
            dim_feedforward = _cfg.get("dim_feedforward", 512)
        else:
            tokenizer = SMILESTokenizer(max_length=128)
            tokenizer.fit(smiles_list)
            d_model = args.d_model if args.d_model is not None else 256
            num_layers = args.num_layers if args.num_layers is not None else 6
            dim_feedforward = max(512, 2 * d_model)
            model = CausalTransformerGenerator(
                vocab_size=tokenizer.vocab_size,
                cond_dim=COND_DIM,
                d_model=d_model,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
            )
        pretrain_epochs = args.epochs if args.epochs is not None else (_PRODUCTION_PRETRAIN_EPOCHS if production else _DEFAULT_PRETRAIN_EPOCHS)
        pretrain_batch = args.batch_size if args.batch_size is not None else (_PRODUCTION_PRETRAIN_BATCH if production else 64)
        config = PretrainConfig(
            epochs=pretrain_epochs, batch_size=pretrain_batch,
            grad_clip=1.0, use_cosine_lr=True, shuffle_seed=args.seed,
        )

        out_dir = Path(args.out) if args.out else (project_root / "checkpoints" / "generator")
        out_dir.mkdir(parents=True, exist_ok=True)

        best_validity = [0.0]
        checkpoint_every = _PRETRAIN_CHECKPOINT_EVERY_PRODUCTION if production else _PRETRAIN_CHECKPOINT_EVERY
        model_config = {"d_model": d_model, "nhead": 8, "num_layers": num_layers, "dim_feedforward": dim_feedforward, "dropout": 0.1, "cond_dim": getattr(model, "cond_dim", 0)}

        target_condition_tensor = None

        def on_epoch_end(epoch: int, m, tok):
            gen = SafeMolGen(tok, m)
            if target_condition_tensor is not None:
                cond = target_condition_tensor
            else:
                cond = torch.zeros(1, getattr(m, "cond_dim", 0), device=config.device, dtype=torch.float32) if getattr(m, "cond_dim", 0) > 0 else None
            samples = gen.generate(n=_VALIDITY_SAMPLE_SIZE, temperature=0.8, device=config.device, condition=cond)
            valid = sum(1 for s in samples if validate_smiles(s))
            pct_valid = 100.0 * valid / max(len(samples), 1)
            print(f"Epoch {epoch} validity: {valid}/{len(samples)} ({pct_valid:.1f}%)")
            if pct_valid > best_validity[0]:
                best_validity[0] = pct_valid
                (out_dir / "best").mkdir(parents=True, exist_ok=True)
                gen.save(str(out_dir / "best"), config=model_config)
            if epoch % checkpoint_every == 0:
                gen.save(str(out_dir), config=model_config)

        pretrain_dataset = None
        if getattr(args, "use_target_condition", False) and getattr(model, "cond_dim", 0) > 0:
            target_condition_tensor = get_target_condition(device=config.device, phase=args.target_phase)
            pretrain_dataset = TargetCondSMILESDataset(smiles_list, tokenizer, target_condition_tensor)
        else:
            oracle = _make_oracle_for_pretrain(project_root, device=config.device)
            if oracle is not None and getattr(model, "cond_dim", 0) > 0:
                pretrain_dataset = CondSMILESDataset(
                    smiles_list, tokenizer, oracle,
                    device=config.device, zero_condition_fraction=0.1,
                )
        train_pretrain(model, tokenizer, smiles_list, config, on_epoch_end=on_epoch_end, dataset=pretrain_dataset)
        if (out_dir / "best" / "model.pt").exists():
            gen = SafeMolGen.from_pretrained(str(out_dir / "best"), device=config.device)
            model, tokenizer = gen.model, gen.tokenizer
        else:
            gen = SafeMolGen(tokenizer, model)
        model_config_for_save = model_config
    else:
        rl_kw = {}
        if args.w_validity is not None:
            rl_kw["w_validity"] = args.w_validity
        if getattr(args, "w_oracle", None) is not None:
            rl_kw["w_oracle"] = args.w_oracle
        if args.w_diversity is not None:
            rl_kw["w_diversity"] = args.w_diversity
        phase_weights_arg = getattr(args, "phase_weights", None)
        if phase_weights_arg:
            parts = [float(x.strip()) for x in phase_weights_arg.split(",")]
            if len(parts) == 3:
                rl_kw["phase_weights"] = (parts[0], parts[1], parts[2])
        if getattr(args, "batch_normalize_oracle", False):
            rl_kw["batch_normalize_oracle"] = True
        if getattr(args, "use_ppo", False):
            rl_kw["use_ppo"] = True
            rl_kw["ppo_eps"] = getattr(args, "ppo_eps", 0.2)
            rl_kw["ppo_epochs"] = getattr(args, "ppo_epochs", 3)
        if getattr(args, "use_value_baseline", False):
            rl_kw["use_value_baseline"] = True
        if getattr(args, "w_alert", 0) != 0:
            rl_kw["w_alert"] = float(args.w_alert)
        rl_epochs = args.epochs if args.epochs is not None else (_PRODUCTION_RL_EPOCHS if production else _DEFAULT_RL_EPOCHS)
        rl_batch = args.batch_size if args.batch_size is not None else (_PRODUCTION_RL_BATCH if production else _DEFAULT_RL_BATCH)
        rl_kw["accumulation_steps"] = getattr(args, "accumulation_steps", 1)
        config = RLConfig(epochs=rl_epochs, batch_size=rl_batch, **rl_kw)
        resume_path = Path(args.resume) if args.resume else (project_root / "checkpoints" / "generator")
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        if not (resume_path / "model.pt").exists() or not (resume_path / "tokenizer.json").exists():
            raise ValueError(f"RL requires a pretrained checkpoint at {resume_path}.")
        gen = SafeMolGen.from_pretrained(str(resume_path), device=config.device)
        tokenizer = gen.tokenizer
        model = gen.model
        out_dir = Path(args.out) if args.out else (project_root / "checkpoints" / "generator_rl")
        out_dir.mkdir(parents=True, exist_ok=True)
        _resume_state = torch.load(resume_path / "model.pt", map_location=config.device, weights_only=False)
        _resume_cfg = _resume_state.get("config", {})
        rl_model_config = {
            "d_model": _resume_cfg.get("d_model", 256), "nhead": _resume_cfg.get("nhead", 8),
            "num_layers": _resume_cfg.get("num_layers", 6), "dim_feedforward": _resume_cfg.get("dim_feedforward", 512),
            "dropout": _resume_cfg.get("dropout", 0.1), "cond_dim": _resume_cfg.get("cond_dim", 0),
        }
        oracle_score_fn = _make_oracle_score_fn(project_root, device=config.device)
        if getattr(config, "phase_weights", None) and oracle_score_fn is not None:
            phase_fn = _make_oracle_phase_fn(project_root, device=config.device)
            if phase_fn is not None:
                oracle_score_fn = phase_fn
        oracle_prediction_fn = None
        if getattr(config, "w_alert", 0) != 0:
            oracle_prediction_fn = _make_oracle_prediction_fn(project_root, device=config.device)
        target_condition = None
        if getattr(model, "cond_dim", 0) > 0:
            target_condition = get_target_condition_for_rl(device=config.device)
        best_reward = [float("-inf")]

        def rl_epoch_cb(epoch: int, reward: float, validity: float) -> None:
            if reward > best_reward[0]:
                best_reward[0] = reward
                g = SafeMolGen(tokenizer, model)
                (out_dir / "best").mkdir(parents=True, exist_ok=True)
                g.save(str(out_dir / "best"), config=rl_model_config)

        train_rl(model, tokenizer, config=config, oracle_score_fn=oracle_score_fn, oracle_prediction_fn=oracle_prediction_fn, target_condition=target_condition, on_epoch_end=rl_epoch_cb)
        if (out_dir / "best" / "model.pt").exists():
            gen = SafeMolGen.from_pretrained(str(out_dir / "best"), device=config.device)
            model, tokenizer = gen.model, gen.tokenizer
        else:
            gen = SafeMolGen(tokenizer, model)
        model_config_for_save = rl_model_config

    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config = model_config_for_save if model_config_for_save is not None else {"d_model": 256, "nhead": 8, "num_layers": 6, "dim_feedforward": 512, "dropout": 0.1, "cond_dim": getattr(gen.model, "cond_dim", 0)}
    gen.save(str(out_dir), config=save_config)
    print("Done.")


if __name__ == "__main__":
    main()
