"""Evaluate Oracle: compute BCE loss on clinical trial data."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.admet.inference import load_model, predict_smiles
from models.oracle.clinical_data import load_clinical_dataset
from models.oracle.phase_predictors import CascadedPhasePredictors
from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim


def main():
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "config" / "endpoints.yaml", "r", encoding="utf-8") as f:
        import yaml
        endpoints_cfg = yaml.safe_load(f)
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}

    admet_ckpt = project_root / "checkpoints" / "admet" / "best_model.pt"
    oracle_ckpt = project_root / "checkpoints" / "oracle" / "best_model.pt"
    data_path = project_root / "data" / "processed" / "oracle" / "clinical_trials.csv"

    if not admet_ckpt.exists():
        raise FileNotFoundError("ADMET checkpoint not found. Train ADMET first.")
    if not oracle_ckpt.exists():
        raise FileNotFoundError("Oracle checkpoint not found. Train Oracle first.")
    if not data_path.exists():
        raise FileNotFoundError(f"Clinical data not found: {data_path}")

    num_node_features = get_admet_node_feature_dim(str(admet_ckpt))
    admet_model = load_model(
        checkpoint_path=str(admet_ckpt),
        endpoint_names=endpoint_names,
        num_node_features=num_node_features,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        device="cpu",
    )

    df = load_clinical_dataset(data_path)
    xs = []
    y1_list, y2_list, y3_list = [], [], []
    for _, row in df.iterrows():
        preds = predict_smiles(
            admet_model, row["smiles"], endpoint_task_types
        )
        xs.append(list(preds.values()))
        y1_list.append(row["phase1"])
        y2_list.append(row["phase2"])
        y3_list.append(row["phase3"])

    x = torch.tensor(xs, dtype=torch.float)
    y1 = torch.tensor(y1_list, dtype=torch.float)
    y2 = torch.tensor(y2_list, dtype=torch.float)
    y3 = torch.tensor(y3_list, dtype=torch.float)

    state = torch.load(oracle_ckpt, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    # Infer in_dim and hidden_dim from checkpoint (Oracle may have been trained with different arch)
    w = model_state["phase1.net.0.weight"]
    in_dim, hidden_dim = int(w.shape[1]), int(w.shape[0])
    model = CascadedPhasePredictors(in_dim=in_dim, hidden_dim=hidden_dim)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        p1, p2, p3 = model(x)
        loss = loss_fn(p1, y1) + loss_fn(p2, y2) + loss_fn(p3, y3)
    bce = loss.item()

    print(f"Oracle eval loss (BCE): {bce:.4f}")
    print("(Lower is better.)")


if __name__ == "__main__":
    main()
