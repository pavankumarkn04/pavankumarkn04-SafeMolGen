"""Train DrugOracle (Phase 2)."""

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from models.admet.inference import load_model, predict_smiles
from models.oracle.clinical_data import load_clinical_dataset
from models.oracle.phase_predictors import CascadedPhasePredictors
from models.oracle.trainer import OracleTrainer
from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim


class OracleDataset(Dataset):
    def __init__(self, df, admet_model, endpoint_task_types):
        self.df = df.reset_index(drop=True)
        self.admet_model = admet_model
        self.endpoint_task_types = endpoint_task_types

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        preds = predict_smiles(
            self.admet_model, row["smiles"], self.endpoint_task_types
        )
        x = torch.tensor(list(preds.values()), dtype=torch.float)
        return {
            "x": x,
            "phase1": torch.tensor(row["phase1"], dtype=torch.float),
            "phase2": torch.tensor(row["phase2"], dtype=torch.float),
            "phase3": torch.tensor(row["phase3"], dtype=torch.float),
        }


def main():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    endpoints_cfg = yaml.safe_load(
        (project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8")
    )

    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}

    device = torch.device("cpu")
    admet_ckpt = project_root / "checkpoints" / "admet" / "best_model.pt"
    if not admet_ckpt.exists():
        raise FileNotFoundError("ADMET checkpoint not found. Train Phase 1 first.")

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

    data_path = project_root / "data" / "processed" / "oracle" / "clinical_trials.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            "Clinical trial dataset not found at data/processed/oracle/clinical_trials.csv"
        )

    df = load_clinical_dataset(data_path)
    dataset = OracleDataset(df, admet_model, endpoint_task_types)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CascadedPhasePredictors(in_dim=len(endpoint_names))
    trainer = OracleTrainer(model, device=device)

    for epoch in range(1, 11):
        loss = trainer.train_epoch(loader)
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

    ckpt_dir = project_root / "checkpoints" / "oracle"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, ckpt_dir / "best_model.pt")


if __name__ == "__main__":
    main()
