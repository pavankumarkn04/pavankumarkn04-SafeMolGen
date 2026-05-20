"""Inference utilities for ADMET predictor."""

from typing import Dict, List

import numpy as np
import torch

from models.admet.multi_task_predictor import MultiTaskADMETPredictor
from utils.chemistry import MoleculeProcessor


def load_model(
    checkpoint_path: str,
    endpoint_names: List[str],
    num_node_features: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: str = "cpu",
) -> MultiTaskADMETPredictor:
    model = MultiTaskADMETPredictor(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        endpoint_names=endpoint_names,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def predict_smiles(
    model: MultiTaskADMETPredictor,
    smiles: str,
    endpoint_task_types: Dict[str, str],
    device: str = "cpu",
) -> Dict[str, float]:
    graph = MoleculeProcessor().smiles_to_graph(smiles)
    if graph is None:
        return {}
    if graph.x.size(0) < 2 or graph.edge_index.size(1) == 0:
        return {}
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)
    with torch.no_grad():
        outputs = model(graph.x, graph.edge_index, graph.batch)
    results = {}
    for name, value in outputs.items():
        v = float(value.detach().cpu().item())
        if endpoint_task_types.get(name) == "classification":
            v = float(1 / (1 + np.exp(-v)))
        results[name] = v
    return results

