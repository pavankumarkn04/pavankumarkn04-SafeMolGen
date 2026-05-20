"""Checkpoint utilities for loading and inferring dimensions."""

from pathlib import Path

import torch

DEFAULT_ADMET_NODE_FEATURES = 10


def get_admet_node_feature_dim(checkpoint_path: str) -> int:
    path = Path(checkpoint_path)
    if not path.exists():
        return DEFAULT_ADMET_NODE_FEATURES
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        key = "encoder.convs.0.nn.0.weight"
        if key not in model_state:
            return DEFAULT_ADMET_NODE_FEATURES
        return int(model_state[key].shape[1])
    except Exception:
        return DEFAULT_ADMET_NODE_FEATURES

