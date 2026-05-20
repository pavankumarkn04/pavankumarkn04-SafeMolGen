"""Loss functions for ADMET tasks."""

from torch import nn


def get_loss(task_type: str) -> nn.Module:
    if task_type == "classification":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()
