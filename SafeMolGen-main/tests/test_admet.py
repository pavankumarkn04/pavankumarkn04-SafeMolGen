import torch
from torch import nn

from models.admet.losses import get_loss
from models.admet.multi_task_predictor import MultiTaskADMETPredictor


def test_get_loss_returns_task_specific_loss():
    assert isinstance(get_loss("classification"), nn.BCEWithLogitsLoss)
    assert isinstance(get_loss("regression"), nn.MSELoss)


def test_multitask_predictor_returns_one_scalar_per_endpoint():
    model = MultiTaskADMETPredictor(
        num_node_features=10,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        endpoint_names=["herg", "ames"],
    )
    x = torch.randn(4, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    ).view(2, -1)
    batch = torch.zeros(4, dtype=torch.long)

    outputs = model(x, edge_index, batch)

    assert set(outputs) == {"herg", "ames"}
    assert outputs["herg"].shape == (1,)
    assert outputs["ames"].shape == (1,)
    assert torch.isfinite(outputs["herg"]).all()
    assert torch.isfinite(outputs["ames"]).all()
