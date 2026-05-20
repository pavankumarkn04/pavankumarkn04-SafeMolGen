import torch

from models.oracle.drug_oracle import DrugOracle
from models.oracle.recommender import generate_recommendations


class _DummyOracleModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return (
            torch.full((batch,), 0.2),
            torch.full((batch,), 0.8),
            torch.full((batch,), -0.4),
        )


def test_generate_recommendations_covers_risks_and_progress():
    recs = generate_recommendations(
        {
            "herg": 0.7,
            "ames": 0.1,
            "dili": 0.45,
            "bioavailability_ma": 0.25,
            "clearance_hepatocyte_az": 90.0,
        },
        ["nitro"],
        prev_admet={
            "herg": 0.8,
            "ames": 0.2,
            "dili": 0.2,
            "bioavailability_ma": 0.15,
            "clearance_hepatocyte_az": 120.0,
        },
    )

    issues = [rec["issue"] for rec in recs]
    types = [rec["type"] for rec in recs]

    assert "Structural Alert" in types
    assert any("hERG inhibition risk" in issue for issue in issues)
    assert any("Low predicted Oral bioavailability" in issue for issue in issues)
    assert any("Improved:" in issue for issue in issues)


def test_drug_oracle_predict_builds_prediction_and_risk_factors(monkeypatch):
    oracle = DrugOracle(
        oracle_model=_DummyOracleModel(),
        admet_model=object(),
        endpoint_task_types={"herg": "classification", "ames": "classification", "dili": "classification"},
        device="cpu",
    )

    monkeypatch.setattr("models.oracle.drug_oracle.validate_smiles", lambda smiles: True)
    monkeypatch.setattr(
        "models.oracle.drug_oracle.predict_smiles",
        lambda model, smiles, endpoint_task_types, device="cpu": {
            "herg": 0.7,
            "ames": 0.2,
            "dili": 0.65,
            "bioavailability_ma": 0.8,
        },
    )
    monkeypatch.setattr(
        "models.oracle.drug_oracle.detect_structural_alerts",
        lambda smiles: (["nitro"], None),
    )

    pred = oracle.predict("CCO")

    assert pred is not None
    assert pred.structural_alerts == ["nitro"]
    assert len(pred.risk_factors) == 2
    assert {risk.name for risk in pred.risk_factors} == {"HERG", "DILI"}
    assert pred.overall_prob < pred.phase2_prob
    assert any(rec["type"] == "Structural Alert" for rec in pred.recommendations)


def test_drug_oracle_returns_none_for_invalid_smiles():
    oracle = DrugOracle(
        oracle_model=_DummyOracleModel(),
        admet_model=object(),
        endpoint_task_types={},
        device="cpu",
    )

    assert oracle.predict("not-a-smiles") is None
