"""API tests for analyze/compare/design routes with edge cases."""

from fastapi.testclient import TestClient

import backend.main as appmod


class _Pred:
    def __init__(self, overall: float):
        self.phase1_prob = 0.6
        self.phase2_prob = 0.55
        self.phase3_prob = 0.52
        self.overall_prob = overall
        self.admet_predictions = {"herg": 0.2, "ames": 0.1, "dili": 0.1}
        self.risk_factors = []
        self.structural_alerts = []
        self.recommendations = [{"type": "Safety", "issue": "none"}]

    def to_dict(self):
        return {
            "phase1_prob": self.phase1_prob,
            "phase2_prob": self.phase2_prob,
            "phase3_prob": self.phase3_prob,
            "overall_prob": self.overall_prob,
            "admet_predictions": self.admet_predictions,
            "risk_factors": [],
            "structural_alerts": self.structural_alerts,
            "recommendations": self.recommendations,
        }


class _Iter:
    def __init__(self, idx: int, pred: _Pred):
        self.iteration = idx
        self.smiles = "CCO"
        self.prediction = pred
        self.improvements = ["overall:+0.02"]
        self.passed_safety = True
        self.used_oracle_feedback = True

    def to_dict(self):
        return {
            "iteration": self.iteration,
            "smiles": self.smiles,
            "prediction": self.prediction.to_dict(),
            "improvements": self.improvements,
            "passed_safety": self.passed_safety,
            "used_oracle_feedback": self.used_oracle_feedback,
        }


class _DesignResult:
    def __init__(self):
        self.final_smiles = "CCO"
        self.final_prediction = _Pred(0.62)
        self.iteration_history = [_Iter(1, self.final_prediction)]
        self.target_achieved = False
        self.total_iterations = 1

    def to_dict(self):
        return {
            "final_smiles": self.final_smiles,
            "final_prediction": self.final_prediction.to_dict(),
            "iteration_history": [i.to_dict() for i in self.iteration_history],
            "target_achieved": self.target_achieved,
            "total_iterations": self.total_iterations,
        }


class _FakePipeline:
    def evaluate_molecule(self, smiles: str):
        if smiles == "invalid":
            return None
        return _Pred(0.61)

    def compare_molecules(self, smiles_list):
        return [{"smiles": s, "prediction": _Pred(0.5 + i * 0.01).to_dict()} for i, s in enumerate(smiles_list)]

    def design_molecule(self, on_iteration_done=None, **_):
        result = _DesignResult()
        if on_iteration_done:
            on_iteration_done(result)
        return result

    def design_molecule_with_restarts(self, **kwargs):
        return self.design_molecule(**kwargs)

    def design_molecule_evolutionary(self, **_):
        return _DesignResult()


def test_analyze_503_when_models_missing(monkeypatch):
    monkeypatch.setattr(appmod, "_get_pipeline", lambda use_rl_model=False: None)
    client = TestClient(appmod.app)
    r = client.post("/api/v1/analyze", json={"smiles": "CCO"})
    assert r.status_code == 503


def test_analyze_edge_cases(monkeypatch):
    monkeypatch.setattr(appmod, "_get_pipeline", lambda use_rl_model=False: _FakePipeline())
    client = TestClient(appmod.app)

    r_empty = client.post("/api/v1/analyze", json={"smiles": "  "})
    assert r_empty.status_code == 400

    r_invalid = client.post("/api/v1/analyze", json={"smiles": "invalid"})
    assert r_invalid.status_code == 400

    r_ok = client.post("/api/v1/analyze", json={"smiles": "CCO"})
    assert r_ok.status_code == 200
    assert r_ok.json()["prediction"]["overall_prob"] > 0


def test_compare_requires_two(monkeypatch):
    monkeypatch.setattr(appmod, "_get_pipeline", lambda use_rl_model=False: _FakePipeline())
    client = TestClient(appmod.app)
    r = client.post("/api/v1/compare", json={"smiles_list": ["CCO"]})
    assert r.status_code == 422 or r.status_code == 400


def test_design_sync_and_stream(monkeypatch):
    monkeypatch.setattr(appmod, "_get_pipeline", lambda use_rl_model=False: _FakePipeline())
    client = TestClient(appmod.app)

    r_sync = client.post("/api/v1/design", json={"target_success": 0.7, "max_iterations": 2})
    assert r_sync.status_code == 200
    body = r_sync.json()
    assert body["_strategy_used"] == "single"
    assert "recommendations" in body

    with client.stream("POST", "/api/v1/design/stream", json={"target_success": 0.7, "max_iterations": 2}) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())
        assert '"event": "started"' in payload
        assert '"event": "iteration"' in payload
        assert '"event": "done"' in payload

