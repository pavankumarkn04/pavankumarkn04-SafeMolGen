from types import SimpleNamespace


def test_design_kw_matches_defaults():
    from backend.main import DesignRequest, _design_kw

    kw = _design_kw(DesignRequest())

    assert kw["selection_mode"] == "phase_weighted"
    assert kw["use_oracle_feedback"] is True
    assert kw["exploration_fraction"] == 0.25
    assert kw["use_reranker"] is False


def test_config_exposes_generation_defaults(monkeypatch):
    from backend import main

    monkeypatch.setattr(
        main,
        "_get_pipeline",
        lambda use_rl_model=False: SimpleNamespace(reranker=None, generator_early=None),
    )

    cfg = main.config()

    assert cfg["max_iterations_max"] == 50
    assert cfg["top_k_min"] == 1
    assert cfg["top_k_max"] == 80
    assert cfg["diversity_tanimoto_max_default"] == 0.7
    assert cfg["exploration_fraction_default"] == 0.25
    assert cfg["first_iteration_temperature_default"] == 1.4


def test_load_pipeline_uses_base_generator_without_reranker(monkeypatch):
    from backend import pipeline_loader

    calls = {}

    def fake_from_pretrained(**kwargs):
        calls.update(kwargs)
        return "pipeline"

    monkeypatch.setattr(pipeline_loader, "get_admet_node_feature_dim", lambda _: 42)
    monkeypatch.setattr(
        pipeline_loader,
        "_read_endpoints",
        lambda _: (["ames"], {"ames": "classification"}),
    )
    monkeypatch.setattr(
        pipeline_loader.SafeMolGenDrugOracle,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    out = pipeline_loader.load_pipeline(use_rl_model=False)

    assert out == "pipeline"
    assert calls["generator_path"].endswith("/checkpoints/generator")
    assert calls["reranker_path"] is None
