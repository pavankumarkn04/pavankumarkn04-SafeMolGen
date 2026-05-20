"""Regression: property targets must match frontend shape (main SafeMolGen-DrugOracle semantics)."""

from models.integrated.pipeline import (
    _filter_by_property_targets,
    _has_polar_or_aromatic_scaffold,
    _is_druglike_complexity,
    _passes_rotatable_cap,
)

_DEFAULT_UI_TARGETS = {
    "logp": [2, 5],
    "mw_min": 150,
    "mw": 500,
    "hbd": 5,
    "hba": 10,
    "tpsa": 140,
    "qed": 0.5,
}


def test_long_alkane_rejected_by_property_targets():
    grease = "CCCCCCCCCCCCCCCCCCCC"
    assert _filter_by_property_targets([grease], _DEFAULT_UI_TARGETS) == []


def test_grease_fails_polar_or_aromatic_gate():
    grease = "CCCCCCCCCCCCCCCCCCCC"
    assert _is_druglike_complexity(grease) is True
    assert _has_polar_or_aromatic_scaffold(grease) is False


def test_typical_aromatic_drug_passes_gates():
    smi = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    assert _has_polar_or_aromatic_scaffold(smi) is True
    assert _is_druglike_complexity(smi) is True
    assert len(_filter_by_property_targets([smi], _DEFAULT_UI_TARGETS)) == 1


def test_rotatable_cap_none_means_unlimited():
    assert _passes_rotatable_cap("CCCCCCCCCCCCCCCCCCCC", None) is True


def test_rotatable_cap_rejects_long_flexible_chain():
    grease = "CCCCCCCCCCCCCCCCCCCC"
    assert _passes_rotatable_cap(grease, 10) is False
