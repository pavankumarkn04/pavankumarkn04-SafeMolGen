"""Tests for RDKit PAINS-based structural alerts."""

from models.oracle.structural_alerts import (
    STRUCTURAL_ALERTS_DB,
    StructuralAlert,
    detect_structural_alerts,
)


def test_structural_alerts_db_uses_pains():
    """DB should contain all 480 PAINS entries."""
    assert isinstance(STRUCTURAL_ALERTS_DB, dict)
    assert len(STRUCTURAL_ALERTS_DB) == 480


def test_structural_alerts_db_entries_have_required_fields():
    for alert in STRUCTURAL_ALERTS_DB.values():
        assert isinstance(alert, StructuralAlert)
        assert alert.name
        assert alert.category == "pains"
        assert alert.smarts  # SMARTS populated from wehi_pains.csv


def test_detect_structural_alerts_known_pains_hit():
    """catechol is a known PAINS hit (catechol_A)."""
    hits, atoms = detect_structural_alerts("Oc1ccccc1O")
    assert any("catechol" in h.lower() for h in hits), f"Expected catechol hit, got: {hits}"
    assert atoms is not None
    assert int(atoms.sum()) >= 1


def test_detect_structural_alerts_clean_molecule():
    """Simple alkane should return no PAINS alerts."""
    hits, atoms = detect_structural_alerts("CCCC")
    assert hits == []
    assert atoms is not None
    assert int(atoms.sum()) == 0


def test_detect_structural_alerts_invalid_smiles():
    hits, atoms = detect_structural_alerts("not_a_smiles$$")
    assert hits == []
    assert atoms is None


def test_detect_structural_alerts_returns_names_resolvable_by_pipeline():
    """All returned hit names must exist as keys in STRUCTURAL_ALERTS_DB."""
    hits, atoms = detect_structural_alerts("Oc1ccccc1O")
    for name in hits:
        assert name in STRUCTURAL_ALERTS_DB
