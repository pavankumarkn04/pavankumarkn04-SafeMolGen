import pytest

from utils.chemistry import (
    calculate_properties,
    generate_mutations,
    smiles_to_graph,
    tanimoto_similarity,
    validate_smiles,
)


def test_smiles_to_graph_builds_expected_shapes():
    graph = smiles_to_graph("CCO")

    assert graph is not None
    assert tuple(graph.x.shape) == (3, 10)
    assert tuple(graph.edge_index.shape) == (2, 4)
    assert tuple(graph.edge_attr.shape) == (4, 6)
    assert graph.smiles == "CCO"


def test_calculate_properties_returns_expected_keys_and_ranges():
    props = calculate_properties("CCO")

    assert props is not None
    assert set(props) == {"logp", "mw", "hbd", "hba", "tpsa", "qed", "rotatable_bonds"}
    assert props["mw"] == pytest.approx(46.069, rel=1e-2)
    assert props["hbd"] == 1
    assert props["hba"] == 1
    assert 0.0 <= props["qed"] <= 1.0
    assert props["rotatable_bonds"] >= 0


def test_tanimoto_similarity_handles_identity_and_invalid_smiles():
    assert tanimoto_similarity("CCO", "CCO") == pytest.approx(1.0)
    assert 0.0 <= tanimoto_similarity("CCO", "CCN") < 1.0
    assert tanimoto_similarity("CCO", "not-a-smiles") == 0.0


def test_generate_mutations_returns_unique_valid_non_seed_smiles():
    seed = "CCO"
    muts = generate_mutations(seed, n=5, random_seed=123)

    assert len(muts) > 0
    assert len(muts) == len(set(muts))
    assert seed not in muts
    assert all(validate_smiles(s) for s in muts)
