"""Chemistry helpers used by generator and evaluator pipeline."""

from typing import Any, Dict, List, Optional
import random

import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Crippen, DataStructs, Descriptors, Lipinski
from torch_geometric.data import Data


RDLogger.DisableLog("rdApp.error")

_MUTATION_REACTIONS = [
    "[*:1]-[H:2]>>[*:1]-[F]",
    "[*:1]-[H:2]>>[*:1]-[Cl]",
    "[*:1]-[H:2]>>[*:1]-[OH]",
    "[*:1]-[H:2]>>[*:1]-[CH3]",
    "[*:1]-[H:2]>>[*:1]-[OCH3]",
    "[*:1]-[H:2]>>[*:1]-[NH2]",
    "[*:1]-[H:2]>>[*:1]-[CF3]",
    "[*:1]-[H:2]>>[*:1]-[C#N]",
    "[*:1]-[H:2]>>[*:1]-[C(=O)OH]",
]


def validate_smiles(smiles: str) -> bool:
    if not (smiles and smiles.strip()):
        return False
    return Chem.MolFromSmiles(smiles) is not None


def generate_mutations(smiles: str, n: int = 25, random_seed: Optional[int] = None) -> List[str]:
    """Generate up to n single-step valid mutations from a seed SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    rng = random.Random(random_seed)
    reactions = list(_MUTATION_REACTIONS)
    rng.shuffle(reactions)
    seen: set[str] = set()
    out: List[str] = []
    base = Chem.MolFromSmiles(smiles)
    base_canon = Chem.MolToSmiles(base, canonical=True) if base is not None else ""
    for rxn_smarts in reactions:
        if len(out) >= n:
            break
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if rxn is None:
                continue
            products = rxn.RunReactants((mol,))
            for product_tuple in products:
                if len(out) >= n:
                    break
                for product in product_tuple:
                    if product is None:
                        continue
                    try:
                        Chem.SanitizeMol(product)
                        smi = Chem.MolToSmiles(product, canonical=True, allHsExplicit=False)
                        if not smi or smi in seen or smi == base_canon or not validate_smiles(smi):
                            continue
                        seen.add(smi)
                        out.append(smi)
                        break
                    except Exception:
                        continue
        except Exception:
            continue
    return out[:n]


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _atom_features(atom: Chem.Atom) -> List[float]:
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        *[1.0 if atom.GetHybridization() == h else 0.0 for h in hybridizations],
    ]


def _bond_features(bond: Chem.Bond) -> List[float]:
    b = bond.GetBondType()
    return [
        1.0 if b == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if b == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if b == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if b == Chem.rdchem.BondType.AROMATIC else 0.0,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])
    if edge_index:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_t = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = torch.empty((0, 6), dtype=torch.float)
    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t, smiles=smiles)


def calculate_properties(smiles: str) -> Optional[Dict[str, Any]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "logp": Crippen.MolLogP(mol),
        "mw": Descriptors.MolWt(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "qed": Descriptors.qed(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
    }


class MoleculeProcessor:
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        return smiles_to_graph(smiles)
