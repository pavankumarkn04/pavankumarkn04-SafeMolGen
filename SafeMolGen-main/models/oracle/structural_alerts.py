"""Structural alerts detection using the RDKit built-in PAINS FilterCatalog.

480 PAINS patterns from Baell & Holloway (2010) are used directly via
``rdkit.Chem.FilterCatalog``.  No external CSV file is required.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


@dataclass
class StructuralAlert:
    name: str
    smarts: str
    category: str
    severity: str
    recommendation: str

    def pattern(self):
        return Chem.MolFromSmarts(self.smarts) if self.smarts else None


def _parse_pains_smarts() -> Dict[str, str]:
    """Return {description: smarts} from RDKit's bundled wehi_pains.csv."""
    pains_csv = Path(Chem.__file__).parent.parent / "Data" / "Pains" / "wehi_pains.csv"
    result: Dict[str, str] = {}
    if not pains_csv.exists():
        logger.warning(f"PAINS CSV not found at {pains_csv}; SMARTS will be empty strings")
        return result
    pattern = re.compile(r'^"(.+)","<regId=(.+)>"$')
    with open(pains_csv, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                result[m.group(2)] = m.group(1)
    return result


def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def _build_structural_alerts_db(
    catalog: FilterCatalog,
    smarts_map: Dict[str, str],
) -> Dict[str, StructuralAlert]:
    db: Dict[str, StructuralAlert] = {}
    for i in range(catalog.GetNumEntries()):
        entry = catalog.GetEntryWithIdx(i)
        desc = entry.GetDescription()
        smarts = smarts_map.get(desc, "")
        db[desc] = StructuralAlert(
            name=desc,
            smarts=smarts,
            category="pains",
            severity="medium",
            recommendation="PAINS alert — review compound for assay interference",
        )
    logger.debug(f"Loaded {len(db)} PAINS structural alerts from RDKit FilterCatalog")
    return db


_PAINS_CATALOG: FilterCatalog = _build_pains_catalog()
_PAINS_SMARTS: Dict[str, str] = _parse_pains_smarts()

STRUCTURAL_ALERTS_DB: Dict[str, StructuralAlert] = _build_structural_alerts_db(
    _PAINS_CATALOG, _PAINS_SMARTS
)


def detect_structural_alerts(smiles: str) -> Tuple[List[str], np.ndarray]:
    """Return (hit_names, per-atom flag array) for PAINS alerts in *smiles*."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], None

    alert_atoms = np.zeros(mol.GetNumAtoms(), dtype=int)
    hits: List[str] = []

    for match in _PAINS_CATALOG.GetMatches(mol):
        hits.append(match.GetDescription())
        for fm in match.GetFilterMatches(mol):
            for pair in fm.atomPairs:
                alert_atoms[pair.target] = 1

    return hits, alert_atoms
