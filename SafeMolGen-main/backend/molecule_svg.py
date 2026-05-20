"""2D molecule SVG via RDKit."""
from typing import Optional, List

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def normalize_smiles_for_display(smiles: str) -> Optional[str]:
    """Return canonical SMILES without isotopes or explicit H. Use for drawing and display."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            atom.SetIsotope(0)
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=False)
    except Exception:
        return None
    return canonical


def draw_molecule_2d(
    smiles: str,
    highlight_atoms: Optional[List[int]] = None,
    size: tuple = (400, 300),
) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            atom.SetIsotope(0)
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=False)
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            return None
    except Exception:
        pass
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    try:
        opts = drawer.drawOptions()
        opts.useBWPalette = False
        if hasattr(opts, "bondLineWidth"):
            opts.bondLineWidth = 2
        if hasattr(opts, "minFontSize"):
            opts.minFontSize = 10
    except Exception:
        pass
    if highlight_atoms:
        colors = {i: (1, 0, 0) for i in highlight_atoms}
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=colors)
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
