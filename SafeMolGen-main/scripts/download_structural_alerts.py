"""
Download Hamburg SMARTS dataset (PAINS, optionally Enoch) and write
data/structural_alerts.csv for use by the Oracle.

Sources (free; please cite):
  - Ehrlich, H. C.; Rarey, M. Systematic Benchmark of Substructure Search in Molecular
    Graphs - From Ullmann to VF2. J Cheminform 2012, 4 (1), 13.
  - PAINS: Baell, J. B.; Holloway, G. A. New Substructure Filters for Removal of Pan
    Assay Interference Compounds (PAINS) from Screening Libraries... J. Med. Chem. 2010.
  - Enoch (skin sensitization): Enoch, S. J.; Madden, J. C.; Cronin, M. T. SAR QSAR
    Environ Res 2008, 19 (5-6), 555-578.

Run from project root: python scripts/download_structural_alerts.py
"""

import csv
import re
import tarfile
import urllib.request
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = PROJECT_ROOT / "data" / "structural_alerts.csv"
DATA_DIR = PROJECT_ROOT / "data"

# Hamburg SMARTS dataset URLs (Universität Hamburg)
HAMBURG_BASE = "https://www.zbh.uni-hamburg.de/forschung/amd/datasets/smarts-dataset"
SOURCES = [
    ("pains", f"{HAMBURG_BASE}/pains-smarts-tar.gz", "PAINS", "medium", "Review for assay interference"),
    ("enoch", f"{HAMBURG_BASE}/enoch-smarts-tar.gz", "skin_sensitization", "medium", "Review for skin sensitization"),
]

REGID_PATTERN = re.compile(r'<regId="([^"]+)"')


def _parse_smarts_content(content: str, source_id: str, category: str, severity: str, recommendation: str):
    """Parse Hamburg-style SMARTS file: lines are SMARTS or SMARTS\t<regId="name">."""
    rows = []
    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: SMARTS\t<regId="..."> or just SMARTS
        parts = line.split("\t")
        smarts = parts[0].strip()
        if not smarts:
            continue
        name_match = REGID_PATTERN.search(line)
        if name_match:
            name = name_match.group(1).strip()
        else:
            name = f"{source_id}_{i}"
        # CSV columns: id, name, smarts, category, severity, recommendation
        uid = f"{source_id}_{i}"
        rows.append((uid, name, smarts, category, severity, recommendation))
    return rows


def download_and_parse(source_id: str, url: str, category: str, severity: str, recommendation: str):
    """Download tar.gz, extract SMARTS file, parse and return list of (id, name, smarts, category, severity, recommendation)."""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"Warning: could not download {url}: {e}")
        return []
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile() and (member.name.endswith(".smarts") or "smarts" in member.name.lower()):
                    f = tf.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8", errors="replace")
                        return _parse_smarts_content(content, source_id, category, severity, recommendation)
        # Fallback: no .smarts file name, take first text file
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    f = tf.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8", errors="replace")
                        return _parse_smarts_content(content, source_id, category, severity, recommendation)
    except Exception as e:
        print(f"Warning: could not parse {url}: {e}")
        return []
    return []


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for source_id, url, category, severity, recommendation in SOURCES:
        print(f"Downloading {source_id}...")
        rows = download_and_parse(source_id, url, category, severity, recommendation)
        print(f"  Parsed {len(rows)} alerts")
        all_rows.extend(rows)

    if not all_rows:
        print("No alerts from download. Writing built-in default set (5 alerts).")
        all_rows = [
            ("nitro_aromatic", "Aromatic Nitro", "[$(c1ccccc1[N+](=O)[O-]),$(c1ccncc1[N+](=O)[O-]),$(c1cnccc1[N+](=O)[O-])]", "mutagenicity", "high", "Replace -NO2 with -CN or -CF3"),
            ("aromatic_amine", "Aromatic Amine (Aniline)", "[NH2,NH1,NH0;!$(N-C=O)]c1ccccc1", "mutagenicity", "high", "Convert to amide or replace with -OH/-OCH3"),
            ("nitroso", "Nitroso Group", "[#6]N=O", "mutagenicity", "critical", "Remove nitroso group"),
            ("azo", "Azo Compound", "[#6]N=N[#6]", "mutagenicity", "medium", "Replace azo linkage with amide/ether"),
            ("epoxide", "Epoxide", "C1OC1", "reactivity", "medium", "Avoid epoxide ring"),
        ]
    else:
        print(f"Writing {len(all_rows)} alerts to {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "smarts", "category", "severity", "recommendation"])
        w.writerows(all_rows)

    print("Done.")


if __name__ == "__main__":
    main()
