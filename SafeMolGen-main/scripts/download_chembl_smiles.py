#!/usr/bin/env python3
"""Download ChEMBL chemreps and extract SMILES for SafeMolGen.

Output: data/processed/generator/smiles.tsv with a 'canonical_smiles' column.
Source file: chembl_36_chemreps.txt.gz
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import ssl
import certifi


CHEMBL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
    "chembl_36_chemreps.txt.gz"
)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "SafeMolGen-Downloader"})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, context=ssl_context) as response, open(dest, "wb") as f:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading ChEMBL",
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def extract_smiles(
    gz_path: Path, out_path: Path, limit: int | None = None
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f_in:
        reader = pd.read_csv(
            f_in,
            sep="\t",
            usecols=["chembl_id", "canonical_smiles"],
            chunksize=100000,
        )
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write("chembl_id\tcanonical_smiles\n")
            for chunk in reader:
                if limit is not None:
                    remaining = limit - count
                    if remaining <= 0:
                        break
                    chunk = chunk.head(remaining)
                chunk.to_csv(f_out, sep="\t", index=False, header=False)
                count += len(chunk)
                if limit is not None and count >= limit:
                    break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/processed/generator/smiles.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--raw",
        default="data/raw/chembl_36_chemreps.txt.gz",
        help="Raw ChEMBL chemreps path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick tests",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / args.raw
    out_path = project_root / args.out

    if not raw_path.exists():
        print(f"Downloading ChEMBL chemreps to {raw_path} ...")
        download_file(CHEMBL_URL, raw_path)

    print(f"Extracting SMILES to {out_path} ...")
    count = extract_smiles(raw_path, out_path, limit=args.limit)
    print(f"Done. Wrote {count} rows.")


if __name__ == "__main__":
    main()
