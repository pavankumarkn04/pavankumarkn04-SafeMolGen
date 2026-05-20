"""SMILES tokenizer for SafeMolGen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json
import re


@dataclass
class TokenizerConfig:
    max_length: int = 128


class SMILESTokenizer:
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    SMILES_PATTERN = r"(\[[^\]]+\]|\%\d{2}|Br|Cl|Si|Se|se|As|Te|te|@@|@|\+{1,2}|\-{1,2}|\[|\]|\(|\)|=|#|:|\/|\\|\d|\.)"

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_length: int = 128):
        self.max_length = max_length
        self.pattern = re.compile(self.SMILES_PATTERN)
        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        if vocab is None:
            self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        else:
            self.vocab = vocab
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def fit(self, smiles_list: List[str]) -> None:
        tokens = set()
        for smiles in smiles_list:
            tokens.update(self.tokenize(smiles))
        for tok in sorted(tokens):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self._round_trip_check(smiles_list)

    def _round_trip_check(self, smiles_list: List[str], n_sample: int = 50) -> None:
        try:
            from utils.chemistry import validate_smiles
        except ImportError:
            return
        subset = smiles_list[: min(n_sample, len(smiles_list))]
        failed = 0
        for s in subset:
            decoded = self.decode(self.encode(s))
            if not validate_smiles(decoded):
                failed += 1
        if failed:
            print(f"Tokenizer round-trip: {failed}/{len(subset)} decoded SMILES failed validation")

    def tokenize(self, smiles: str) -> List[str]:
        tokens: List[str] = []
        pos = 0
        while pos < len(smiles):
            remainder = smiles[pos:]
            m = self.pattern.match(remainder)
            if m:
                tokens.append(m.group(1))
                pos += m.end()
            else:
                tokens.append(smiles[pos])
                pos += 1
        return tokens

    def encode(self, smiles: str) -> List[int]:
        tokens = [self.BOS_TOKEN] + self.tokenize(smiles) + [self.EOS_TOKEN]
        ids = [self.vocab.get(tok, self.vocab[self.UNK_TOKEN]) for tok in tokens]
        if len(ids) < self.max_length:
            ids += [self.vocab[self.PAD_TOKEN]] * (self.max_length - len(ids))
        else:
            ids = ids[: self.max_length]
            ids[-1] = self.vocab[self.EOS_TOKEN]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            tok = self.inv_vocab.get(idx, self.UNK_TOKEN)
            if tok in {self.BOS_TOKEN, self.PAD_TOKEN}:
                continue
            if tok == self.EOS_TOKEN:
                break
            if tok == self.UNK_TOKEN:
                continue
            tokens.append(tok)
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        payload = {"vocab": self.vocab, "max_length": self.max_length}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SMILESTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab=payload["vocab"], max_length=payload["max_length"])
