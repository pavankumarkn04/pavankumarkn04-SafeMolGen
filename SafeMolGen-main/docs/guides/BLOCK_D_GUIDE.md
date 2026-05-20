# Block D — DrugOracle

> Companion file to `LEARNING_GUIDE.md`. Block D content lives here to keep the main guide from growing unwieldy.
>
> **Prerequisites:** Blocks A (chemistry), B (generator + RL), C (ADMET GNN). If any of those are unfamiliar, read them first in `LEARNING_GUIDE.md`.

---

## Block D orientation

### 1. What Block D is for

Block C answered 22 mechanistic ADMET questions: "what is this molecule's logP?", "is it hERG-toxic?", "what's its solubility?". Block D answers a fundamentally different question:

> **Given this molecule's ADMET profile, what is the probability it will clear Phase I, Phase II, and Phase III clinical trials, and make it to market?**

This is the scalar that B.5's RL reward ultimately chases. Everything in the pipeline — generation, steering, reranking — is in service of maximizing this number.

### 2. Why Block D has to exist as a separate block

The ADMET model (Block C) predicts individual pharmacokinetic and toxicity endpoints. What it does *not* predict is the thing pharma cares about most: "will this molecule survive phase 1 → phase 2 → phase 3 → approval?" That outcome depends on ADMET properties in a non-linear way (a safe-but-unabsorbed drug fails; an absorbed-but-hepatotoxic drug fails differently), plus factors the GNN cannot see (target-pathway relevance, formulation, trial design).

Block D approximates this by taking the 22 ADMET predictions as a feature vector and training a small stacked model on **historical clinical-trial outcomes**. It is deliberately modest — 3 small MLPs on top of 22 floats — because clinical-trial data is scarce (~thousands of labeled drugs at best) and the ADMET layer below already does the heavy lifting.

### 3. The attrition problem Block D models

The real-world attrition of drug candidates is brutal:

| Stage | Approx. success rate |
|---|---|
| Preclinical → Phase I | ~60% of tested compounds |
| Phase I → Phase II | ~52% |
| Phase II → Phase III | ~29% |
| Phase III → FDA approval | ~58% |
| **Overall: preclinical → market** | **~9%** |

(Figures from Wong C. H., Siah K. W., Lo A. W., *Biostatistics* 20: 273–286, 2019 — the canonical dataset-backed reference.)

Most compounds fail in Phase II — the "valley of death," where efficacy has to be proven in patients. Failures cluster around predictable issues: hepatotoxicity (DILI), cardiac toxicity (hERG), bioavailability problems, metabolic instability. These are exactly the 22 endpoints Block C predicts. Block D learns the mapping from "22 ADMET values" → "phase-by-phase success probability."

### 4. The core idea in one line

\[
\text{SMILES} \xrightarrow{\text{Block C}} \underbrace{22 \text{ ADMET floats}}_{\text{mechanistic}} \xrightarrow{\text{Block D phase MLPs}} \{p_1, p_2, p_3\} \xrightarrow{\text{weighted combine} + \text{penalties}} \underbrace{\text{overall probability } p}_{\text{clinical}}
\]

Plus a lookup-based structural-alert module that penalizes known toxic substructures directly from the SMILES.

### 5. Why cascaded predictors, not one model

A naive approach would train one MLP to predict `P(approved)` directly. The repo uses a **cascade** — three separate predictors, each conditioned on the previous phase's success probability:

```
         ADMET (22-dim)
           │
           ▼
      Phase 1 MLP  →  p1 (probability of clearing Phase I)
      │                │
      │                └──────────┐
      │                           ▼
      ├───────────────► [ADMET ‖ σ(p1)] (23-dim)
      │                           │
      │                           ▼
      │                      Phase 2 MLP  →  p2
      │                           │
      │                           └───┐
      │                               ▼
      └───────────────────► [ADMET ‖ σ(p1) ‖ σ(p2)] (24-dim)
                                      │
                                      ▼
                                 Phase 3 MLP  →  p3
```

Each phase predictor sees the raw ADMET features **plus** the sigmoided outputs of the previous phases. This reflects clinical reality: Phase II success depends on whether the molecule was already filtered through Phase I safety; Phase III success depends on both prior phases. A cascade captures that conditional dependency structure; three independent predictors would double-count safety signals (a hepatotoxic drug fails all three phases *for the same reason*).

### 6. The four components of Block D

```123:144:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
        probs = self._predict_oracle(admet_preds)
        alerts, alert_atoms = detect_structural_alerts(smiles)
        recs = generate_recommendations(admet_preds, alerts)
        risk_factors = []
        for name, val in admet_preds.items():
            if name in {"herg", "ames", "dili"} and val > 0.5:
                risk_factors.append(...)
        overall = self._clinical_quality(probs["phase1"], probs["phase2"], probs["phase3"], admet_preds, alerts)
```

| Component | File | Role |
|---|---|---|
| **Cascaded phase predictors** | `models/oracle/phase_predictors.py` | 3 stacked MLPs producing p₁, p₂, p₃ — the trainable neural part |
| **Structural-alert matcher** | `models/oracle/structural_alerts.py` | SMARTS-based substructure lookup — non-trainable, rule-based |
| **Clinical-quality scorer** | `DrugOracle._clinical_quality` (inline in `drug_oracle.py`) | weighted sum of phase probs minus penalties → one overall number |
| **Recommender** | `models/oracle/recommender.py` | human-readable suggestions based on ADMET + alerts — not used by the generator or reward system |

The generator's RL reward depends on the first three. The recommender is an output-side convenience for the inference API.

### 7. Block D architecture at a glance

```
DrugOracle (orchestrator, models/oracle/drug_oracle.py)
  ├── admet_model: MultiTaskADMETPredictor (Block C — frozen at serve time)
  ├── oracle_model: CascadedPhasePredictors (the new part)
  │     ├── phase1: MLP(22 → 256 → 256 → 1)
  │     ├── phase2: MLP(23 → 256 → 256 → 1)   ← 22 ADMET + 1 phase1 prob
  │     └── phase3: MLP(24 → 256 → 256 → 1)   ← 22 ADMET + phase1 + phase2
  ├── structural_alerts (rule-based SMARTS matcher)
  │     └── 2+ SMARTS patterns (aromatic nitro, aniline, loadable from CSV)
  └── composite scorer (weighted sum + penalties)
```

**Params:** ~210k across the 3 MLPs. Tiny next to the generator (~2.5M) and close in size to the ADMET model (~104k). The cascade structure is the distinctive architectural choice.

### 8. The composite score — the "Oracle value" RL uses

Phase probabilities are combined into one scalar through a weighted sum, then penalized for toxicity and structural alerts:

\[
\text{overall} = \text{clip}_{[0,1]} \Big( 0.2\, p_1 + 0.5\, p_2 + 0.3\, p_3 - \min(0.5,\; 0.12 \cdot |\{\text{hERG, AMES, DILI}\} \cap \{x > 0.5\}| + 0.08 \cdot |\text{alerts}|) \Big)
\]

Implementation:

```15:20:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
PHASE_WEIGHT_1 = 0.2
PHASE_WEIGHT_2 = 0.5
PHASE_WEIGHT_3 = 0.3
RISK_PENALTY_PER_ENDPOINT = 0.12
STRUCTURAL_ALERT_PENALTY = 0.08
MAX_RISK_PENALTY = 0.5
```

```92:100:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
    def _clinical_quality(self, p1, p2, p3, admet, alerts) -> float:
        base = PHASE_WEIGHT_1 * p1 + PHASE_WEIGHT_2 * p2 + PHASE_WEIGHT_3 * p3
        penalty = 0.0
        for key in ("herg", "ames", "dili"):
            if admet.get(key, 0) > 0.5:
                penalty += RISK_PENALTY_PER_ENDPOINT
        penalty += len(alerts) * STRUCTURAL_ALERT_PENALTY
        penalty = min(penalty, MAX_RISK_PENALTY)
        return max(0.0, min(1.0, base - penalty))
```

Three design choices baked into the numbers:

- **Phase weights 0.2 / 0.5 / 0.3:** phase 2 is the hardest and most informative stage, so it gets the highest weight.
- **Toxicity penalties:** hERG / AMES / DILI above 0.5 each subtract 0.12. These are the classical "killer" flags.
- **Structural alert penalty 0.08 per alert, capped at 0.5:** prevents a molecule with 10 alerts from getting overwhelmingly negative score; caps the damage.

All numbers are hand-tuned heuristics, not learned. **⚠️ Known design choice** — a more principled version would learn the weights jointly with phase probabilities from end-to-end approval data, but the training set for that is tiny (~1–2k approved drugs).

The `overall` scalar is what B.5's `rewards.py _oracle_scalar` extracts. It's the one number that flows all the way back to the generator's RL gradient.

### 9. What Block D is NOT

- Not a brand-new paradigm: it's straightforward supervised learning on clinical-trial data.
- Not a survival model: it predicts binary success per phase, not time-to-event.
- Not calibrated: outputs are raw sigmoided logits, not probability-calibrated.
- Not a replacement for wet-lab testing: it's a **prior** that prioritizes candidates; the real trials still have to run.

### 10. Data Block D needs

A CSV with columns `smiles, phase1, phase2, phase3` at `data/processed/oracle/clinical_trials.csv`, each phase column a binary indicator (1 = succeeded, 0 = failed).

Public datasets that could populate this (varying degrees of effort to process):

- **ClinicalTrials.gov** — millions of trial records; needs text mining to extract drug SMILES and outcomes.
- **DrugBank** (non-commercial license) — ~13k approved + investigational drugs with phase status.
- **ChEMBL** — has some trial-phase metadata.
- **Citeline / PharmaProjects** — commercial, most reliable, expensive.
- **Aggregate estimates from Wong et al. 2019** — gives population success rates, not per-molecule labels; useful for calibration sanity checks.

**⚠️ Data availability is the weakest link in Block D.** With fewer than ~1000 labeled molecules per phase, the cascade risks overfitting; the current `hidden_dim=256` requires several thousand examples for stable fitting.

### 11. Training-loop snapshot

| Parameter | Value | Source |
|---|---|---|
| Phase predictor hidden dim | 256 | `CascadedPhasePredictors.__init__` default |
| Phase predictor dropout | 0.15 | same |
| Optimizer | Adam, lr=1e-3, wd=1e-4 | `OracleTrainer.__init__` defaults |
| Loss | `BCEWithLogitsLoss` × 3, summed | `OracleTrainer.train_epoch` line 38 |
| Epochs | 10 | `scripts/train_oracle.py` line 82 (hard-coded) |
| Batch size | 64 | `scripts/train_oracle.py` line 77 |
| Scheduler | none | absent |
| Val split / early stopping | none | absent |
| Checkpoint | overwrite after full run | `scripts/train_oracle.py` lines 86–88 |

Compared to Block C (80 epochs, plateau scheduler, val-based checkpoint), Block D is deliberately simpler:

- 10 epochs, not 80 — data is small (~1k drugs vs. ~80k endpoint samples); more epochs would memorize.
- No scheduler — short training doesn't need LR decay.
- No early stopping or val — dataset is too small to justify a val split; last-epoch weights saved. **⚠️ Gap 8.**
- Hard-coded epoch count (not in config.yaml) — **⚠️ Gap 9** (config inconsistency vs. Block C).

### 12. How Block D fits in the full system

```
SMILES
  │
  ├──► Block C (ADMET GNN) ──► 22 ADMET predictions
  │                                    │
  │                                    ▼
  │                            Block D — CascadedPhasePredictors
  │                                    │
  │                                    ▼
  │                            {p₁, p₂, p₃}
  │                                    │
  │                                    ▼
  │                            Structural alerts (SMARTS lookup)
  │                                    │
  │                                    ▼
  │                     _clinical_quality → overall_prob ∈ [0,1]
  │                                    │
  │                                    ├──► Generator reward (B.5) — the Oracle scalar in REINFORCE/PPO
  │                                    ├──► Condition vector positions 22–24 (C.7) — phase probs
  │                                    └──► UI / recommendations (not in the learning loop)
  │
  └──► SafeMolGen (Block B) ──► new SMILES ──► back to Block C
```

Closed loop: generator → ADMET → Oracle → reward → generator. Block D is the feedback source that tells the generator "produce molecules that look like clinically successful drugs."

### 13. Step plan for Block D

| Step | Content |
|---|---|
| **D.1** | Clinical trial data — sources, schema, labeling strategy, class imbalance, what the repo expects on disk |
| **D.2** | `CascadedPhasePredictors` architecture — per-phase MLP structure, the concatenation mechanics, param count |
| **D.3** | Training loop — `OracleTrainer`, Adam 1e-3 / wd 1e-4, BCE × 3 summed, 10 epochs, batch 64; why no scheduler, what breaks without a val split |
| **D.4** | Structural alerts — SMARTS substructure matching, built-in vs. CSV-loaded alert database, `detect_structural_alerts` mechanics |
| **D.5** | Composite clinical-quality score — phase weights, toxicity penalties, alert penalties, clipping; worked aspirin example; how this becomes the RL oracle scalar |
| **D.6** | `DrugOracle.predict` serving path — `OraclePrediction` dataclass, risk factors, recommendations, how the integrated pipeline and the reranker consume it |

After Block D, the learning path ends: Block A (chemistry), B (generator + RL), C (ADMET GNN), D (Oracle) together form the full loop.

### 14. What's novel in Block D

| Component | Status | Source |
|---|---|---|
| Clinical attrition statistics (used for context) | Reused | Wong C. H., Siah K. W., Lo A. W., *Biostatistics* 20: 273–286, 2019, DOI 10.1093/biostatistics/kxx069 |
| Cascaded binary predictors (σ(p₁) → phase₂ input, etc.) | Project-specific architectural choice — sequential conditioning is common in ML, but this specific application to clinical phases is the project's contribution |
| SMARTS structural alerts | Reused | Brenk R. et al., *ChemMedChem* 3: 435–444, 2008, DOI 10.1002/cmdc.200700139; Bruns R. F., Watson I. A., *J. Med. Chem.* 55: 9763–9772, 2012 |
| Hand-weighted composite score | Project-specific heuristic — no principled derivation |
| MLP phase predictor architecture | Reused (standard) — `Linear → ReLU → Dropout` stacks |
| 22 ADMET floats → downstream phase classifier | Reused — standard multi-task-then-downstream pattern |

### 15. Verified?

| Claim | Source tier | Location |
|---|---|---|
| Oracle = 3 cascaded MLPs + alert lookup + composite | `repo` | `models/oracle/drug_oracle.py`, `phase_predictors.py`, `structural_alerts.py` |
| Phase 2 input = 22 ADMET + sigmoid(phase1) | `repo` | `phase_predictors.py` lines 32, 38 |
| Phase 3 input = 22 ADMET + sigmoid(phase1) + sigmoid(phase2) | `repo` | `phase_predictors.py` lines 33, 40 |
| Phase weights 0.2 / 0.5 / 0.3 | `repo` | `drug_oracle.py` lines 15–17 |
| Risk penalty 0.12 per flagged endpoint | `repo` | `drug_oracle.py` lines 18, 95–97 |
| Structural alert penalty 0.08 per match, cap 0.5 | `repo` | `drug_oracle.py` lines 19–20, 98–99 |
| Clip to [0, 1] | `repo` | `drug_oracle.py` line 100 |
| Training: 10 epochs, batch 64, Adam lr=1e-3 wd=1e-4 | `repo` | `scripts/train_oracle.py` lines 77, 82; `trainer.py` lines 21–22 |
| Loss = BCEWithLogits × 3 summed | `repo` | `trainer.py` line 38 |
| No val split, no scheduler, no early stopping | `repo` | full-file read of `scripts/train_oracle.py` |
| Clinical-trial "valley of death" concept (phase 2 hardest) | `web` | DiMasi J. A. et al., *J. Health Econ.* 47: 20–33, 2016; Wong C. H. et al., *Biostatistics* 20: 273–286, 2019 |
| Brenk structural alerts list | `web` | Brenk R. et al., *ChemMedChem* 3: 435–444, 2008 |

---

✅ Block D orientation complete. Running gap ledger (carried over from Block C):

- Gap 1 (B.4): no training-corpus novelty filter
- Gap 2 (C.1): missing `scripts/featurize_admet.py`
- Gap 3 (C.2): `edge_attr` ignored by GIN
- Gap 4 (C.5): no `pos_weight` on BCE
- Gap 5 (C.6): no scale-normalization in score aggregation; no `last_model.pt` snapshot
- Gap 6 (C.7): no per-dim normalization of ADMET before condition injection
- Gap 7 (C.7): no batched `predict_smiles`
- **Gap 8 (D orientation):** no val split or early stopping in Oracle training
- **Gap 9 (D orientation):** Oracle training hyperparameters hard-coded in script, not config-driven

Say **"next"** for Step D.1 (clinical-trial dataset — sources, schema, labeling, the data-availability problem).

---

## Step D.1 — clinical-trial dataset

### 1. The question

Block D trains on a CSV the repo expects to exist at `data/processed/oracle/clinical_trials.csv`. This step covers:

1. The exact schema the code demands.
2. What each column means scientifically.
3. Where the data comes from (no download script ships with the repo).
4. Class imbalance realities.
5. Missing-data handling.
6. What's absent and flagged.

### 2. The schema — four columns, strict contract

```8:15:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/clinical_data.py
def load_clinical_dataset(path: Path) -> pd.DataFrame:
    """Load clinical dataset with SMILES and phase labels."""
    df = pd.read_csv(path)
    required = {"smiles", "phase1", "phase2", "phase3"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    return df
```

Four required columns, nothing else validated:

| Column | dtype expected | Meaning |
|---|---|---|
| `smiles` | string | canonical or non-canonical SMILES of the compound |
| `phase1` | 0 or 1 | did the compound clear Phase I trials? |
| `phase2` | 0 or 1 | did it clear Phase II? |
| `phase3` | 0 or 1 | did it clear Phase III? |

No splits, no timestamps, no drug identifiers, no target/indication metadata, no patient counts. The model uses only what's in these four columns.

### 3. What "phase cleared" means — label semantics

These labels encode a binary classification per phase. The standard convention (and the one the composite score in orientation §8 assumes) is:

- `phase1 = 1` → at least one Phase I trial of this compound completed with a reported result and the compound advanced (or is still in development past Phase I).
- `phase1 = 0` → a Phase I trial was initiated but the compound did not progress (failure due to safety, tolerability, PK, or abandonment).

Labels are **not** mutually exclusive or cumulative by convention. A compound that cleared Phase I and failed Phase II would be `(1, 0, 0)`. A compound currently in Phase III with no failure yet would be `(1, 1, ?)` — the `?` is the unresolved censoring problem (see §7).

The cascaded phase predictors in Block D learn to respect this structure only because of the architecture (Phase 2's input includes σ(p₁), Phase 3's includes both). The labels themselves don't encode dependencies.

### 4. Where the data comes from

The repo does **not** ship a download script for the clinical-trial dataset. `scripts/train_oracle.py` assumes the file exists and fails if not:

```70:73:/Users/sreevardhandesu/Desktop/prj_demo/scripts/train_oracle.py
    data_path = project_root / "data" / "processed" / "oracle" / "clinical_trials.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            "Clinical trial dataset not found at data/processed/oracle/clinical_trials.csv"
        )
```

Running `scripts/train_oracle.py` on a fresh clone raises this error. **⚠️ Gap 10 — no dataset acquisition script.**

The expected sources:

| Source | Coverage | Effort | Licensing |
|---|---|---|---|
| **ClinicalTrials.gov API** | ~500k trials, millions of outcomes since 1997 | High — needs text mining to extract drug SMILES + map to outcomes | Public domain |
| **DrugBank** | ~13k approved + investigational drugs with status | Medium — structured but phase info patchy | Non-commercial free; commercial paid |
| **ChEMBL assays with max_phase** | ~2M compounds; `max_phase` field ∈ {0, 1, 2, 3, 4} | Low — SQL filter; need to derive binary phase labels | Open |
| **Wong et al. 2019 dataset** | Trial-level, aggregate success rates | Low — but gives population rates, not per-molecule labels | Supplementary CSV |
| **Citeline / Pharmaprojects** | Commercial drug-level phase tracking | Low if licensed | Expensive |

The easiest bootstrap in practice:

1. Query **ChEMBL** for `max_phase ≥ 1` compounds, pull SMILES + phase.
2. Derive binary columns: `phase1 = 1 if max_phase >= 1 else 0`; similarly for 2 and 3.
3. Augment with failure cases from ClinicalTrials.gov or published failure lists.

This gives a few thousand positives but very few labeled failures — which is the root class-imbalance problem.

### 5. Class imbalance — the invisible skew

Public trial data has a severe reporting bias: **successes are recorded more completely than failures.** Failures often become "discontinued without specific reason given" or simply vanish from databases. A naive ChEMBL scrape will look like:

| Phase | Positives (cleared) | Negatives (failed) |
|---|---|---|
| phase1 | ~10,000 | <1,000 (under-reported) |
| phase2 | ~5,000 | <1,500 |
| phase3 | ~3,000 | <500 |

So the labels are dominated by "cleared" cases, which is exactly backwards from the industry reality (9% overall market success from orientation §3). A model trained on raw public data will be over-optimistic unless:

- **Failure-augmentation:** explicitly add known Phase II/III failures (scraped from FDA CRLs, published retrospective analyses, company press releases).
- **Class weighting:** apply `pos_weight` to `BCEWithLogitsLoss` — which the repo does not currently do (see Gap 4 from Block C; same issue applies here). **⚠️ Gap 11.**
- **Negative sampling:** treat preclinical-only compounds as "implicit Phase 1 failures" (controversial but sometimes used).

The repo does none of these. This is the **central weakness of Block D's current training pipeline** — the model's calibration depends entirely on the quality of the CSV you hand it, and the default public-data assembly will bias it toward over-prediction.

### 6. Missing-data handling in the current loader

```26:37:/Users/sreevardhandesu/Desktop/prj_demo/scripts/train_oracle.py
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        preds = predict_smiles(
            self.admet_model, row["smiles"], self.endpoint_task_types
        )
        x = torch.tensor(list(preds.values()), dtype=torch.float)
        return {
            "x": x,
            "phase1": torch.tensor(row["phase1"], dtype=torch.float),
            "phase2": torch.tensor(row["phase2"], dtype=torch.float),
            "phase3": torch.tensor(row["phase3"], dtype=torch.float),
        }
```

No handling of NaN labels. Two failure modes:

1. **Invalid SMILES:** `predict_smiles` returns `{}` for molecules RDKit rejects. `list({}.values())` is `[]`; `torch.tensor([], dtype=torch.float)` is a zero-length tensor. The downstream `CascadedPhasePredictors.forward(x)` expects shape `(B, 22)` and will crash on shape `(B, 0)`.
2. **Missing phase label:** `pd.read_csv` fills empty cells with `NaN`. `torch.tensor(float('nan'))` is legal, but `BCEWithLogitsLoss(logit, nan)` returns `nan` and poisons the whole batch gradient.

**⚠️ Gap 12 — no pre-filter for invalid SMILES or missing labels in the Oracle loader.** A robust loader would drop rows where `predict_smiles` returns empty or any phase is NaN, ideally at `__init__` time rather than `__getitem__` time.

### 7. What's NOT in the schema

Five ordinary things a serious clinical-trial dataset would include that this one doesn't:

- **Censoring indicators:** "in Phase III, outcome pending" — currently conflated with "failed."
- **Time-to-event:** how long was the trial, when did it fail. Enables survival models (which this doesn't use — orientation §9).
- **Indication / therapeutic area:** oncology drugs have very different success rates from cardiovascular drugs. A single Oracle across all indications fundamentally averages out large differences.
- **Target protein:** the biological target matters. Kinase inhibitors, GPCR ligands, and antibodies have wildly different clinical trajectories.
- **Drug class / modality:** small molecule vs. biologic vs. peptide vs. ADC.

The repo chooses the narrowest possible schema (SMILES + three phase bits) and lets Block C's 22 ADMET predictions carry the structural information. This is a reasonable simplification for a small-scale project but would fail in production — industry Oracle models use 50+ features.

### 8. Data transformation trace — one row to one training sample

| Stage | Representation | Source |
|---|---|---|
| 0 | CSV row: `aspirin, 1, 1, 0` | user-provided `clinical_trials.csv` |
| 1 | `df.iloc[idx]` → pandas Series | `OracleDataset.__getitem__` line 27 |
| 2 | ADMET forward pass via `predict_smiles` | Block C inference (C.7) |
| 3 | 22-dim ADMET dict | `predict_smiles` return |
| 4 | `x = Tensor(22)` (order: dict insertion order, which is model head order) | `torch.tensor(list(preds.values()))` |
| 5 | `phase1 = Tensor(0.0 or 1.0)` etc. | `torch.tensor(row["phase1"])` |
| 6 | Dict `{"x": (22,), "phase1": (), "phase2": (), "phase3": ()}` | dataloader collation |
| 7 | Batched: `{"x": (64, 22), "phase1": (64,), "phase2": (64,), "phase3": (64,)}` | PyTorch `DataLoader` |

Notice line 4: **`list(preds.values())`**. This depends on the dict ordering returned by `predict_smiles`, which in turn depends on the `ModuleDict` iteration order of the ADMET model's heads. `nn.ModuleDict` preserves insertion order (Python 3.7+ dicts do), and the insertion order matches the `endpoint_names` list from the YAML. So the 22-dim feature vector positions are:

| Position | Endpoint | Source |
|---|---|---|
| 0 | first enabled endpoint in `endpoints.yaml` | alphabetical in practice since the YAML lists them that way |
| 1 | second enabled endpoint | ... |
| ... | ... | ... |
| 21 | last enabled endpoint | ... |

This is **a different ordering from the condition vector** used in C.7 (`sorted(admet.keys())`) — the Oracle training uses `list(preds.values())` which is **dict-insertion order (= YAML order)**, while the generator condition vector uses alphabetical sort. As long as `endpoints.yaml` already lists endpoints in sorted order, they coincide; if the YAML is ever reordered, the Oracle's trained weights would be misaligned with the generator's condition semantics. **⚠️ Gap 13 — feature ordering dependency on YAML order is fragile.**

### 9. Dataset size required for stable training

Rough rules of thumb for the `hidden_dim=256` cascade (~210k params):

| Dataset size | Expected behavior |
|---|---|
| < 500 rows | Overfits within 2–3 epochs. Use dropout > 0.3 or reduce hidden dim. |
| 500–2,000 rows | Marginal; the current `dropout=0.15` + `wd=1e-4` is adequate. |
| 2,000–10,000 rows | Sweet spot for the current architecture. |
| > 10,000 rows | Can safely increase `hidden_dim` to 512, reduce dropout. |

The repo defaults assume the middle range. Without a sizing sanity check on CSV ingestion, users can silently train on 100-row datasets and get garbage predictions. **⚠️ Gap 14 — no minimum-dataset-size warning.**

### 10. How & Why

- **How:** CSV with strict 4-column schema.
  **Why:** minimal friction for users bringing their own labels; no coupling to a specific data source.
- **How:** Features are computed on-the-fly from SMILES via the frozen ADMET model.
  **Why:** avoids storing 22 floats per compound; automatically benefits from any ADMET model improvements.
- **How:** No split by time, indication, or drug class.
  **Why:** simplicity. **⚠️ Known limitation** — models trained this way can't generalize across therapeutic areas well.
- **How:** No handling of censored (in-progress) trials.
  **Why:** labels treated as frozen binary. Real clinical-trial modeling would need survival analysis.

### 11. Alternatives the next iteration might consider

- **ChEMBL bootstrap script** at `scripts/download_clinical_trials.py` that queries ChEMBL for `max_phase ≥ 1`, derives binary phase labels, writes `clinical_trials.csv` → closes **Gap 10**.
- **`pos_weight` on BCE in `OracleTrainer`** → closes **Gap 11**.
- **`__init__`-time row filter** in `OracleDataset` that drops invalid SMILES and NaN labels → closes **Gap 12**.
- **Indication/target columns** in the schema + per-indication sub-models → addresses §7 gaps.
- **Censoring-aware Cox model or discrete-time hazard network** → proper handling of ongoing trials.
- **Pre-computation of ADMET features** with an on-disk cache → avoids redundant forward passes through the frozen ADMET model during training epochs.

### 12. Novelty ledger

| Component | Status | Source |
|---|---|---|
| 4-column CSV schema | Project-specific (very minimal) | no precedent |
| Binary per-phase labels | Reused (common in ML-for-drug-development literature) | Lo A. W. et al., *Biostatistics* 20: 273–286, 2019 (uses this label style for trial-level predictions) |
| `max_phase`-derived labels from ChEMBL | Reused | Mendez D. et al., *Nucleic Acids Research* 47: D930–D940, 2019 (ChEMBL database paper) |
| ADMET-first → phase classifier | Reused | Gayvert K. M., Madhukar N. S., Elemento O., *Cell Chem. Biol.* 23: 1294–1301, 2016 — trial-success prediction from ADMET features |

### 13. Verified?

| Claim | Source tier | Location |
|---|---|---|
| Schema: `{smiles, phase1, phase2, phase3}` required | `repo` | `models/oracle/clinical_data.py` lines 11–14 |
| FileNotFound raised if CSV absent | `repo` | `scripts/train_oracle.py` lines 70–73 |
| Per-row ADMET forward pass at `__getitem__` time | `repo` | `scripts/train_oracle.py` lines 26–37 |
| Feature vector built via `list(preds.values())` | `repo` | `scripts/train_oracle.py` line 31 |
| No NaN/empty-dict handling in the Oracle dataset | `repo` | full-file read of `scripts/train_oracle.py` |
| No download script ships with the repo | `repo` | `Glob: scripts/download*.py` shows only `download_data.py` (ADMET) |
| No `data/processed/oracle/*` in the tree at repo inspection time | `repo` | `Glob: data/processed/oracle/*` returned 0 files |
| ChEMBL `max_phase` as data-sourcing strategy | `web` | Mendez D. et al., *Nucleic Acids Res.* 47: D930–D940, 2019, DOI 10.1093/nar/gky1075 |
| ADMET-first trial success prediction | `web` | Gayvert K. M. et al., *Cell Chem. Biol.* 23: 1294–1301, 2016, DOI 10.1016/j.chembiol.2016.07.023 |
| Wong et al. attrition reference | `web` | Wong C. H., Siah K. W., Lo A. W., *Biostatistics* 20: 273–286, 2019, DOI 10.1093/biostatistics/kxx069 |

---

✅ Step D.1 complete. Running gap ledger:
- Gap 1 (B.4): no training-corpus novelty filter
- Gap 2 (C.1): missing `scripts/featurize_admet.py`
- Gap 3 (C.2): `edge_attr` ignored by GIN
- Gap 4 (C.5): no `pos_weight` on BCE
- Gap 5 (C.6): no scale-normalization in score aggregation; no `last_model.pt` snapshot
- Gap 6 (C.7): no per-dim normalization of ADMET before condition injection
- Gap 7 (C.7): no batched `predict_smiles`
- Gap 8 (D orient.): no val split or early stopping in Oracle training
- Gap 9 (D orient.): Oracle training hyperparameters hard-coded, not config-driven
- **Gap 10 (D.1):** no clinical-trial dataset download script
- **Gap 11 (D.1):** no class weighting on Oracle BCE (severe given label imbalance)
- **Gap 12 (D.1):** no invalid-SMILES / NaN-label filtering in `OracleDataset`
- **Gap 13 (D.1):** Oracle feature ordering depends on YAML endpoint order (fragile)
- **Gap 14 (D.1):** no minimum-dataset-size sanity warning

Say **"next"** for Step D.2 (`CascadedPhasePredictors` architecture — per-phase MLP structure, concatenation mechanics, the detach-or-not decision, param count breakdown), or ask questions to stay on D.1.

---

## Step D.2 — the cascaded phase predictors

### 1. The question

`CascadedPhasePredictors` is ~40 lines of code but embodies three non-trivial architectural choices:

1. What each per-phase MLP looks like internally.
2. How sigmoid(p₁) and sigmoid(p₂) are threaded into later phases' inputs.
3. Why gradients flow *through* the cascade instead of being detached.
4. How the param count breaks down.

### 2. The per-phase building block — `PhasePredictor`

```9:23:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/phase_predictors.py
class PhasePredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

A 3-layer MLP: `Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(·, 1)`.

Choices explained:

| Element | Choice | Rationale |
|---|---|---|
| Depth | 3 Linear layers | enough capacity for non-linear ADMET → phase mappings; not so deep that ~1k rows overfit |
| Hidden dim | 128 (default), **256 in cascade** | cascade overrides to 256 for more capacity |
| Activation | ReLU | standard; no evidence a fancier activation (GELU, SiLU) helps at this scale |
| Dropout | 0.2 (default), **0.15 in cascade** | regularization — critical for small datasets; 0.2 is aggressive, 0.15 is moderate |
| No normalization | no LayerNorm / BatchNorm | input is already in [0, 1]-ish range (ADMET probabilities + sigmoided phase probs); BN on tiny batches is unstable |
| Squeeze at output | `.squeeze(-1)` | makes output shape `(B,)` instead of `(B, 1)` — matches label tensor shape for BCE |

Output is a **logit**, not a probability. The sigmoid is applied later (at inference or when plugging into the next phase). This matches the pattern established in Block C (C.5): train with logits + `BCEWithLogitsLoss`, sigmoid at serve time.

### 3. The cascade wrapper

```26:41:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/phase_predictors.py
class CascadedPhasePredictors(nn.Module):
    """Cascaded predictors: Phase I -> Phase II -> Phase III."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.phase1 = PhasePredictor(in_dim, hidden_dim, dropout)
        self.phase2 = PhasePredictor(in_dim + 1, hidden_dim, dropout)
        self.phase3 = PhasePredictor(in_dim + 2, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1 = self.phase1(x)
        p1_sig = torch.sigmoid(p1).unsqueeze(-1)
        p2 = self.phase2(torch.cat([x, p1_sig], dim=-1))
        p2_sig = torch.sigmoid(p2).unsqueeze(-1)
        p3 = self.phase3(torch.cat([x, p1_sig, p2_sig], dim=-1))
        return p1, p2, p3
```

The six-line forward pass is the architectural heart of Block D.

### 4. Line-by-line dissection of the forward pass

Suppose `x = (B, 22)` where B is batch size.

**Line 36: `p1 = self.phase1(x)`**

- Input: `(B, 22)` — the 22 ADMET features.
- Output: `p1` shape `(B,)` — logit for Phase I success.
- Params touched: `phase1` MLP only.

**Line 37: `p1_sig = torch.sigmoid(p1).unsqueeze(-1)`**

- Apply sigmoid → probabilities in (0, 1).
- `.unsqueeze(-1)` reshapes `(B,)` → `(B, 1)` so it can be concatenated along the feature axis.
- **Critically: sigmoid is differentiable.** The computation graph connects `p1_sig` back to `phase1`'s weights. When `loss.backward()` is called later on `phase2`'s loss, gradients flow through `p1_sig` into `phase1`.

**Line 38: `p2 = self.phase2(torch.cat([x, p1_sig], dim=-1))`**

- Concatenation: `(B, 22) ++ (B, 1) = (B, 23)`.
- Input to `phase2`: the original ADMET features **plus** one extra feature — Phase I's predicted probability.
- Output: `p2` shape `(B,)`.
- Params touched during this line's forward: `phase2` MLP. During backward, gradients also reach `phase1` through `p1_sig`.

**Line 39: `p2_sig = torch.sigmoid(p2).unsqueeze(-1)`** — same pattern as line 37.

**Line 40: `p3 = self.phase3(torch.cat([x, p1_sig, p2_sig], dim=-1))`**

- Concatenation: `(B, 22) ++ (B, 1) ++ (B, 1) = (B, 24)`.
- Input to `phase3`: ADMET + Phase I prob + Phase II prob.
- Output: `p3` shape `(B,)`.

**Line 41: Return the three raw logits**, not probabilities. Why logits? Because `BCEWithLogitsLoss` (used in training — D.3) expects logits.

### 5. Gradient flow — the crucial `detach` question

This is the design decision worth scrutinizing carefully.

**What actually happens without `.detach()`:**

When `L = BCE(p1, y1) + BCE(p2, y2) + BCE(p3, y3)` is backpropped, PyTorch traces the full computation graph:

- `BCE(p3, y3)` produces gradients that flow back through `phase3` → `p2_sig` → `phase2` → `p1_sig` → `phase1`.
- `BCE(p2, y2)` produces gradients that flow through `phase2` → `p1_sig` → `phase1`.
- `BCE(p1, y1)` produces gradients that flow through `phase1` directly.

**So `phase1` receives gradients from all three losses.** `phase2` receives gradients from `L2 + L3`. `phase3` receives gradients only from `L3`.

**Consequence:** `phase1` is trained to produce logits that (a) match `y1`, (b) lead to good predictions when fed as `p1_sig` into `phase2`, and (c) similarly for `phase3`. It is not just a Phase I predictor — it is also "whatever scalar best helps Phase II and Phase III predictors," which may or may not equal the "actual Phase I clearance probability."

**What `.detach()` would do:**

If line 37 were `p1_sig = torch.sigmoid(p1).detach().unsqueeze(-1)`, the computation graph through `p1_sig` would be broken. `phase1` would only receive gradients from `L1`. `phase2` would treat `p1_sig` as a fixed constant during its own training, matching the "staged independent training" intuition.

**The repo chose NOT to detach.** Three implications:

1. **Pro:** the cascade is trained end-to-end; `phase1` implicitly learns useful structure for later phases.
2. **Con:** `p1_sig` may drift from meaning "P(Phase I clearance)" toward meaning "whatever helps Phase II most." Interpretability of individual phase probabilities degrades.
3. **Con:** the three losses compete. If Phase III is harder to fit, its gradient may push `phase1` in directions that hurt `L1`. Not detaching assumes these objectives are aligned.

In practice, at this small scale (~210k params, ~1k training rows), the non-detached cascade is a reasonable choice — the implicit regularization from sharing gradients is useful. At larger scale, staged training (fit `phase1` to convergence with `.detach()`, then `phase2`, then `phase3`) would likely produce cleaner phase-specific calibration. **⚠️ Minor design note — not a bug, but worth flagging.**

### 6. Parameter count breakdown

Per `PhasePredictor(in_dim, hidden_dim=256)`:

| Layer | Params |
|---|---|
| `Linear(in_dim, 256)` | 256 × in_dim + 256 |
| `Linear(256, 256)` | 256 × 256 + 256 = 65,792 |
| `Linear(256, 1)` | 256 × 1 + 1 = 257 |
| **Total** | 256 × in_dim + 256 + 65,792 + 257 |

Plugging in each phase's `in_dim`:

| Phase | `in_dim` | Params |
|---|---|---|
| phase1 | 22 | 256 × 22 + 256 + 65,792 + 257 = **71,937** |
| phase2 | 23 | 256 × 23 + 256 + 65,792 + 257 = **72,193** |
| phase3 | 24 | 256 × 24 + 256 + 65,792 + 257 = **72,449** |
| **Total (CascadedPhasePredictors)** | — | **216,579** |

About **217k params** — call it ~210k for napkin math. As noted in orientation §7: tiny compared to the generator (~2.5M), comparable to the ADMET model (~104k).

**Per-parameter training data requirement:** rule of thumb for well-regularized MLPs is 10–100 training examples per parameter, so 210k params would ideally want 2M+ rows. We have ~1k. The `dropout=0.15` + `weight_decay=1e-4` are the only things holding overfitting at bay. **⚠️ Known scale mismatch** — the architecture is over-parameterized for the available data.

### 7. What the cascade actually computes — a worked example

Suppose aspirin's 22 ADMET predictions from Block C include (fictitious but plausible):
- `hERG = 0.12`, `AMES = 0.04`, `DILI = 0.11`
- `bioavailability_ma = 0.67`, `caco2_wang = -5.1` (log)
- ... 17 more values

Pass this 22-dim vector through the cascade:

1. **phase1:**
   - `(B=1, 22)` → MLP → `p1 = 1.8` (raw logit).
   - `σ(p1) = 0.86` — model predicts 86% chance of clearing Phase I (plausible — aspirin is low-tox, good PK).
2. **phase2:**
   - Input: 22 ADMET + [0.86] → `(1, 23)`.
   - → MLP → `p2 = 0.7` (raw logit).
   - `σ(p2) = 0.67` — 67% Phase II.
3. **phase3:**
   - Input: 22 ADMET + [0.86, 0.67] → `(1, 24)`.
   - → MLP → `p3 = -0.4` (raw logit).
   - `σ(p3) = 0.40` — 40% Phase III.

Returned: raw logits `(1.8, 0.7, -0.4)`. These feed into:

- Training loss: `BCE(1.8, y1) + BCE(0.7, y2) + BCE(-0.4, y3)`.
- Inference: sigmoid applied externally (§5 of D orientation), yielding `{phase1: 0.86, phase2: 0.67, phase3: 0.40}`, then plugged into `_clinical_quality`.

The test file confirms this pattern — see `tests/test_oracle.py` lines 7–14, where a dummy model returns three logits (0.2, 0.8, −0.4) and the downstream `_clinical_quality` sigmoids them internally.

### 8. Data transformation trace

| Stage | Shape | Operation |
|---|---|---|
| input | `(B, 22)` | from `predict_smiles` batched |
| phase1 logit | `(B,)` | `phase1.forward(x)` |
| phase1 sig | `(B, 1)` | `sigmoid().unsqueeze(-1)` |
| phase2 input | `(B, 23)` | concat with ADMET |
| phase2 logit | `(B,)` | `phase2.forward(·)` |
| phase2 sig | `(B, 1)` | `sigmoid().unsqueeze(-1)` |
| phase3 input | `(B, 24)` | concat with ADMET + phase1_sig |
| phase3 logit | `(B,)` | `phase3.forward(·)` |
| return | `(p1, p2, p3)` each `(B,)` | three raw logits |

### 9. Alternative architectures

| Alternative | Trade-off |
|---|---|
| **Independent MLPs** (no cascade) | simpler, interpretable per-phase calibration; loses the dependency structure |
| **Multi-head from one trunk** (share feature extractor, 3 heads) | more parameter-efficient; doesn't encode sequential dependency |
| **Cascade with `.detach()` between stages** | cleaner per-phase meaning; staged training complexity |
| **Autoregressive LSTM over phases** | generalizes to arbitrary number of phases; overkill for 3 stages |
| **Survival model (Cox, DeepHit)** | handles censoring properly; requires time-to-event labels (not in current schema) |
| **Cascade with residual connection** (p2 input = ADMET + p1_sig, p3 input = ADMET + p1_sig + p2_sig + phase2_hidden) | richer information flow; more params, more overfitting risk |

The chosen design is the simplest architecture that encodes conditional dependency between phases. Given the data scale, it's appropriately sized. A production version would probably move to a survival framework, but that requires a different dataset.

### 10. How & Why

- **How:** Three stacked MLPs with sigmoid feedback between them.
  **Why:** encodes clinical-phase conditionality (Phase 2 depends on Phase 1 outcome).
- **How:** 3-layer MLP per phase, `Linear → ReLU → Dropout` stacks, no normalization.
  **Why:** small + regularized for the small dataset; dropout > 0 is essential; BN unstable at `B=64` on a 1k-row dataset.
- **How:** Output raw logits, not probabilities.
  **Why:** `BCEWithLogitsLoss` for numerical stability (same pattern as Block C).
- **How:** Gradients flow through the sigmoid feedback (no `.detach()`).
  **Why:** end-to-end training; implicit regularization. Trade-off: slightly noisier individual-phase interpretation.
- **How:** Hidden dim 256, dropout 0.15.
  **Why:** cascade's `__init__` defaults; tunable via `CascadedPhasePredictors(hidden_dim=..., dropout=...)` but currently not exposed in config.yaml (Gap 9 from orientation).

### 11. Novelty ledger

| Component | Status | Source |
|---|---|---|
| Stacked MLP block | Reused (standard) | `Linear → ReLU → Dropout` pattern, Srivastava et al. (Dropout) 2014 |
| Sequential conditional predictors | Reused pattern | applied in structured prediction (e.g., Koller & Friedman, *Probabilistic Graphical Models*, 2009) |
| Specific cascade-for-clinical-phases | Project-specific | the application of sequential conditioning to Phase I/II/III is this project's choice |
| `BCEWithLogitsLoss` + sigmoid at serve | Reused | PyTorch idiom |

### 12. Verified?

| Claim | Source tier | Location |
|---|---|---|
| Per-phase MLP: `Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(·,1)` | `repo` | `phase_predictors.py` lines 12–20 |
| Cascade: phase1 in=22, phase2 in=23, phase3 in=24 | `repo` | `phase_predictors.py` lines 31–33 |
| Sigmoid feedback (no `.detach()`) between phases | `repo` | `phase_predictors.py` lines 37–40 |
| Raw logits returned | `repo` | `phase_predictors.py` line 41 (no sigmoid on outputs) |
| Default `hidden_dim=256, dropout=0.15` for the cascade | `repo` | `phase_predictors.py` line 29 |
| `squeeze(-1)` on output for shape `(B,)` | `repo` | `phase_predictors.py` line 23 |
| Test confirms 3-logit output pattern | `repo` | `tests/test_oracle.py` lines 7–14 |

---

✅ Step D.2 complete. Running gap ledger unchanged (14 items).

Say **"next"** for Step D.3 (`OracleTrainer` training loop — Adam + BCE × 3 summed, 10 epochs, why no scheduler, what goes wrong without a val split, gradient-flow implications of the cascade), or ask questions to stay on D.2.

---

## Step D.3 — the Oracle training loop

### 1. The question

Training the cascade is ~40 lines split across `OracleTrainer` (the epoch mechanic) and `train_oracle.py` (the orchestrator). The loop is intentionally the simplest possible, which makes every choice (and every omission) highly visible.

What we'll trace:

1. Optimizer configuration — Adam + what hyperparameters, why.
2. The summed 3-way BCE loss — gradient arithmetic.
3. The epoch loop end-to-end.
4. What's *not* there (val split, scheduler, early stop, grad clip, gradient accumulation) and why each matters.
5. Runtime characteristics — per-epoch cost, bottleneck (ADMET inference, not Oracle forward).

### 2. The `OracleTrainer` class

```11:43:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/trainer.py
class OracleTrainer:
    def __init__(
        self,
        model: CascadedPhasePredictors,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            x = batch["x"].to(self.device)
            y1 = batch["phase1"].to(self.device)
            y2 = batch["phase2"].to(self.device)
            y3 = batch["phase3"].to(self.device)

            self.optimizer.zero_grad()
            p1, p2, p3 = self.model(x)
            loss = self.loss_fn(p1, y1) + self.loss_fn(p2, y2) + self.loss_fn(p3, y3)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            count += 1
        return total_loss / max(count, 1)
```

Everything the Oracle learns is driven by the interaction of these three lines:

- `p1, p2, p3 = self.model(x)` — forward through the cascade.
- `loss = BCE(p1, y1) + BCE(p2, y2) + BCE(p3, y3)` — unweighted sum.
- `loss.backward()` + `optimizer.step()` — single gradient step per batch.

### 3. Optimizer: Adam, lr=1e-3, wd=1e-4

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | adaptive per-param LR; robust to small-data, low-LR-tuning regimes; same choice as Blocks B & C |
| `lr` | 1e-3 | PyTorch default for Adam; appropriate for ~200k-param MLP on normalized inputs |
| `weight_decay` | 1e-4 | mild L2 regularization — critical given the ~200:1 parameter-to-data ratio |
| `betas` | (0.9, 0.999) default | unchanged |
| `eps` | 1e-8 default | unchanged |

**Comparison with other blocks:**

| Block | Optimizer | LR | Schedule | WD |
|---|---|---|---|---|
| B (generator pretrain) | Adam | 3e-4 | cosine annealing | 0 |
| B (RL fine-tune) | Adam | 1e-5 (actor), 1e-4 (critic) | none | 0 |
| C (ADMET) | Adam | 5e-4 | `ReduceLROnPlateau` | 1e-5 |
| **D (Oracle)** | **Adam** | **1e-3** | **none** | **1e-4** |

Block D has the **highest LR and strongest weight decay** — consistent with the smallest dataset and the simplest model (no sequence positional structure, no graph message passing).

**Why no scheduler?**

Schedulers are most useful when: (a) you have enough epochs for decay to matter (50+), (b) you have a validation signal to react to (`ReduceLROnPlateau`), or (c) you want warm restarts to escape local minima. Block D trains for **10 epochs** with no val split — none of these conditions is met. A constant LR is defensible at this scale. **Still flagged as ⚠️ Gap 8 (orientation)** because the *reason* for no scheduler is "we never built a val split," not "a flat LR is optimal."

### 4. The loss — unweighted sum of three BCEs

The loss on line 38 is:

\[
\mathcal{L} = \text{BCE}(p_1, y_1) + \text{BCE}(p_2, y_2) + \text{BCE}(p_3, y_3)
\]

where each `BCE` is `BCEWithLogitsLoss` reduced by **mean** over the batch.

**Gradient flow implications (from D.2):**

- `∂L/∂(phase3 params)` has one contributor: `∂L3/∂(phase3)`.
- `∂L/∂(phase2 params)` has two contributors: `∂L2/∂(phase2)` **and** `∂L3/∂(phase3)/∂(p2_sig)/∂(phase2)`.
- `∂L/∂(phase1 params)` has three contributors: `∂L1/∂(phase1)` plus two paths through the cascade.

**So `phase1` is trained ~3× harder than `phase3` in gradient magnitude.** This is a subtle form of implicit loss weighting:

- Phase I gets the most attention → learns earliest (beneficial; it's the shared foundation).
- Phase III gets the least → learns slowest (but it's also the most informative signal — rare successes).

At small scale this balance often works. At larger scale, one would typically introduce **explicit loss weights** (e.g., `loss = w1*L1 + w2*L2 + w3*L3`) to counter the gradient-accumulation imbalance — but the repo does not expose these weights.

**No `pos_weight` for class imbalance** (Gap 11 from D.1). If failures are under-represented in the training data (they are — publication bias), the unweighted BCE drives all three logits toward "pass" predictions. Remedy: `BCEWithLogitsLoss(pos_weight=neg_count/pos_count)` per phase.

### 5. The epoch loop — mechanics

For each batch:

1. **Device transfer** (lines 31–34): `x`, `y1`, `y2`, `y3` moved to `self.device` (CPU in current config).
2. **Zero grad** (line 36): clears previous batch's gradients.
3. **Forward** (line 37): `p1, p2, p3 = self.model(x)` — the full cascade described in D.2.
4. **Loss** (line 38): three BCEs summed.
5. **Backward** (line 39): autograd traces the summed scalar back through all three phases.
6. **Step** (line 40): Adam applies the update.
7. **Accumulate** (lines 41–42): track running mean loss.

**One gradient step per batch.** No gradient accumulation (not needed — `B=64` is already a full-size batch for this model).

**No gradient clipping.** Block B uses `nn.utils.clip_grad_norm_` because transformer gradients can spike on rare long sequences. Block D has bounded inputs (ADMET in [0,1]) and a small model — gradients are well-behaved. Defensible omission.

### 6. The orchestration script — `train_oracle.py`

```82:88:/Users/sreevardhandesu/Desktop/prj_demo/scripts/train_oracle.py
for epoch in range(1, 11):
    loss = trainer.train_epoch(loader)
    print(f"Epoch {epoch} | Loss: {loss:.4f}")

ckpt_dir = project_root / "checkpoints" / "oracle"
ckpt_dir.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict()}, ckpt_dir / "best_model.pt")
```

Seven lines that encode the entire "training lifecycle":

- **`range(1, 11)`:** exactly **10 epochs**, hard-coded (Gap 9 from orientation — not config-driven).
- **No validation call:** `trainer.evaluate(val_loader)` does not exist for the Oracle.
- **No scheduler step:** no `scheduler.step(val_metric)`.
- **No best-tracking:** saved checkpoint is the *last* epoch, misleadingly named `best_model.pt`.
- **No `last_model.pt`:** same file serves both purposes (no crash recovery distinction).

**The `best_model.pt` naming is a minor but real bug.** Elsewhere in the codebase (Block C), `best_model.pt` specifically means "model with best val score," and `load_model` utilities are written to trust that semantic. Block D breaks that contract — it saves *last* epoch under the same name. Consumers are safe only because Oracle doesn't track val metrics at all. **⚠️ Naming inconsistency — small but worth a note.**

### 7. Why 10 epochs?

At `B=64` on a hypothetical 1k-row dataset, one epoch = ~16 gradient steps. 10 epochs = ~160 steps. That's enough for a 200k-param MLP to:

- With a 1k dataset: start overfitting around epoch 5–7 (no val split means we won't see it).
- With a 5k dataset: converge around epoch 10–15 — plausibly where the "10" was tuned.
- With a 10k+ dataset: undertrain — needs 30–50 epochs.

The hard-coded 10 is a **data-size-dependent magic number** with no validation signal to confirm it's right. This is the primary risk of the whole training setup: we can't tell whether the Oracle is undertrained, converged, or overfit. The loss curve printed to stdout is the only signal, and training loss alone cannot distinguish these three states.

### 8. Runtime characteristics

The dominant cost per epoch is **ADMET inference**, not the Oracle forward pass.

Walk through one sample's data-loading path in `OracleDataset.__getitem__` (D.1):

1. Parse SMILES → RDKit mol → featurize → PyG `Data` object.
2. Run the **full ADMET GNN** (3-layer GIN + 22 heads, ~104k params) to produce 22 predictions.
3. Stack into a 22-dim tensor.

Step 2 dominates. For CPU, an ADMET forward pass is ~10–30 ms per molecule. On a 1k-row dataset:

- Per epoch: 1000 × ~20 ms = ~20 s for ADMET inference.
- Per epoch: ~16 batches × ~2 ms for Oracle forward/backward = ~30 ms.
- **ADMET inference is ~600× more expensive than Oracle training.**

**⚠️ Inefficiency:** `OracleDataset.__getitem__` runs ADMET inference **every epoch, every sample**. ADMET predictions for the same SMILES are deterministic at inference time (model in eval mode, no dropout). They could be cached once and reused. With 10 epochs, caching would be 10× faster. Not a correctness bug — a severe training-speed gap. **Adding to gap ledger as Gap 15.**

Batched ADMET inference is already unavailable (Gap 7 from C.7), so this loop also suffers from per-molecule overhead (no CUDA graph reuse, no vectorized RDKit path).

### 9. What the loss curve looks like (expected behavior)

Even without a val split, the training loss curve reveals a few things:

| Epoch | Loss | What it means |
|---|---|---|
| 1 | ~2.0 (`3 × 0.69` for random init on balanced data) | baseline — BCE at 50% predictions = ln(2) per head × 3 |
| 2–3 | 1.5–1.8 | phase1 starts fitting (easiest — no cascade input needed) |
| 4–6 | 1.2–1.5 | phase2 fits; phase1 stable |
| 7–10 | 0.8–1.2 | phase3 fits; possibly starting to overfit |

If you see loss spike or plateau at ~2.0 after epoch 1: likely a label-encoding bug (labels not in {0, 1}) or NaN in `x` (Gap 12). If loss drops to ~0.1 within 2 epochs: likely overfitting — dataset too small or model too large.

### 10. Data transformation trace (training batch)

| Stage | Shape | Notes |
|---|---|---|
| Raw batch from `DataLoader` | dict of `(B=64, ·)` tensors | `x:(64,22)`, `phase1/2/3:(64,)` |
| After device move | same | CPU → CPU (currently); or CPU → GPU if `device=cuda` |
| After `model(x)` | 3 × `(B,)` logits | raw, unbounded |
| BCE per head | scalar | reduced by mean over batch |
| Summed loss | scalar | `L = L1 + L2 + L3` |
| `loss.backward()` | — | gradients populated on all cascade params |
| `optimizer.step()` | — | one Adam update |

### 11. Alternatives to the current training loop

| Alternative | What changes | Trade-off |
|---|---|---|
| **Add val split + early stopping** | split df 90/10, check val loss every epoch, stop if no improvement for N | essential for correctness at small scale; ~10 lines |
| **Cache ADMET predictions** | run ADMET once, store 22-dim vectors in-memory or on disk | 10× speedup with no behavior change (Gap 15) |
| **Staged (phase-by-phase) training** | train `phase1` to convergence with `.detach()` feedback, then `phase2`, then `phase3` | cleaner per-phase calibration; complex orchestration |
| **Weighted BCE** | `BCEWithLogitsLoss(pos_weight=·)` per phase | addresses class imbalance; needs pos/neg count from df |
| **`ReduceLROnPlateau`** | scheduler on val loss | only useful once val split exists |
| **Mixed-precision** | `torch.cuda.amp` | marginal for this model size; irrelevant on CPU |
| **Config-driven hyperparameters** | move `lr`, `wd`, `epochs`, `batch_size` to `config.yaml` | consistency with Blocks B/C (Gap 9) |

### 12. How & Why

- **How:** Adam + unweighted summed BCE × 3 over 10 epochs, batch 64, single gradient step per batch.
  **Why:** matches the model's small size and the dataset's small size. Minimum viable loop.
- **How:** No scheduler, no grad clip, no val split.
  **Why (good):** model is bounded and shallow — no need for clipping. Why (bad): no val means no overfitting detection.
- **How:** Save `best_model.pt` only at the end (last epoch, not best).
  **Why:** consistent naming with Block C's checkpoint path, but with broken semantic — it's the *last*, not the best.
- **How:** ADMET inference per-sample per-epoch.
  **Why:** simpler code. Why (bad): 10× redundant work, since ADMET predictions are deterministic at eval.

### 13. Novelty ledger

| Component | Status | Source |
|---|---|---|
| Adam optimizer | Reused | Kingma & Ba 2015 |
| `BCEWithLogitsLoss` | Reused | PyTorch standard |
| Summed multi-task loss | Reused pattern | Kendall et al. (*Multi-Task Learning Using Uncertainty*) 2018, though without uncertainty weighting here |
| 10-epoch training loop | Project-specific | chosen magic number |
| Per-sample on-the-fly ADMET inference | Project-specific | expedient but inefficient |

### 14. Verified?

| Claim | Source tier | Location |
|---|---|---|
| `OracleTrainer` uses Adam, lr=1e-3, wd=1e-4 | `repo` | `trainer.py` lines 16–22 |
| Single `nn.BCEWithLogitsLoss` applied to each phase logit | `repo` | `trainer.py` lines 24, 38 |
| Loss is unweighted sum of three BCEs | `repo` | `trainer.py` line 38 |
| No gradient clipping | `repo` | absent in `train_epoch` |
| 10 epochs, hard-coded | `repo` | `train_oracle.py` line 82 |
| Batch size 64, hard-coded | `repo` | `train_oracle.py` line 77 |
| Checkpoint saved as `best_model.pt` despite being last epoch | `repo` | `train_oracle.py` line 88 |
| No validation loop, no scheduler, no early stop | `repo` | absent in `train_oracle.py` |
| `OracleDataset.__getitem__` runs ADMET inference per sample | `repo` | `train_oracle.py` lines 26–31 |

### 15. Gap ledger update

New gap added in this step:

- **⚠️ Gap 15 (D.3):** ADMET predictions are recomputed every epoch for every sample inside `OracleDataset.__getitem__`. Since ADMET inference is deterministic at eval time, this causes ~10× redundant compute over a 10-epoch run. A one-shot cache (in-memory dict or precomputed `.pt` tensor) would make training ~10× faster with zero behavior change.

Running total: **15 gaps** tracked across Blocks B/C/D.

---

✅ Step D.3 complete.

Say **"next"** for Step D.4 (structural alert matcher — SMARTS-based substructure filters, how penalties feed into the composite score, the three-tier severity ranking), or ask questions to stay on D.3.

---

## Step D.4 — the structural alert matcher

### 1. The question

Before Block D even runs its neural predictors, every candidate passes through a **hard-coded substructure filter**. This is the fastest, most interpretable signal in the entire pipeline — no training, no probabilities, just yes/no pattern matches.

We'll trace:

1. What a "structural alert" is (cheminformatics background).
2. The SMARTS pattern language — how it differs from SMILES.
3. The 5 patterns currently shipped in `data/structural_alerts.csv`.
4. The matching algorithm (`detect_structural_alerts`).
5. How alert hits feed into `_clinical_quality` as penalties.
6. The atom-level `alert_atoms` output (for UI highlighting).
7. What's missing vs. production alert libraries (PAINS, BRENK, Glaxo, SureChEMBL).

### 2. Background — what are structural alerts?

A **structural alert** is a substructure pattern that, when present in a molecule, has been historically associated with toxicity, reactivity, or other drug-development failures. The underlying idea:

> *"If this tiny chemical motif shows up, the molecule probably has problem X."*

Alerts are **rule-based, not probabilistic**. They encode decades of medicinal chemistry knowledge (published Ames test results, DILI databases, FDA withdrawals) as pattern matches. They trade recall for precision: many toxic molecules won't trigger any alert (false negatives), but molecules that *do* trigger alerts are high-risk with high probability (high precision).

**Why include them at all when we have a trained ADMET model?** Three reasons:

1. **Complementary signal.** ADMET predictions generalize from training data; alerts encode *rules* that apply even to out-of-distribution molecules.
2. **Interpretability.** "Your molecule has an aromatic nitro group → known mutagen" is actionable; "Your hERG probability is 0.73" is opaque.
3. **Latency.** Substructure matching is ~1 ms; ADMET inference is ~20 ms. Alerts filter cheaply before expensive models run.

Published alert collections include **PAINS** (Pan-Assay Interference Compounds, Baell & Holloway 2010), **BRENK** (Brenk et al. 2008, 105 alerts), **Glaxo** (Hann et al. 1999, 55 alerts), and **NIH MLSMR** filters. This project ships **5 alerts** — a deliberately tiny, illustrative subset.

### 3. SMARTS — the pattern matching language

**SMILES** represents *one specific molecule*: `c1ccccc1` = benzene. **SMARTS** extends SMILES with *pattern-matching operators*:

| SMARTS construct | Meaning |
|---|---|
| `c1ccccc1` | aromatic benzene (plain SMILES, works as a SMARTS too) |
| `[#6]` | any carbon (atomic number 6) |
| `[NH2,NH1,NH0]` | nitrogen with 2, 1, or 0 H's (list match) |
| `[N+](=O)[O-]` | nitro group (formal charges explicit) |
| `!$(...)` | *not* matching the subpattern |
| `;` | and (within atom brackets) |
| `,` | or (within atom brackets) |
| `$(...)` | recursive SMARTS — matches if the pattern inside is present |

SMARTS is implemented in RDKit via `Chem.MolFromSmarts()`. Once compiled into a `RWMol` pattern, matching is done with `mol.GetSubstructMatches(pattern)` — returns a list of tuples, each tuple an atom-index mapping of a match.

**Why SMARTS and not regex on SMILES?** SMILES is not a regular language — `c1ccccc1` and `c1ccc(cc1)` and `C1=CC=CC=C1` all describe benzene, but they're different strings. Regex would miss most. SMARTS operates on the *molecular graph* (after SMILES parsing), not the string, so it matches regardless of canonicalization.

### 4. The shipped alert set

The CSV at `data/structural_alerts.csv` has exactly **5 alerts** (6 lines including header):

```1:6:/Users/sreevardhandesu/Desktop/prj_demo/data/structural_alerts.csv
id,name,smarts,category,severity,recommendation
nitro_aromatic,Aromatic Nitro,"[$(c1ccccc1[N+](=O)[O-]),$(c1ccncc1[N+](=O)[O-]),$(c1cnccc1[N+](=O)[O-])]",mutagenicity,high,Replace -NO2 with -CN or -CF3
aromatic_amine,Aromatic Amine (Aniline),"[NH2,NH1,NH0;!$(N-C=O)]c1ccccc1",mutagenicity,high,Convert to amide or replace with -OH/-OCH3
nitroso,Nitroso Group,[#6]N=O,mutagenicity,critical,Remove nitroso group
azo,Azo Compound,[#6]N=N[#6],mutagenicity,medium,Replace azo linkage with amide/ether
epoxide,Epoxide,C1OC1,reactivity,medium,Avoid epoxide ring
```

Breakdown:

| ID | SMARTS | Category | Severity | Chemical rationale |
|---|---|---|---|---|
| `nitro_aromatic` | nitro attached to benzene/pyridine variants | mutagenicity | high | aromatic nitro → nitroreductase → reactive nitrenium ion → DNA adducts (Ames-positive) |
| `aromatic_amine` | aniline-like (aryl-NH₂, not amide) | mutagenicity | high | cytochrome P450 N-oxidation → hydroxylamine → DNA-reactive nitrenium |
| `nitroso` | C–N=O | mutagenicity | critical | direct DNA alkylator; NDMA-class carcinogen |
| `azo` | C–N=N–C | mutagenicity | medium | metabolic reduction → primary arylamines |
| `epoxide` | three-membered C-O-C ring | reactivity | medium | strained ring → direct electrophile → protein/DNA adducts |

Note **4 of 5 are mutagenicity alerts**, 1 is general reactivity. This matches published literature: Ames-positive substructures are the best-studied class.

**SMARTS subtleties:**

- `nitro_aromatic` uses **recursive SMARTS** (`$(...)`) to require the nitro be attached to *specific* aromatic rings (benzene, pyridine at two positions). Without this, `[c][N+](=O)[O-]` would match the group on any aromatic atom, missing some edge cases.
- `aromatic_amine` uses `!$(N-C=O)` to **exclude amides** — `R-C(=O)-NH-Ar` is not a genuine aniline and has very different reactivity.
- `epoxide` uses **aliphatic C** (`C`, capital) and aliphatic O — specifically targets the 3-ring; doesn't match furans.

### 5. Aspirin worked example

SMILES: `CC(=O)Oc1ccccc1C(=O)O` (acetylsalicylic acid).

Walk through each alert:

| Alert | Match? | Why |
|---|---|---|
| `nitro_aromatic` | No | no `[N+](=O)[O-]` group |
| `aromatic_amine` | No | no aniline N atom |
| `nitroso` | No | no N=O |
| `azo` | No | no N=N |
| `epoxide` | No | no 3-membered C-O-C ring |

**Aspirin clears all 5 alerts** → zero penalty contribution from structural alerts. This is consistent with aspirin being a >120-year-old drug with no flagged-substructure-based concerns.

### 6. The matching algorithm

```81:98:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/structural_alerts.py
def detect_structural_alerts(smiles: str) -> Tuple[List[str], np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], None
    hits = []
    alert_atoms = np.zeros(mol.GetNumAtoms(), dtype=int)
    for key, alert in STRUCTURAL_ALERTS_DB.items():
        pattern = alert.pattern()
        if pattern is None:
            logger.warning(f"Invalid SMARTS for alert: {key}")
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            hits.append(alert.name)
            for match in matches:
                for idx in match:
                    alert_atoms[idx] = 1
    return hits, alert_atoms
```

Walkthrough:

1. **Line 82:** parse SMILES. Invalid SMILES → return empty hits + `None` (the `None` is a fragile return type — see gap).
2. **Line 86:** initialize a 0/1 mask of size `mol.GetNumAtoms()` — one bit per atom.
3. **Lines 87–97:** iterate over every alert in the DB:
   - `alert.pattern()` compiles SMARTS (via `Chem.MolFromSmarts`) on each call — redundant work (see gap).
   - `mol.GetSubstructMatches(pattern)` returns a tuple of tuples: each inner tuple is the atom indices of one match. Multiple matches possible.
   - If any match: record the alert *name* (not id) in `hits`, and set `alert_atoms[idx] = 1` for every atom touched by any match.
4. **Return:** `(hits: List[str], alert_atoms: np.ndarray)`.

**Design observations:**

- **Linear scan over alerts** — O(|alerts| × |atoms|²) in worst case, but each SMARTS is small, and |alerts| = 5, so this is trivially fast (<1 ms).
- **Pattern compilation in `alert.pattern()`** — `Chem.MolFromSmarts(self.smarts)` runs every time `detect_structural_alerts` is called. Small set → imperceptible. Would matter if DB grew to 100+ alerts. **⚠️ Minor gap: compile once, cache.**
- **Hits record `alert.name`, not `alert.id`** — display-friendly but not machine-friendly (the `recommendations` module later joins on `name`, which works only because names are unique).
- **Severity field is read but unused in penalty math.** Critical alerts and medium alerts both contribute the same 0.08 penalty to `_clinical_quality`. **⚠️ Gap 16 — severity-weighted penalty is a natural extension that's not implemented.**

### 7. How alerts flow into `_clinical_quality`

```92:100:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
def _clinical_quality(self, p1: float, p2: float, p3: float, admet: Dict[str, float], alerts: List[str]) -> float:
    base = PHASE_WEIGHT_1 * p1 + PHASE_WEIGHT_2 * p2 + PHASE_WEIGHT_3 * p3
    penalty = 0.0
    for key in ("herg", "ames", "dili"):
        if admet.get(key, 0) > 0.5:
            penalty += RISK_PENALTY_PER_ENDPOINT
    penalty += len(alerts) * STRUCTURAL_ALERT_PENALTY
    penalty = min(penalty, MAX_RISK_PENALTY)
    return max(0.0, min(1.0, base - penalty))
```

Two penalty sources are combined:

**ADMET penalties** (lines 95–97):

- For each of `hERG`, `AMES`, `DILI`: if predicted probability > 0.5, add `0.12`.
- Max from this source: `3 × 0.12 = 0.36`.

**Structural alert penalties** (line 98):

- `len(alerts) × 0.08` — each triggered alert adds 0.08 regardless of severity.
- With 5 alerts total, max from this source: `5 × 0.08 = 0.40`.

**Hard cap** (line 99):

- `penalty = min(penalty, 0.5)` — total penalty can never exceed 0.5.
- Prevents a "toxic all the way" molecule from having its phase-weighted base completely zeroed.

**Final clipping** (line 100):

- `max(0.0, min(1.0, base - penalty))` — output in [0, 1].

The key numeric intuition:

> A molecule with median phase probs (p1=p2=p3=0.5) has `base = 0.2·0.5 + 0.5·0.5 + 0.3·0.5 = 0.5`. Two structural alerts drop it to `0.5 - 0.16 = 0.34`. A single hERG hit + nitroso alert drops it to `0.5 - 0.12 - 0.08 = 0.30`.

The penalty **dominates** the base score for borderline molecules — alerts carry a lot of weight. For strong-prediction molecules (p1=p2=p3=0.9), `base = 0.9` and two alerts barely matter: `0.9 - 0.16 = 0.74`.

### 8. The `alert_atoms` output

The second return value of `detect_structural_alerts` is a `numpy.ndarray` of shape `(num_atoms,)` with 0/1 values. Every atom index participating in *any* alert match gets a 1.

**Who uses it?** Currently, only `OraclePrediction.alert_atoms` stores it. There's no UI in this clean rebuild (UI was explicitly scoped out), so the `alert_atoms` mask is prepared for a consumer that doesn't exist. Kept for compatibility with the original project's highlighting feature.

**⚠️ Dead-but-harmless field.** Could be removed in a minimalist pass, but it's essentially free to compute and keeps the API stable if a viewer is ever added.

### 9. DB loading — CSV-first, builtin fallback

```73:78:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/structural_alerts.py
def _get_structural_alerts_db() -> Dict[str, StructuralAlert]:
    loaded = load_structural_alerts_from_csv(_DEFAULT_ALERTS_PATH)
    return loaded if loaded else _BUILTIN_STRUCTURAL_ALERTS.copy()


STRUCTURAL_ALERTS_DB: Dict[str, StructuralAlert] = _get_structural_alerts_db()
```

Loading policy:

1. Try `data/structural_alerts.csv` — if present and non-empty, use it.
2. Else fall back to `_BUILTIN_STRUCTURAL_ALERTS` (2 entries: nitro_aromatic + aromatic_amine only).
3. **DB is built once at module import** (`STRUCTURAL_ALERTS_DB = _get_structural_alerts_db()`). Changes to the CSV require a Python restart.

The CSV loader has robust failure handling:

- **Missing file:** returns empty dict → fall back to builtins.
- **Invalid SMARTS:** `Chem.MolFromSmarts(smarts) is None` → skip with warning, don't crash.
- **Missing required fields (id/name/smarts):** skip silently.

### 10. Data transformation trace (end to end)

| Stage | Input | Output |
|---|---|---|
| module import | `data/structural_alerts.csv` | `STRUCTURAL_ALERTS_DB: Dict[str, StructuralAlert]` with 5 entries |
| per-prediction call | SMILES string | mol parsed via `Chem.MolFromSmiles` |
| pattern compilation | alert.smarts | `RWMol` SMARTS pattern (5× per call — see gap) |
| substructure match | `(mol, pattern)` | `tuple` of atom-index tuples (one per match) |
| aggregation | all alert hits | `hits: List[str]` (name-strings) + `alert_atoms: np.ndarray` |
| into `_clinical_quality` | `len(hits)` | scalar penalty = `len(hits) × 0.08` |

### 11. Alternatives

| Alternative | Trade-off |
|---|---|
| Ship full PAINS (480 patterns) | much stricter filter; ~20% of ChEMBL flagged as hits; many "false alarms" for experienced chemists |
| Ship BRENK (105 patterns) | cleaner than PAINS, broader than the current 5 |
| Severity-weighted penalty | `penalty_per_alert = {critical: 0.15, high: 0.10, medium: 0.05}` — tune to align with medicinal-chemist intuition |
| Deep learning alerts | learn substructures from data (e.g., DeepTox); loses interpretability |
| RDKit's built-in `FilterCatalog` | drop-in access to PAINS/BRENK/Glaxo filters via `rdkit.Chem.rdMolDescriptors` |

The chosen "5 hand-picked SMARTS + CSV override" is the simplest viable filter. A production version would swap to `FilterCatalog` with PAINS + BRENK enabled. **⚠️ Gap 17 — not a bug, but the filter is genuinely minimal for a clinical-trial oracle.**

### 12. How & Why

- **How:** SMARTS pattern matching via RDKit's `GetSubstructMatches`.
  **Why:** industry standard; battle-tested; graph-based matching handles SMILES variants.
- **How:** 5 alerts in CSV, loaded at import, falling back to 2 builtins.
  **Why:** deliberately minimal; demonstrates the plug-in path (CSV → DB) without committing to a specific alert library. CSV is easy to extend.
- **How:** Each alert adds fixed `0.08` penalty regardless of severity.
  **Why:** simplicity. Severity tiers are captured in the data but not used in math — this is a known simplification.
- **How:** Atom-level mask (`alert_atoms`) alongside hit names.
  **Why:** prepares the API for a highlighting viewer (not built in this clean rebuild).
- **How:** Module-level DB (`STRUCTURAL_ALERTS_DB`), loaded once.
  **Why:** avoid per-call CSV parsing cost; downside is you must restart Python after CSV edits.

### 13. Novelty ledger

| Component | Status | Source |
|---|---|---|
| Structural alert concept | Reused (decades old) | Ashby & Tennant 1988; Kazius et al. 2005 |
| SMARTS pattern language | Reused | Daylight SMARTS spec; RDKit implementation |
| Specific 5-alert set | Reused from literature | nitro, aniline, nitroso, azo, epoxide — all classical mutagenicity alerts |
| CSV-based override mechanism | Project-specific | simple extensibility pattern |
| Fixed 0.08 penalty / alert | Project-specific | chosen to balance against 0.12 per ADMET toxicity hit |

### 14. Verified?

| Claim | Source tier | Location |
|---|---|---|
| 5 structural alerts in `data/structural_alerts.csv` | `repo` | `structural_alerts.csv` (6 lines total including header) |
| SMARTS matched via RDKit `Chem.MolFromSmarts` + `mol.GetSubstructMatches` | `repo` | `structural_alerts.py` lines 24, 88, 92 |
| Built-in fallback has 2 alerts (nitro_aromatic + aromatic_amine) | `repo` | `structural_alerts.py` lines 27–42 |
| CSV-first, builtin fallback policy | `repo` | `structural_alerts.py` lines 73–75 |
| DB loaded once at module import | `repo` | `structural_alerts.py` line 78 |
| Invalid SMARTS / missing fields skipped with warnings | `repo` | `structural_alerts.py` lines 58–60 |
| `alert_atoms` mask per atom, 0/1 | `repo` | `structural_alerts.py` lines 86, 95–97 |
| `_clinical_quality` penalty: `len(alerts) × 0.08`, capped at 0.5 | `repo` | `drug_oracle.py` lines 19–20, 98–99 |
| Severity field exists in schema but not used in penalty math | `repo` | `drug_oracle.py` line 98 (no severity lookup) |
| Aspirin triggers 0 of 5 alerts | `analysis` | manual SMARTS inspection of each pattern vs. `CC(=O)Oc1ccccc1C(=O)O` |

### 15. Gap ledger update

New gaps added in this step:

- **⚠️ Gap 16 (D.4):** Severity field in `StructuralAlert` is loaded and stored but not used in penalty math. All alerts contribute the same 0.08 regardless of `critical`/`high`/`medium`. A severity-weighted penalty (`{critical: 0.15, high: 0.10, medium: 0.05}`) is a natural, no-retraining improvement.
- **⚠️ Gap 17 (D.4):** The shipped alert set is 5 patterns — far below production-grade filters (PAINS: 480, BRENK: 105). The architecture supports extension (CSV override works), but out-of-the-box coverage is minimal. Recommendation: integrate `rdkit.Chem.FilterCatalog` for PAINS + BRENK as an opt-in flag.
- **⚠️ Gap 18 (D.4):** `alert.pattern()` recompiles SMARTS on every call (inside the hot loop). For 5 alerts this is imperceptible, but at 100+ alerts it becomes ~10 ms per prediction — unnecessary overhead. Cache compiled patterns in the `StructuralAlert` dataclass.

Running total: **18 gaps** across Blocks B/C/D.

---

✅ Step D.4 complete.

Say **"next"** for Step D.5 (recommender — how `generate_recommendations` translates ADMET hits + structural alerts into actionable medicinal-chemistry suggestions), or ask questions to stay on D.4.

---

## Step D.5 — the recommender

### 1. The question

`generate_recommendations` is the final post-processing layer of Block D. It converts numeric predictions (ADMET probabilities, structural-alert hits) into **human-readable medicinal-chemistry advice** — the kind of thing a chemist could act on at the bench.

This is a **rule-based translator**, not a model. No learning, no probabilities — just threshold lookups against a curated knowledge table. We'll trace:

1. The `_ADMET_THRESHOLDS` configuration table — what's in it, why those specific endpoints.
2. Per-endpoint logic branches (classification-style vs. regression-style vs. directional).
3. The alert-to-rec translation.
4. The comparison mode (`prev_admet` — molecule vs. prior iteration).
5. The "positive" (strength) recommendations — often-missed UX detail.
6. What's *not* covered and why that matters.

### 2. What the recommender is for

Given a candidate molecule, the predict pipeline produces ~22 ADMET numbers + 5 possible structural alerts + 3 phase probabilities. A chemist looking at `{hERG: 0.73, ames: 0.12, bioavailability_ma: 0.42, ...}` needs:

- Which of these are actually problems? (threshold check)
- Which matter enough to fix first? (severity ranking)
- What structural change might help? (suggestion)
- What improvement can I expect? (expected outcome)

The recommender answers all four in a single pass. Output is a `List[Dict]` — each dict a structured "card" with fields `{type, issue, suggestion, severity, expected_improvement}`.

### 3. The knowledge table — `_ADMET_THRESHOLDS`

Only **7 of the 22 ADMET endpoints** are covered by the threshold table:

```6:29:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
_ADMET_THRESHOLDS = {
    "herg": {"bad": 0.5, "warn": 0.35, ...},
    "ames": {"bad": 0.5, "warn": 0.35, ...},
    "dili": {"bad": 0.5, "warn": 0.35, ...},
    "bioavailability_ma": {"low": 0.5, ...},
    "bbb_martins": {"low": 0.3, ...},
    "clearance_hepatocyte_az": {"bad": 80, "warn": 50, ...},
    "ppbr_az": {"bad": 95, "warn": 85, ...},
}
```

Breakdown by endpoint:

| Endpoint | Threshold type | `bad`/`low`/`warn` | Chemistry rationale |
|---|---|---|---|
| `herg` | probability (high = bad) | 0.5 / 0.35 | QT prolongation → arrhythmia; hERG >50% → cardiac tox risk |
| `ames` | probability (high = bad) | 0.5 / 0.35 | mutagenicity (DNA damage) → regulatory blocker |
| `dili` | probability (high = bad) | 0.5 / 0.35 | drug-induced liver injury (#1 cause of post-approval withdrawal) |
| `bioavailability_ma` | probability (low = bad) | 0.5 | oral F < 50% limits oral dosing |
| `bbb_martins` | probability (low = bad) | 0.3 | BBB penetration (only "bad" if CNS target — see context note) |
| `clearance_hepatocyte_az` | regression value (high = bad) | 80 / 50 (µL/min/10⁶ cells) | fast clearance → short half-life → dosing burden |
| `ppbr_az` | regression value (high = bad) | 95 / 85 (% bound) | high protein binding → low free drug → lowers efficacy |

Each entry carries: threshold(s), human label, severity tier, structural suggestion, expected improvement.

**Why only 7?** The other 15 endpoints (solubility, caco2, half-life, CYP inhibitors, etc.) are either:

- Less immediately actionable (e.g., BBB binding logp — chemists can't easily dial it in isolation),
- Redundant with existing entries (CYP3A4 inhibition ≈ DILI risk proxy),
- Or ambiguous without indication context (BBB — bad for peripheral targets, good for CNS targets).

This is deliberate pruning — a recommender that flags *everything* is noise. **⚠️ Minor gap: coverage trade-off is not documented; a reader looking at the codebase won't know why 15 endpoints are silent.**

### 4. The main loop — four logic branches

The recommender has a dispatch table baked into if-chains, not an explicit switch. Four branches handle the four threshold styles:

**Branch A — bioavailability (probability, low = bad, with positive recognition):**

```53:70:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
if key == "bioavailability_ma":
    if val < cfg["low"]:
        recs.append({"type": "Bioavailability", ...})
    elif val >= 0.7:
        recs.append({"type": "Strength", "issue": f"Good {cfg['label']} ({val:.0%})", ...})
    continue
```

Has **both** a bad-case flag AND a good-case strength note (≥ 0.7 is actively good). Single hard-coded threshold for strength (0.7) — not in `cfg`.

**Branch B — BBB (probability, low = bad only):**

```72:81:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
if key in ("bbb_martins",):
    if val < cfg.get("low", 0):
        recs.append({"type": "ADMET", ...})
    continue
```

Only flags *low* BBB — no warning for high BBB. This is context-fragile: high BBB is bad for peripheral targets and good for CNS targets, but the recommender has no indication flag to disambiguate, so it just stays silent on the high side. Defensible default.

**Branch C — regression endpoints (clearance, PPB; high = bad, two-tier):**

```83:100:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
if key in ("clearance_hepatocyte_az", "ppbr_az"):
    if val > cfg.get("bad", 999):
        recs.append({..., "issue": f"High {cfg['label']} ({val:.1f})", ...})
    elif val > cfg.get("warn", 999):
        recs.append({..., "issue": f"Borderline {cfg['label']} ({val:.1f})", "severity": "low", ...})
    continue
```

Two-tier: "High" (use severity from config) vs. "Borderline" (downgrade severity to "low"). Note the format string uses `:.1f` (decimal) instead of `:.0%` because these are regression values, not probabilities.

**Branch D — toxicity probabilities (hERG/AMES/DILI; high = bad, with positive recognition):**

```102:125:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
if val > cfg.get("bad", 999):
    recs.append({"type": "Safety", "issue": f"{cfg['label']} risk ({val:.0%})", ...})
elif val > cfg.get("warn", 999):
    recs.append({"type": "Safety", "issue": f"Borderline {cfg['label']} ({val:.0%})", "severity": "medium", ...})
elif key in ("herg", "ames", "dili") and val < 0.2:
    recs.append({"type": "Strength", "issue": f"Low {cfg['label']} risk ({val:.0%})", ...})
```

Three-tier: bad (0.5+) → warn (0.35–0.5) → strength (< 0.2). The "Strength" branch uses a hard-coded 0.2 threshold that isn't in `cfg`. A chemist reads the full card and knows which endpoint is actively clean vs. actively risky vs. ambiguous middle.

### 5. Alert recommendations

```39:46:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
for alert in alerts:
    recs.append({
        "type": "Structural Alert",
        "issue": alert,
        "suggestion": "Modify or replace substructure to eliminate the alerting motif",
        "severity": "high",
        "expected_improvement": "Reduce toxicity risk",
    })
```

Per-alert behavior:

- Every triggered alert gets a rec card, tagged `severity: "high"`.
- Suggestion is **generic** ("Modify or replace substructure") — the specific replacement guidance stored in each `StructuralAlert.recommendation` (e.g., "Replace -NO2 with -CN or -CF3") is **not used here**. 

**⚠️ Gap 19 — the per-alert `recommendation` field in `structural_alerts.csv` is ignored by the recommender.** The CSV stores actionable chemistry (e.g., "Convert to amide or replace with -OH/-OCH3") but the recommender returns only a generic string. Fix is one-line: pass the full `StructuralAlert` object (or at least its `recommendation`) through `detect_structural_alerts` instead of just names.

### 6. Comparison mode — `prev_admet`

```127:160:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
if prev_admet:
    improved = []
    regressed = []
    for k in admet_preds:
        if k not in prev_admet:
            continue
        delta = admet_preds[k] - prev_admet[k]
        is_lower_better = k in ("herg", "ames", "dili", "clearance_hepatocyte_az", "ppbr_az")
        if is_lower_better:
            if delta < -0.05:
                improved.append(k)
            elif delta > 0.05:
                regressed.append(k)
        else:
            if delta > 0.05:
                improved.append(k)
            elif delta < -0.05:
                regressed.append(k)
    if improved: recs.append({"type": "Progress", ...})
    if regressed: recs.append({"type": "Regression", ...})
```

When the caller supplies a *previous* iteration's ADMET values, the recommender computes delta per endpoint and sorts into "improved" / "regressed" / "flat" (±0.05 threshold). Produces two summary cards.

**Direction lookup is hard-coded** (`is_lower_better` set). Only 5 endpoints are in that list; all others are treated as "higher is better," which is correct for absorption/permeability/bioavailability but wrong for toxicity endpoints not in the list (e.g., `cyp_inhibitors`).

**⚠️ Gap 20 — `is_lower_better` whitelist is fragile.** When more ADMET endpoints are enabled in `endpoints.yaml`, the delta-direction lookup silently treats them as "higher = better." Cleaner design: read direction from the endpoint config (task type + "is_lower_better" field).

`prev_admet` is **optional** and has no callers in the current codebase that supply it — the plumbing is dead in this clean rebuild. It was designed to support iteration-to-iteration feedback in an interactive UI (Compare/Analyze pages, which are scoped out per the request).

### 7. The empty-state card

```162:169:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/recommender.py
if not recs:
    recs.append({
        "type": "Status",
        "issue": "No critical flags detected",
        ...
    })
```

If no alerts fired and no thresholds tripped, return a single "all clear" card. Prevents the caller from getting an empty list (which a naive UI would show as a blank box).

### 8. Aspirin worked example

Suppose aspirin's ADMET predictions (hypothetical):

| Endpoint | Value | Branch | Rec? |
|---|---|---|---|
| `herg` | 0.08 | D (low, below 0.2) | **Strength** (low cardiotox) |
| `ames` | 0.05 | D (low, below 0.2) | **Strength** (low mutagenicity) |
| `dili` | 0.11 | D (low, below 0.2) | **Strength** (low hepatotox) |
| `bioavailability_ma` | 0.78 | A (≥ 0.7) | **Strength** (good oral F) |
| `bbb_martins` | 0.42 | B (above 0.3) | silent |
| `clearance_hepatocyte_az` | 35 µL/min | C (below 50) | silent |
| `ppbr_az` | 68% | C (below 85) | silent |
| structural alerts | none (all 5 clear) | alerts loop | no cards |

**Result: 4 cards, all positive** — 3 toxicity strengths + 1 bioavailability strength. A chemist sees "this is a clean molecule, no red flags" at a glance.

### 9. Output shape

A typical rec card:

```python
{
    "type": "Safety",  # Safety / Bioavailability / ADMET / Structural Alert / Strength / Progress / Regression / Status
    "issue": "hERG inhibition risk (72%)",
    "suggestion": "Reduce LogP or remove basic amines",
    "severity": "high",  # high / medium / low / positive
    "expected_improvement": "Lower cardiotoxicity risk",
}
```

Fields are intentionally flat (no nesting) so the list is trivially serializable to JSON for any downstream consumer (API response, report export, etc.).

### 10. Severity palette

Five severity levels used across all rec sources:

| Severity | Meaning | Intended UI treatment |
|---|---|---|
| `high` | immediate action needed | red / warning |
| `medium` | borderline / recent regression | amber |
| `low` | mild note / borderline regression endpoint | yellow |
| `positive` | a strength to preserve | green |
| (no `critical`) | severity isn't used here; "critical" lives only in alert severity field (D.4, unused) | — |

This palette is **inconsistent with the structural alerts severity field** (D.4: `critical/high/medium`). Two different severity scales coexist. A careful integration would unify them, but currently recs use one scale and alerts use another. **Not a gap — just a minor inconsistency worth noting.**

### 11. Data transformation trace

| Stage | Input | Output |
|---|---|---|
| Call site (`drug_oracle.predict`) | `admet_preds: Dict[str, float]`, `alerts: List[str]` | `recs: List[Dict]` |
| Alert loop | `alerts` (list of names) | one "Structural Alert" card per hit |
| ADMET threshold loop | 7 known endpoints from `_ADMET_THRESHOLDS` | 0+ cards (Safety / ADMET / Bioavailability / Strength) |
| Comparison block | `prev_admet` if supplied | 0/1/2 Progress/Regression cards |
| Empty-state fallback | `recs == []` | single "No flags" card |
| Final | all cards concatenated | `List[Dict]` |

### 12. Alternatives

| Alternative | Trade-off |
|---|---|
| LLM-based recommender (GPT / Claude with prompts including ADMET values + alerts) | richer, context-aware advice; latency + cost + non-determinism + hallucination risk |
| Graph-based structural edit suggestions (match alert → look up known bioisosteres → propose specific SMILES edits) | far more actionable; needs a bioisostere database + edit tool |
| Per-indication recommender (CNS vs. oncology vs. metabolic) | more precise (BBB direction matters); needs indication metadata on input |
| Learned recommender (train on chemist feedback) | adaptive; needs labeled feedback data |
| Drop the recommender entirely | simpler; forces consumer to build its own UI on raw predictions |

The rule-based recommender is the simplest useful layer. At this scale it's robust, fast (<1 ms), and deterministic. An LLM layer could sit *on top* of it (summarize recs in natural language), but replacing it with an LLM would sacrifice determinism.

### 13. How & Why

- **How:** Fixed threshold table covering 7 of 22 ADMET endpoints + 5 alert patterns.
  **Why:** only endpoints with clear medicinal-chemistry action items are included. Noise reduction.
- **How:** Per-card dict with `{type, issue, suggestion, severity, expected_improvement}`.
  **Why:** flat JSON-safe schema; any UI can render it trivially.
- **How:** Both "bad" (flag) and "positive" (strength) cards; empty-state fallback.
  **Why:** UX parity — a clean molecule should *show* its strengths, not silently produce an empty list.
- **How:** Alert recommendations use a generic suggestion string.
  **Why:** current behavior. The per-alert `recommendation` in the CSV is richer but not threaded through (Gap 19).
- **How:** `prev_admet` comparison path exists but no caller uses it.
  **Why:** legacy hook for the iteration/compare UI that was scoped out.

### 14. Novelty ledger

| Component | Status | Source |
|---|---|---|
| Threshold-based ADMET flagging | Reused pattern | standard med-chem rules |
| Specific thresholds (0.5/0.35/0.2, 80/50, 95/85) | Reused from literature + project-tuned | values align with published ADMET interpretation guidelines |
| Severity palette + rec-card schema | Project-specific | UI-driven schema |
| Empty-state fallback | Project-specific | UX detail |

### 15. Verified?

| Claim | Source tier | Location |
|---|---|---|
| 7 endpoints covered in `_ADMET_THRESHOLDS` | `repo` | `recommender.py` lines 6–29 |
| Per-alert rec uses generic suggestion, ignores CSV `recommendation` field | `repo` | `recommender.py` lines 39–46 (no read of `alert.recommendation`) |
| Four-branch dispatch (bioavailability / BBB / regression / toxicity) | `repo` | `recommender.py` lines 53, 72, 83, 102 |
| Strength path exists for hERG/AMES/DILI at val < 0.2 | `repo` | `recommender.py` lines 118–125 |
| `prev_admet` path with delta threshold ±0.05 | `repo` | `recommender.py` lines 133–144 |
| `is_lower_better` whitelist hardcoded to 5 endpoints | `repo` | `recommender.py` line 134 |
| Empty-state "No critical flags detected" fallback | `repo` | `recommender.py` lines 162–169 |
| Severity values used: high / medium / low / positive | `repo` | grep on `recommender.py` |

### 16. Gap ledger update

New gaps added in this step:

- **⚠️ Gap 19 (D.5):** Per-alert `recommendation` text from `structural_alerts.csv` is ignored by the recommender; all alerts get the generic "Modify or replace substructure" string. The richer advice (e.g., "Replace -NO2 with -CN or -CF3") is stored but never surfaced. One-line fix: thread `StructuralAlert` object (not just name) through `detect_structural_alerts`.
- **⚠️ Gap 20 (D.5):** `is_lower_better` direction whitelist is hardcoded to 5 endpoints. Enabling new endpoints in `endpoints.yaml` means the comparison mode silently mis-classifies their deltas. Direction should be stored in endpoint config, not in the recommender.

Running total: **20 gaps** across Blocks B/C/D.

---

✅ Step D.5 complete.

**Block D is now substantially covered:** orientation + D.1 (clinical data) + D.2 (cascade architecture) + D.3 (training loop) + D.4 (structural alerts) + D.5 (recommender). 

The DrugOracle component itself — `DrugOracle.predict` in `drug_oracle.py` — wires these 5 pieces together. We've already read it in D.4. If you want, D.6 can close the block by tracing `DrugOracle.predict` end-to-end as an integration step (showing the full data flow from SMILES to `OraclePrediction`), then cross-linking Block D back to the system loop and RL reward (which consumes Oracle scores).

Say **"next"** for Step D.6 (`DrugOracle.predict` integration pass — end-to-end walkthrough + RL reward coupling), or ask questions to stay on D.5, or say **"stop block d"** to close Block D here and return to the main `LEARNING_GUIDE.md` for whatever comes next (e.g., reranker, pipelines, full-system integration).

---

## Step D.6 — `DrugOracle.predict` integration + RL reward coupling

### 1. The question

Five sub-systems have been built in isolation: cascaded predictors (D.2), trainer (D.3), structural alerts (D.4), recommender (D.5), and the clinical-quality scorer introduced in the D orientation. They're wired together in `DrugOracle.predict`, which is the single callable that external code sees.

This final step traces:

1. The `DrugOracle` class as a composition facade.
2. `from_pretrained` — how the Oracle loads itself (both ADMET + cascade checkpoints).
3. `DrugOracle.predict` — single SMILES → `OraclePrediction`, line by line.
4. `OraclePrediction` — the dataclass contract with consumers.
5. The coupling with Block B's RL fine-tuning: how `compute_reward_per_smiles` and `compute_rewards_per_sample` use Oracle output as a reward signal.
6. Three integration shapes the reward path accepts (scalar fn, prediction fn, override list) and when each is preferred.
7. Aspirin end-to-end worked example through the full `DrugOracle.predict`.
8. Closing gaps and novelty ledger for Block D.

### 2. `DrugOracle` as a composition facade

`DrugOracle` (in `drug_oracle.py`) holds three things:

```57:63:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
class DrugOracle:
    def __init__(self, oracle_model, admet_model, endpoint_task_types: Dict[str, str], device: str = "cpu"):
        self.oracle_model = oracle_model.to(device)
        self.admet_model = admet_model
        self.endpoint_task_types = endpoint_task_types
        self.device = device
```

- `oracle_model`: the trained `CascadedPhasePredictors` (D.2/D.3).
- `admet_model`: the trained ADMET multi-task predictor (Block C).
- `endpoint_task_types`: dict of `{endpoint_name: "classification"|"regression"}` — needed because `predict_smiles` must know which heads to sigmoid (from C.7).

It exposes two operations: load (`from_pretrained`) and predict (`predict`). Everything else (`_predict_oracle`, `_clinical_quality`) is private.

### 3. `from_pretrained` — smart checkpoint loading

```64:90:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
@classmethod
def from_pretrained(
    cls,
    oracle_path: str,
    admet_path: str,
    endpoint_names: List[str],
    endpoint_task_types: Dict[str, str],
    input_dim: int,
    device: str = "cpu",
) -> "DrugOracle":
    admet_model = load_model(
        checkpoint_path=admet_path,
        endpoint_names=endpoint_names,
        num_node_features=input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        device=device,
    )
    state = torch.load(oracle_path, map_location=device, weights_only=False)
    model_state = state.get("model", state)
    in_dim_ckpt = int(model_state["phase1.net.0.weight"].shape[1])
    hidden_dim_ckpt = int(model_state["phase1.net.0.weight"].shape[0])
    oracle = CascadedPhasePredictors(in_dim=in_dim_ckpt, hidden_dim=hidden_dim_ckpt).to(device)
    oracle.load_state_dict(model_state, strict=True)
    oracle.eval()
    return cls(oracle, admet_model, endpoint_task_types, device=device)
```

Two nice details:

**(a) ADMET model is loaded with fixed architecture hyperparameters** — `hidden_dim=128, num_layers=3, dropout=0.1`. These must match what was used to train the ADMET checkpoint (see C.6). Hardcoding here is fragile if ADMET training config ever changes. ⚠️ Worth noting — a cleaner design would read architecture from the ADMET checkpoint's own metadata.

**(b) Oracle's `in_dim` and `hidden_dim` are *inferred from the checkpoint*** — not from config. Lines 85–86 read the `phase1.net.0.weight` tensor's shape:

- `.shape[1]` = input dim (number of ADMET features, typically 22).
- `.shape[0]` = hidden dim (256 by default).

This makes the Oracle **self-describing**: the checkpoint contains enough metadata to rebuild the exact same architecture. If you later train a cascade with different dimensions, `from_pretrained` adapts automatically. This is good engineering.

`strict=True` on line 88 ensures no silent weight mismatch. `.eval()` on line 89 disables dropout for inference.

### 4. `DrugOracle.predict` — line by line

```112:144:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
def predict(self, smiles: str) -> Optional[OraclePrediction]:
    if not validate_smiles(smiles):
        return None
    admet_preds = predict_smiles(self.admet_model, smiles, self.endpoint_task_types, device=self.device)
    if not admet_preds:
        return None
    probs = self._predict_oracle(admet_preds)
    alerts, alert_atoms = detect_structural_alerts(smiles)
    recs = generate_recommendations(admet_preds, alerts)
    risk_factors = []
    for name, val in admet_preds.items():
        if name in {"herg", "ames", "dili"} and val > 0.5:
            risk_factors.append(
                RiskFactor(
                    name=name.upper(),
                    category="toxicity",
                    description="Predicted risk above threshold",
                    impact=float(val),
                    source="ADMET",
                )
            )
    overall = self._clinical_quality(probs["phase1"], probs["phase2"], probs["phase3"], admet_preds, alerts)
    return OraclePrediction(
        phase1_prob=probs["phase1"],
        phase2_prob=probs["phase2"],
        phase3_prob=probs["phase3"],
        overall_prob=overall,
        admet_predictions=admet_preds,
        risk_factors=risk_factors,
        structural_alerts=alerts,
        recommendations=recs,
        alert_atoms=alert_atoms,
    )
```

Walkthrough:

1. **Line 113 — SMILES validation.** `validate_smiles` (in `utils/chemistry.py`) runs `Chem.MolFromSmiles` and checks for kekulization. Invalid SMILES → early `None` return. This is the first fail-fast guard.
2. **Line 115 — ADMET inference.** Calls `predict_smiles` (C.7). Returns a dict `{endpoint_name: float}` for all 22 endpoints, with sigmoid already applied for classification heads. This is the **bottleneck step** — ~20 ms on CPU per SMILES.
3. **Line 116 — second guard.** If ADMET inference returns empty (shouldn't normally happen after validation, but defensive), bail.
4. **Line 118 — Oracle cascade forward pass.** Calls `_predict_oracle(admet_preds)` (lines 102–110):
   - Pack ADMET values into a tensor `[list(admet_preds.values())]` (shape `(1, 22)`).
   - Run the cascade → 3 logits.
   - Sigmoid each → `{phase1, phase2, phase3}` probabilities.
   - Wrapped in `torch.no_grad()` — inference-only.
5. **Line 119 — structural alerts.** `detect_structural_alerts(smiles)` (D.4). Returns `(hits: List[str], alert_atoms: np.ndarray)`.
6. **Line 120 — recommender.** `generate_recommendations(admet_preds, alerts)` (D.5). Returns `List[Dict]`.
7. **Lines 121–132 — risk-factor aggregation.** Only 3 endpoints (`herg`, `ames`, `dili`) are checked at the 0.5 threshold; if hit, a `RiskFactor` dataclass is constructed with source="ADMET". Structural-alert-based risk factors are **not** added here (intentional — structural alerts are in their own list).
8. **Line 133 — composite score.** `_clinical_quality(p1, p2, p3, admet, alerts)` applies the phase-weighted formula + penalties (D orientation §5).
9. **Lines 134–144 — build the `OraclePrediction` dataclass.** Returned to caller.

**Key dataflow — one pass, no caching.** Every call repeats all 5 sub-operations. For batch scoring, callers must loop externally — there's no batched `predict_many`. This is the throughput bottleneck when coupling with RL (which generates 32–128 SMILES per step). See Gap 7 (C.7) + Gap 15 (D.3) — both compound here.

### 5. The `OraclePrediction` contract

```32:54:/Users/sreevardhandesu/Desktop/prj_demo/models/oracle/drug_oracle.py
@dataclass
class OraclePrediction:
    phase1_prob: float
    phase2_prob: float
    phase3_prob: float
    overall_prob: float
    admet_predictions: Dict[str, float]
    risk_factors: List[RiskFactor]
    structural_alerts: List[str]
    recommendations: List[Dict]
    alert_atoms: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "phase1_prob": self.phase1_prob,
            ...
            "structural_alerts": self.structural_alerts,
            "recommendations": self.recommendations,
        }
```

- **9 fields** (+ optional `alert_atoms`).
- `to_dict()` serializes for JSON; note it **omits `alert_atoms`** (numpy array, not JSON-safe).
- `risk_factors` entries are `RiskFactor` dataclasses — also serialized via `__dict__` in `to_dict`.

Consumers of `OraclePrediction`:

1. **RL reward computation** (Block B.5): reads `phase1_prob` / `phase2_prob` / `phase3_prob` / `overall_prob` / `structural_alerts` / `risk_factors`.
2. **Reranker** (future — not in this clean build): would read all fields.
3. **Reporting / export**: calls `to_dict()`.

### 6. Coupling with RL — the reward path

This is where Block D closes the loop with Block B. The generator in RL fine-tuning (B.5) needs a scalar reward per generated SMILES. Block D supplies it through three integration shapes, all implemented in `rewards.py`.

**Shape 1 — scalar score function** (simplest, legacy):

```27:40:/Users/sreevardhandesu/Desktop/prj_demo/models/generator/rewards.py
def _oracle_scalar(
    fn: Optional[Callable[[str], Union[float, Dict[str, float], None]]],
    smiles: str,
    phase_weights: Optional[Tuple[float, float, float]] = None,
) -> float:
    out = fn(smiles) if fn else None
    if out is None:
        return 0.0
    if isinstance(out, dict) and phase_weights is not None:
        p1 = out.get("phase1", 0.0)
        p2 = out.get("phase2", 0.0)
        p3 = out.get("phase3", 0.0)
        return phase_weights[0] * p1 + phase_weights[1] * p2 + phase_weights[2] * p3
    return float(out)
```

Caller passes `oracle_score_fn: Callable[[str], Dict|float]`. If it returns `{"phase1","phase2","phase3"}`, those are phase-weighted. If it returns a scalar, used directly. If `None`, reward = 0.

**Shape 2 — full prediction function** (new, richer):

```43:70:/Users/sreevardhandesu/Desktop/prj_demo/models/generator/rewards.py
def _scalar_from_prediction(
    pred: Any,
    phase_weights: Optional[Tuple[float, float, float]] = None,
) -> float:
    if pred is None:
        return 0.0
    if isinstance(pred, dict):
        if phase_weights is not None:
            p1 = pred.get("phase1_prob", pred.get("phase1", 0.0))
            ...
        return float(pred.get("overall_prob", 0.0))
    if phase_weights is not None:
        return (
            phase_weights[0] * getattr(pred, "phase1_prob", 0.0)
            + phase_weights[1] * getattr(pred, "phase2_prob", 0.0)
            + phase_weights[2] * getattr(pred, "phase3_prob", 0.0)
        )
    return float(getattr(pred, "overall_prob", 0.0))


def _alert_penalty(pred: Any) -> int:
    if pred is None:
        return 0
    alerts = getattr(pred, "structural_alerts", None) or (pred.get("structural_alerts", []) if isinstance(pred, dict) else [])
    risks = getattr(pred, "risk_factors", None) or (pred.get("risk_factors", []) if isinstance(pred, dict) else [])
    return len(alerts) + len(risks)
```

Caller passes `oracle_prediction_fn: Callable[[str], OraclePrediction|dict]`. The reward code then has access to structural alerts and risk factors, and can subtract `w_alert × (num_alerts + num_risk_factors)`. This is the **preferred integration** — richer signal, direct from `DrugOracle.predict`.

**Shape 3 — precomputed override list** (batch optimization):

```107:120:/Users/sreevardhandesu/Desktop/prj_demo/models/generator/rewards.py
if oracle_scores_override is not None and len(oracle_scores_override) == len(smiles_list):
    oracle_scalars = oracle_scores_override
elif oracle_prediction_fn is not None:
    oracle_scalars = []
    for s in smiles_list:
        pred = oracle_prediction_fn(s)
        sc = _scalar_from_prediction(pred, phase_weights) - w_alert * _alert_penalty(pred)
        oracle_scalars.append(sc)
else:
    oracle_scalars = [
        _oracle_scalar(oracle_score_fn, s, phase_weights) if oracle_score_fn else 0.0
        for s in smiles_list
    ]
```

Caller supplies `oracle_scores_override: List[float]` — already computed. Skips Oracle evaluation entirely. Useful when Oracle scores were computed upstream (e.g., in a separate process) and cached for the RL step.

### 7. The composite reward formula

```73:92:/Users/sreevardhandesu/Desktop/prj_demo/models/generator/rewards.py
def compute_reward_per_smiles(
    smiles: str,
    ...
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
    validity_gated_oracle: bool = True,
    ...
    w_alert: float = 0.0,
) -> float:
    validity = validity_reward(smiles)
    qed = qed_reward(smiles)
    if oracle_prediction_fn is not None and w_alert != 0:
        pred = oracle_prediction_fn(smiles)
        oracle_raw = _scalar_from_prediction(pred, phase_weights) - w_alert * _alert_penalty(pred)
    else:
        oracle_raw = _oracle_scalar(oracle_score_fn, smiles, phase_weights) if oracle_score_fn else 0.0
    oracle = oracle_raw * validity if validity_gated_oracle else oracle_raw
    return w_validity * validity + w_qed * qed + w_oracle * oracle
```

Per-sample reward:

\[
R(s) = w_{\text{val}} \cdot v(s) + w_{\text{qed}} \cdot \text{QED}(s) + w_{\text{orc}} \cdot \tilde{O}(s)
\]

where

- \(v(s) \in \{0,1\}\) — validity (RDKit parseable).
- \(\text{QED}(s) \in [0,1]\) — drug-likeness (Bickerton et al. 2012).
- \(\tilde{O}(s) = \big[w_1 p_1 + w_2 p_2 + w_3 p_3 - w_{\text{alert}} \cdot (n_{\text{alerts}} + n_{\text{risks}})\big] \cdot v(s)\).

Defaults: `w_validity = w_qed = w_oracle = 0.3`, `w_alert = 0` (alert penalty disabled by default).

**`validity_gated_oracle = True`** on line 91: **invalid SMILES → oracle contribution zeroed**. Prevents the RL agent from being "rewarded" by the Oracle for scoring garbage strings. Important guardrail.

**Batch-level diversity bonus** added in `compute_rewards_per_sample`:

```129:/Users/sreevardhandesu/Desktop/prj_demo/models/generator/rewards.py
return [b + (w_diversity * diversity) for b in base]
```

Diversity = `unique_smiles / batch_size`. Same scalar added to every sample in the batch. Encourages the policy to produce varied outputs per rollout. Default `w_diversity = 0.1`.

### 8. Aspirin end-to-end worked example

Input SMILES: `CC(=O)Oc1ccccc1C(=O)O`.

**`DrugOracle.predict` trace:**

1. `validate_smiles` → valid ✓
2. `predict_smiles` → ADMET dict, e.g., `{hERG: 0.08, ames: 0.05, dili: 0.11, bioavailability_ma: 0.78, ...}` (22 entries; fictitious but plausible)
3. `_predict_oracle(admet)` → `{phase1: 0.86, phase2: 0.67, phase3: 0.40}` (from D.2's example)
4. `detect_structural_alerts` → `([], alert_atoms=zeros(13))` (D.4: aspirin clears all 5 alerts)
5. `generate_recommendations` → 4 Strength cards (from D.5)
6. `risk_factors` loop → empty (hERG/AMES/DILI all < 0.5)
7. `_clinical_quality`:
   - `base = 0.2·0.86 + 0.5·0.67 + 0.3·0.40 = 0.172 + 0.335 + 0.120 = 0.627`
   - `penalty = 0` (no hERG/AMES/DILI > 0.5, no alerts)
   - `overall = clip(0.627 - 0, 0, 1) = 0.627`
8. Build `OraclePrediction`:
   - `phase1_prob=0.86, phase2_prob=0.67, phase3_prob=0.40, overall_prob=0.627`
   - `admet_predictions={...}, risk_factors=[], structural_alerts=[], recommendations=[4 cards]`

**Then if aspirin feeds into RL reward** (default weights, `w_alert=0`):

- `validity = 1.0`
- `qed = 0.56` (approximate; aspirin's actual QED)
- `oracle_raw = 0.627` (the `overall_prob`)
- `oracle = 0.627 × 1.0 = 0.627` (validity gated, passes)
- `R = 0.3·1.0 + 0.3·0.56 + 0.3·0.627 = 0.3 + 0.168 + 0.188 = **0.656**`

If this was in a batch of 32 unique molecules, diversity contribution adds `0.1 × (32/32) = 0.1` → total reward ≈ **0.756**.

### 9. Why the reward is linear in Oracle output

Cascading the Oracle's overall clinical probability into a linear reward has one subtle property: the reward is **smooth and differentiable** in any continuous function of `DrugOracle.predict` outputs. This matters because:

- REINFORCE uses `R` as a multiplier on log-probabilities; any scalar is fine.
- PPO clips the policy ratio but uses the advantage `A = R - V(s)` linearly; again, any scalar is fine.
- The composite design lets you reweight components (e.g., `w_oracle = 0.5, w_qed = 0.2, w_validity = 0.1, w_diversity = 0.2`) without retraining anything — reward weights are hyperparameters.

**Why not use raw phase3 probability as reward?** Phase 3 alone would ignore Phase 1/2 conditioning and the alert/ADMET penalties. The `overall_prob` is a curated blend designed for this purpose. But `phase_weights` is exposed as a hyperparameter — callers who want to optimize for late-stage success specifically can pass `(0.0, 0.0, 1.0)`.

### 10. Integration with the system loop

Block D sits downstream of Block C (ADMET) and upstream of Block B (as a reward signal). The composition:

```
Block A (data)
  → Block B generator (proposes SMILES)
  → Block C ADMET (scores each SMILES on 22 endpoints)
  → Block D DrugOracle (adds phase probs, alerts, recs, composite score)
  → Reward function (folds into scalar R)
  → Block B RL update (gradient on R)
  → ... loop ...
```

Three backward dependencies:

1. **D needs C's checkpoint** at serve time. No ADMET model → Oracle can't compute `admet_preds` → predict returns `None`.
2. **D needs the C endpoint task types.** `predict_smiles` requires it to know which heads to sigmoid.
3. **D's cascade architecture** (`in_dim=22`, `hidden_dim=256`) must match the ADMET output dimension. Inferred at load time (good engineering, §3 above).

One forward dependency:

4. **B's reward function consumes D's output.** Three integration shapes support flexibility (§6).

### 11. Block D closing thoughts — interpretive summary

**What Block D does well:**

- Clean separation: cascade (learned) + alerts (rule-based) + recommender (rule-based). Each is independently interpretable.
- Self-describing checkpoints (inferred `in_dim`/`hidden_dim`).
- Validity-gated oracle reward (prevents garbage-reward amplification).
- Three reward integration shapes → forward-compatible.
- Strength-recognition in the recommender (positive UX parity).

**What Block D does badly:**

- Hardcoded clinical trial dataset path + no downloader (Gap 10).
- No val split / early stopping / scheduler / grad clip (Gaps 8, 9).
- No class weighting despite inherent imbalance (Gap 11).
- No invalid-SMILES / NaN filtering in dataset loader (Gap 12).
- Fragile feature ordering vs. Block C alignment (Gap 13).
- ADMET inference recomputed per-sample-per-epoch (Gap 15).
- Severity field in alerts is stored but ignored (Gap 16).
- Only 5 alert patterns shipped (Gap 17).
- SMARTS recompiled per call (Gap 18).
- Per-alert recommendation text in CSV unused (Gap 19).
- `is_lower_better` direction hardcoded (Gap 20).

**What Block D is NOT:**

- Not a survival model (no censoring handling).
- Not a calibrated probability model (BCE doesn't guarantee calibration; no isotonic / Platt scaling).
- Not a wet-lab prediction (no experimental validation).
- Not a fairness-audited predictor (no subgroup analysis).

**Block D is:** a useful, interpretable, fast proxy for overall "clinical worth" of a candidate. Good enough to drive RL reward; not good enough to replace clinical judgment.

### 12. Verified?

| Claim | Source tier | Location |
|---|---|---|
| `DrugOracle` holds oracle_model, admet_model, endpoint_task_types | `repo` | `drug_oracle.py` lines 58–62 |
| `from_pretrained` loads ADMET with fixed architecture (hidden_dim=128, num_layers=3) | `repo` | `drug_oracle.py` lines 74–82 |
| Oracle `in_dim` / `hidden_dim` inferred from `phase1.net.0.weight` shape | `repo` | `drug_oracle.py` lines 85–86 |
| `strict=True` loading; `.eval()` after load | `repo` | `drug_oracle.py` lines 88–89 |
| `predict` returns `None` on invalid SMILES or empty ADMET | `repo` | `drug_oracle.py` lines 113–117 |
| `_predict_oracle` wraps cascade forward + sigmoid in `torch.no_grad()` | `repo` | `drug_oracle.py` lines 102–110 |
| `OraclePrediction` has 9 required + 1 optional field | `repo` | `drug_oracle.py` lines 32–42 |
| `to_dict()` omits `alert_atoms` (numpy array) | `repo` | `drug_oracle.py` lines 44–54 |
| Three reward integration shapes (scalar fn, prediction fn, override list) | `repo` | `rewards.py` lines 27–40, 43–62, 107–120 |
| Default reward weights: 0.3/0.3/0.3/0.1 (validity/qed/oracle/diversity), w_alert=0 | `repo` | `rewards.py` lines 76–82, 98–101 |
| Validity-gated oracle (invalid SMILES zeros oracle contribution) | `repo` | `rewards.py` line 91 |
| Alert penalty is `len(alerts) + len(risk_factors)` × `w_alert` | `repo` | `rewards.py` lines 65–70, 88 |

### 13. Novelty ledger (Block D overall)

| Component | Status | Source |
|---|---|---|
| Cascaded phase predictors (architecture) | Project-specific composition | MLPs + sigmoid feedback — the cascade applied to clinical trial phases is the project's choice |
| MLP building block | Reused | Srivastava et al. 2014 (dropout); standard PyTorch |
| BCE + Adam training | Reused | PyTorch idiom |
| SMARTS structural alerts | Reused concept | Ashby & Tennant 1988; Kazius et al. 2005 |
| Composite clinical-quality score | Project-specific | specific `0.2/0.5/0.3 − 0.12·(herg+ames+dili) − 0.08·alerts` formula |
| Rule-based recommender with strength cards | Project-specific | UX layer |
| Three-shape reward integration (scalar/pred/override) | Project-specific | designed for flexibility across RL/reranker/batch use |
| Validity-gated oracle reward | Project-specific | guardrail |
| QED | Reused | Bickerton et al. 2012 |

### 14. Final gap ledger (Block D)

1. Gap 8 — no validation split or early stopping in Oracle training.
2. Gap 9 — hyperparameters hardcoded in `train_oracle.py` (not config-driven).
3. Gap 10 — no clinical trial dataset downloader.
4. Gap 11 — no `pos_weight` for BCE despite class imbalance.
5. Gap 12 — no invalid-SMILES / NaN filtering in `OracleDataset`.
6. Gap 13 — fragile feature ordering (`list(preds.values())` vs. `sorted(keys)`).
7. Gap 14 — no minimum dataset size warning.
8. Gap 15 — ADMET inference recomputed per sample per epoch.
9. Gap 16 — severity field in alerts unused in penalty math.
10. Gap 17 — only 5 structural alerts shipped.
11. Gap 18 — SMARTS patterns recompiled per call.
12. Gap 19 — per-alert `recommendation` from CSV ignored by recommender.
13. Gap 20 — `is_lower_better` direction hardcoded.

**Total Block D gaps: 13** (Gaps 8–20, some numbered earlier in orientation).

Combined running total across Blocks B/C/D: **20 gaps**.

---

✅ Step D.6 complete. ✅ **Block D (DrugOracle) guide complete.**

Block D coverage: orientation → D.1 (clinical dataset) → D.2 (cascade architecture) → D.3 (training loop) → D.4 (structural alerts) → D.5 (recommender) → D.6 (integration + RL coupling).

**Return to `LEARNING_GUIDE.md`** for the remaining system topics:

- **Reranker** — ranks candidate molecules across validity + QED + Oracle + diversity + novelty
- **Data pipelines** — end-to-end orchestration (A → B → C → D → rerank → report)
- **End-to-end integration** — tying generator + ADMET + Oracle + reranker in a single inference path
- **Tests** — what the `tests/` suite covers
- **Any remaining gaps** — synthesizing the 20 gaps into a roadmap

Say **"next"** and the next step will resume in `LEARNING_GUIDE.md` with the reranker (or tell me if you want a different topic first).
