# SafeMolGen-DrugOracle: Detailed Methodology & Workflow Report

This document describes the end-to-end methodology, data flow, model components, and workflows used in the SafeMolGen-DrugOracle project.

---

## 1. Project Overview & Objectives

**SafeMolGen-DrugOracle** is an integrated AI system for drug design that combines:

1. **ADMET prediction** — Multi-task prediction of absorption, distribution, metabolism, excretion, and toxicity (22 endpoints from TDC).
2. **DrugOracle** — Clinical phase success estimation (Phase I → II → III) with explainability (structural alerts, risk factors, recommendations).
3. **SafeMolGen** — SMILES-based molecular generator (transformer decoder) for candidate generation.
4. **Integration** — Iterative design: generate candidates → score with Oracle → select best → repeat until target success or max iterations.
5. **Web UI (FastAPI + React)** — Generate, Analyze, Compare molecules and view Oracle outputs.

The workflow is organized in **four phases**, each with its own data pipeline, models, and training scripts.

---

## 2. High-Level Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA & TRAINING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: ADMET                                                                  │
│  TDC ADMET Group → download_data.py → preprocess_data.py → train_admet.py        │
│       → checkpoints/admet/best_model.pt                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Phase 2: DrugOracle                                                              │
│  Clinical trials CSV + ADMET model → train_oracle.py → checkpoints/oracle/        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Phase 3: SafeMolGen                                                              │
│  ChEMBL/ADMET SMILES → train_generator.py (pretrain → optional RL)                │
│       → checkpoints/generator/ (model.pt, tokenizer.json)                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE & DESIGN PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Load: Generator + Oracle (ADMET + Phase predictors)                               │
│  Design loop:                                                                     │
│    Generate N candidates (SafeMolGen) → Evaluate each (DrugOracle)                │
│    → Rank by overall_prob → Update best → Repeat until target or max_iter         │
│  Output: DesignResult (final_smiles, phase probs, history, recommendations)        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**→ For a top-down flowchart (training + design loop), see [PROJECT_FLOWCHART.md](PROJECT_FLOWCHART.md).**

---

## 3. Phase 1: ADMET Predictor

### 3.1 Purpose

Train a single multi-task model that predicts 22 ADMET endpoints from molecular structure. Molecules are represented as **graphs**; a **GNN encoder** produces a graph embedding that is then used by task-specific heads.

### 3.2 Data Workflow

| Step | Script / Component | Input | Output |
|------|--------------------|--------|--------|
| 1 | `scripts/download_data.py` | `config/endpoints.yaml` | TDC ADMET Group data per endpoint: `data/admet_group/<endpoint>/train_val.csv`, `test.csv` (or `data/processed/admet/<endpoint>/train.csv`, `val.csv`, `test.csv` via `TDCDataLoader.save_splits`) |
| 2 | `utils/data_utils.TDCDataLoader` | Endpoint config (name, tdc_name, task_type, metric) | For each endpoint: fetch benchmark → 90/10 train/val split (stratified if classification) → save CSV splits |
| 3 | `scripts/preprocess_data.py` | CSV splits under `data/processed/admet/<endpoint>/` | For each split: SMILES → molecular graph (RDKit) → `train.pt`, `val.pt`, `test.pt` (list of PyG `Data` objects) |

**Data source:** TDC ADMET Group (22 benchmarks: Caco2_Wang, HIA_Hou, Pgp_Broccatelli, Bioavailability_Ma, Lipophilicity_AstraZeneca, Solubility_AqSolDB, BBB_Martins, PPBR_AZ, VDss_Lombardo, CYP variants, Half_Life_Obach, Clearance_*, LD50_Zhu, hERG, AMES, DILI, etc.). See `config/endpoints.yaml` and `docs/reports/DATA_SOURCES.md`.

### 3.3 Molecular Graph Construction

- **Location:** `utils/chemistry.smiles_to_graph()` (and `MoleculeProcessor` wrapper).
- **Nodes (atoms):** Feature vector per atom from `_atom_features()`:
  - Atomic number, degree, formal charge, total H count, aromatic flag.
  - One-hot hybridization (SP, SP2, SP3, SP3D, SP3D2).
- **Edges:** Bonds (undirected); each bond adds `[i,j]` and `[j,i]` to `edge_index`. Optional `edge_attr`: bond type (single/double/triple/aromatic), conjugated, in ring (6-D).
- **Output:** PyTorch Geometric `Data(x, edge_index, edge_attr, smiles)`; targets `y` attached during preprocessing.

### 3.4 Model Architecture

- **GNN Encoder** (`models/admet/gnn_encoder.GNNEncoder`):
  - **Layers:** 3× GIN (Graph Isomorphism Network) via PyTorch Geometric `GINConv`.
  - Each layer: MLP(2 layers) → GINConv → LayerNorm → ReLU → Dropout.
  - Input: node features `x`, `edge_index`. Output: node-level embeddings (same shape as nodes, no edge features in conv).
- **Pooling:** Attention pooling (`models/admet/attention_pooling.AttentionPooling`): learned attention over nodes → single graph embedding.
- **Heads:** One linear head per endpoint (`MultiTaskADMETPredictor`): graph embedding → scalar logit (classification) or value (regression).

**Config:** `config/config.yaml`: `gnn_hidden_dim: 128`, `gnn_layers: 3`, `dropout: 0.1`, `batch_size: 64`, `epochs: 10`, `lr: 0.001`, `weight_decay: 0.0001`.

### 3.5 Training Workflow

- **Script:** `scripts/train_admet.py`.
- **Process:**
  1. Load config and endpoints; build per-endpoint `DataLoader`s from `data/processed/admet/<endpoint>/train.pt` and `val.pt`.
  2. Infer `num_node_features` from a sample batch.
  3. Instantiate `MultiTaskADMETPredictor` (GNN + attention pool + multi-head).
  4. `ADMETTrainer`: for each epoch, iterate over all endpoint loaders; for each batch compute multi-task outputs, endpoint-specific loss (BCE or MSE via `models/admet/losses.get_loss`), backprop, optimizer step.
  5. Validation: compute per-endpoint metrics (ROC-AUC, MAE, Spearman, etc.) via `utils.metrics.compute_metrics`.
  6. Save best checkpoint by aggregate score (maximize metric; for RMSE/MAE, score = negative value) to `checkpoints/admet/best_model.pt`.

### 3.6 Evaluation

- **Script:** `scripts/evaluate_admet.py`.
- Loads best ADMET checkpoint, runs on test loaders, reports per-endpoint metrics (see `docs/reports/EXPERIMENT_LOG.md` for baseline).

---

## 4. Phase 2: DrugOracle

### 4.1 Purpose

Provide **clinical phase success probabilities** (Phase I, II, III) and **explainability**: ADMET predictions, structural alerts, risk factors, and text recommendations. The Oracle uses the **trained ADMET model** to get 22-dimensional features, then maps them through **cascaded phase predictors**.

### 4.2 Data

- **Clinical dataset:** `data/processed/oracle/clinical_trials.csv` with columns `smiles`, `phase1`, `phase2`, `phase3` (binary or continuous labels). Loaded by `models/oracle/clinical_data.load_clinical_dataset`. (TrialBench / clinical outcomes; see DATA_SOURCES.md.)

### 4.3 Model Architecture

- **CascadedPhasePredictors** (`models/oracle/phase_predictors.py`):
  - **Input:** 22-D vector of ADMET predictions (one per endpoint).
  - **Phase 1:** MLP(in_dim=22) → logit → sigmoid → p1.
  - **Phase 2:** MLP(22 + 1) with p1 as extra input → p2.
  - **Phase 3:** MLP(22 + 2) with p1, p2 as extra inputs → p3.
  - Each branch: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → 1.
- **Overall success:** overall_prob = p1 × p2 × p3.

### 4.4 Training Workflow

- **Script:** `scripts/train_oracle.py`.
- **Process:**
  1. Load ADMET checkpoint (`checkpoints/admet/best_model.pt`) and endpoint config.
  2. Build `OracleDataset`: for each row in clinical CSV, run `predict_smiles(admet_model, smiles, endpoint_task_types)` to get 22-D ADMET vector; targets are phase1, phase2, phase3.
  3. Train `CascadedPhasePredictors` with `OracleTrainer` (MSE or BCE on phase logits), 10 epochs.
  4. Save `checkpoints/oracle/best_model.pt`.

### 4.5 Prediction Workflow (DrugOracle.predict)

1. **Validate SMILES** (RDKit).
2. **ADMET:** SMILES → graph (`smiles_to_graph`) → ADMET model (GNN encoder → attention pool → 22 heads) → 22-D prediction vector; classification endpoints passed through sigmoid.
3. **Phase probs:** 22-D vector → CascadedPhasePredictors → p1, p2, p3 (sigmoid); overall = p1×p2×p3.
4. **Structural alerts:** `models/oracle/structural_alerts.detect_structural_alerts(smiles)` — SMARTS-based patterns loaded from `data/structural_alerts.csv` (or built-in fallback). The CSV can be regenerated from the Hamburg SMARTS dataset (PAINS, Enoch) via `scripts/download_structural_alerts.py`. Sources: Ehrlich & Rarey, J Cheminform 2012; Baell & Holloway, J. Med. Chem. 2010 (PAINS); Enoch et al., SAR QSAR Environ Res 2008 (skin sensitization). Returns list of alert names and atom indices.
5. **Recommendations:** `models/oracle/recommender.generate_recommendations(admet_preds, alerts)` — e.g. hERG > 0.5 → suggest reducing LogP/basic amines; low bioavailability → add polar groups; plus per-alert suggestions.
6. **Risk factors:** e.g. herg/ames/dili > 0.5 → add toxicity risk factor.
7. **Return:** `OraclePrediction` (phase probs, admet_predictions, risk_factors, structural_alerts, recommendations, alert_atoms).

---

## 5. Phase 3: SafeMolGen (Generator)

### 5.1 Purpose

Generate novel SMILES strings. Used in the integrated pipeline to produce candidates that are then scored by the DrugOracle.

### 5.2 Data

- **Primary:** `data/processed/generator/smiles.tsv` (e.g. from `scripts/download_chembl_smiles.py`) — column `canonical_smiles` or `smiles`.
- **Fallback:** Aggregate valid SMILES from `data/admet_group/*/train_val.csv` via `utils/data_utils.aggregate_admet_smiles`.
- **Processing:** `load_and_prepare_smiles()` — validate with RDKit, optional canonicalization, dedup, limit (e.g. 50k).

### 5.3 Tokenizer

- **Location:** `models/generator/tokenizer.SMILESTokenizer`.
- **Tokenization:** Regex-based SMILES tokenization (bracket atoms, %XX, Br/Cl, symbols, digits, etc.); special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`.
- **Encode:** BOS + tokens + EOS, padded to `max_length` (128); vocab built by `fit(smiles_list)`.

### 5.4 Model Architecture

- **TransformerDecoderModel** (`models/generator/transformer.py`):
  - Token embedding → positional encoding (sinusoidal) → Transformer **encoder** stack (causal mask so autoregressive): 6 layers, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1.
  - Output: linear projection to vocab size (next-token logits).
- **Generation:** Autoregressive: start from BOS; at each step take last token logits, apply temperature and top-k filtering, sample (or argmax); stop at EOS or max_length. Option `disallow_special` masks BOS/PAD/UNK in sampling.

### 5.5 Training Workflow

- **Script:** `scripts/train_generator.py`; stages: `pretrain` (default), `rl`.
- **Pretrain:**
  1. Load SMILES (from TSV or ADMET fallback), build tokenizer, `TransformerDecoderModel(vocab_size=tokenizer.vocab_size)`.
  2. `SMILESDataset`: encode SMILES to input/target (shift-by-one); `DataLoader` batch training.
  3. Cross-entropy loss (ignore PAD); Adam; optional `on_epoch_end` to report validity (generate N samples, count valid SMILES).
  4. Save to `checkpoints/generator/` (model.pt, tokenizer.json, config).
- **RL (optional):** `train_rl()` in `models/generator/rl_trainer` — fine-tune with oracle score as reward; requires pretrained checkpoint (e.g. `--stage rl --resume checkpoints/generator`).

### 5.6 Generation API

- `SafeMolGen.generate(n, temperature, top_k, device)` — raw decoding (may include invalid SMILES).
- `SafeMolGen.generate_valid(n, ...)` — repeat until n valid SMILES (validate_smiles).
- `generate_iterative(oracle_score_fn, n_iter, n)` — multiple rounds of generation, score with oracle, keep best.

---

## 6. Phase 4: Integration & Design Loop

### 6.1 Pipeline Class

- **Location:** `models/integrated/pipeline.SafeMolGenDrugOracle`.
- **Loading:** `from_pretrained(generator_path, oracle_path, admet_path, endpoint_names, endpoint_task_types, admet_input_dim, device)` loads SafeMolGen and DrugOracle (ADMET + CascadedPhasePredictors).

### 6.2 Design Molecule Workflow

- **Method:** `design_molecule(target_success, max_iterations, candidates_per_iteration, top_k)`.
- **Algorithm:**
  1. Temperature schedule (e.g. 1.1 → 0.65 over iterations).
  2. For each iteration:
     - Generate `candidates_per_iteration` SMILES via `generator.generate(..., temperature, top_k)`.
     - For each candidate: `evaluate_molecule(smiles)` → `oracle.predict(smiles)` (ADMET + phase probs + alerts + recommendations).
     - Keep only valid predictions; sort by `overall_prob` descending.
     - Record iteration best (smiles + prediction); append to `iteration_history` with improvement tags (e.g. “Phase I: +X%”).
     - Update global best if current iteration best has higher overall_prob.
     - If best_score ≥ target_success, return immediately with `target_achieved=True`.
  3. Return `DesignResult(final_smiles, final_prediction, iteration_history, target_achieved, total_iterations)`.

### 6.3 Other Pipeline Methods

- `evaluate_molecule(smiles)` → single `OraclePrediction` or None.
- `generate_candidates(n, temperature, top_k)` → list of SMILES.
- `compare_molecules(smiles_list)` → list of dicts (smiles, prediction, properties), sorted by overall_prob.
- `save_result(DesignResult, path)` → JSON export.

### 6.4 FastAPI + React App

- **Entry:** Backend `uvicorn backend.main:app` (or `scripts/run_app.py`); frontend `cd frontend && npm run dev` (React at http://localhost:5173, API at http://localhost:8000).
- **Pipeline load:** Lazy load of SafeMolGen + DrugOracle from `checkpoints/generator`, `checkpoints/oracle`, `checkpoints/admet`; endpoints from `config/endpoints.yaml`; `admet_input_dim=11` (must match graph node feature dimension used at ADMET train time).
- **Pages:** Generate (design_molecule with SSE streaming, iteration chart), Analyze (single molecule Oracle dashboard), Compare (multiple molecules), About.
- **Generate page:** User sets target success, max iterations, top_k; on “Generate” calls `POST /api/v1/design/stream` and shows iteration timeline and best molecule + Oracle dashboard.

---

## 7. End-to-End Methodology Summary

| Stage | Data | Model | Output |
|-------|------|--------|--------|
| **Data (Phase 1)** | TDC ADMET Group | — | data/admet_group, data/processed/admet (CSV + .pt graphs) |
| **Train ADMET** | Processed graphs | GNN (GIN) + attention pool + 22 heads | checkpoints/admet/best_model.pt |
| **Data (Phase 2)** | clinical_trials.csv (SMILES, phase1/2/3) | — | data/processed/oracle/ |
| **Train Oracle** | ADMET features per SMILES | CascadedPhasePredictors | checkpoints/oracle/best_model.pt |
| **Data (Phase 3)** | ChEMBL/ADMET SMILES | — | data/processed/generator/smiles.tsv or admet_group |
| **Train Generator** | SMILES corpus | Tokenizer + Transformer decoder | checkpoints/generator/ |
| **Inference** | User / design loop | Generator + Oracle (ADMET + phase + alerts + recs) | DesignResult, OraclePrediction, UI |

---

## 8. Key Design Decisions

- **GNN for ADMET:** Molecules as graphs; GIN encoder for structure-aware embeddings; single multi-task model for all 22 endpoints to share representation and reduce compute.
- **Cascaded phases:** Phase II conditioned on Phase I success, Phase III on I+II, reflecting real clinical dependency.
- **Generator:** Autoregressive transformer on SMILES tokens; pretrain then optional RL with Oracle as reward.
- **Design loop:** Generate many candidates per iteration, score with full Oracle, keep best and iterate with decreasing temperature for exploitation; stop when overall success probability meets target.

---

## 9. File Reference (Key Scripts & Modules)

| Purpose | Path |
|--------|------|
| ADMET data download | `scripts/download_data.py` |
| ADMET preprocessing | `scripts/preprocess_data.py` |
| ADMET training | `scripts/train_admet.py` |
| ADMET evaluation | `scripts/evaluate_admet.py` |
| Oracle training | `scripts/train_oracle.py` |
| Generator training | `scripts/train_generator.py` |
| Graph construction | `utils/chemistry.py` |
| GNN encoder | `models/admet/gnn_encoder.py` |
| ADMET predictor | `models/admet/multi_task_predictor.py` |
| ADMET inference | `models/admet/inference.py` |
| Phase predictors | `models/oracle/phase_predictors.py` |
| DrugOracle | `models/oracle/drug_oracle.py` |
| Structural alerts | `models/oracle/structural_alerts.py` |
| Recommender | `models/oracle/recommender.py` |
| Generator | `models/generator/safemolgen.py`, `transformer.py`, `tokenizer.py` |
| Integrated pipeline | `models/integrated/pipeline.py` |
| App | `app/app.py` |
| Config | `config/config.yaml`, `config/endpoints.yaml` |

This methodology workflow report reflects the implementation as of the current codebase and can be updated as phases (e.g. clinical data integration, RL completion, run_pipeline implementation) are finalized.
