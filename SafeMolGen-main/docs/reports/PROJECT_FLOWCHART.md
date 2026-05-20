# SafeMolGen-DrugOracle — Project Flowchart (Top-Down)

A single top-down view of how the project works: training first, then the design loop at runtime.

## Get a copy-pasteable image

1. **Open `flowchart.html`** in this folder in a browser (double-click or drag into Chrome/Firefox/Safari).
2. Wait for the diagram to render, then either:
   - **Right-click the diagram → "Save image as…"** (if your browser supports it), or
   - **Screenshot:** Mac `Cmd+Shift+4`, Windows `Win+Shift+S`, then crop the flowchart.
3. Paste the saved image into slides or docs.

Alternatively, use [Mermaid Live Editor](https://mermaid.live): paste the Mermaid code from below and export as PNG/SVG.

---

## Full flowchart (Mermaid)

Copy the block below into any Mermaid-supported viewer (GitHub, GitLab, VS Code, Notion, etc.) to render the diagram.

```mermaid
flowchart TB
    Start([Start])

    %% ========== TRAINING ==========
    Start --> P1
    subgraph P1["Phase 1 — ADMET"]
        A1[Download TDC ADMET data]
        A2[Preprocess: SMILES → graphs]
        A3[Train GNN + 22 heads]
        A4[(checkpoints/admet)]
        A1 --> A2 --> A3 --> A4
    end

    P1 --> P2
    subgraph P2["Phase 2 — Oracle"]
        B1[Load ADMET checkpoint]
        B2[Load clinical trials CSV]
        B3[Train Phase I / II / III predictors]
        B4[(checkpoints/oracle)]
        B1 --> B2 --> B3 --> B4
    end

    P2 --> P3
    subgraph P3["Phase 3 — Generator"]
        C1[Load ChEMBL or ADMET SMILES]
        C2[Pretrain transformer]
        C3[(checkpoints/generator)]
        C4{RL?}
        C5[RL fine-tune with Oracle reward]
        C6[(checkpoints/generator_rl)]
        C1 --> C2 --> C3 --> C4
        C4 -->|Yes| C5 --> C6
        C4 -->|No| Deploy
    end

    P3 --> Deploy[Load pipeline: Generator + Oracle + ADMET]

    %% ========== DESIGN LOOP ==========
    Deploy --> Loop
    subgraph Loop["Design loop (each run)"]
        direction TB
        L0[User: target success, max iterations, filters]
        L1[Generate N SMILES]
        L2{Property / scaffold filters?}
        L3[Filter candidates]
        L2 -->|Yes| L3 --> L4
        L2 -->|No| L4
        L4[Score each with Oracle: ADMET → Phase I/II/III]
        L5[Pick best by overall probability]
        L6{Pass safety?}
        L7[Store Oracle feedback for next iter]
        L6 -->|No, feedback on| L7 --> L8
        L6 -->|Yes or feedback off| L8
        L8{Best ≥ target or max iter?}
        L8 -->|No| L1
        L8 -->|Yes| Done
        L0 --> L1
        L4 --> L5 --> L6
    end

    Done([Return best molecule + history])
```

---

## Simplified one-page flowchart (top-down)

Same flow, fewer boxes — good for slides.

```mermaid
flowchart TB
    Start([Start]) --> Phase1[Phase 1: Train ADMET GNN on TDC data]
    Phase1 --> Phase2[Phase 2: Train Oracle on clinical data using ADMET]
    Phase2 --> Phase3[Phase 3: Pretrain Generator on SMILES; optional RL]
    Phase3 --> Load[Load Generator + Oracle + ADMET]
    Load --> Gen[Generate N candidate SMILES]
    Gen --> Score[Score each with Oracle]
    Score --> Best[Pick best by overall probability]
    Best --> Check{Target reached or max iterations?}
    Check -->|No| Feedback[Optionally use Oracle feedback next iteration]
    Feedback --> Gen
    Check -->|Yes| End([Return best molecule])
```

---

## Visual summary (ASCII, top-down)

```
                    ┌─────────────────────────────────────┐
                    │            TRAINING                  │
                    └─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
   ┌─────────┐                  ┌─────────┐                  ┌─────────┐
   │ Phase 1 │                  │ Phase 2 │                  │ Phase 3 │
   │  ADMET  │ ──► checkpoint ──►│ Oracle  │ ──► checkpoint ──►│  Gen    │
   │  GNN    │                  │ Phase   │                  │ Transf. │
   └─────────┘                  │ I/II/III│                  └─────────┘
        ▲                             │                             │
        │                             │                             │
   TDC data                    Clinical CSV                  ChEMBL SMILES
                    ┌─────────────────────────────────────┐
                    │         INFERENCE / DESIGN            │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Load Generator + Oracle + ADMET    │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   │
            ┌───────────────┐                            │
            │ Generate N    │                            │
            │ SMILES        │                            │
            └───────┬───────┘                            │
                    ▼                                    │
            ┌───────────────┐                            │
            │ Oracle:       │                            │
            │ ADMET → Phase │                            │
            │ I / II / III  │                            │
            └───────┬───────┘                            │
                    ▼                                    │
            ┌───────────────┐                            │
            │ Best ≥ target │── No ──────────────────────┘
            │ or max iter?  │
            └───────┬───────┘
                    │ Yes
                    ▼
            ┌───────────────┐
            │ Return best   │
            │ molecule      │
            └───────────────┘
```

---

---

## Project architecture flowchart

Strict top-down: one tier (Pipeline) calls two components below (Generator, Oracle). Return flow: Generator → SMILES to Pipeline; Oracle → phase probs, alerts, recs to Pipeline.

```mermaid
flowchart TB
    P["Pipeline · SafeMolGenDrugOracle<br/>Design loop, filters, Oracle feedback"]
    G["Generator · SafeMolGen<br/>Tokenizer → Transformer"]
    O["Oracle · DrugOracle<br/>Graph → GNN → 22-D → Phase I/II/III, Alerts, Recs"]
    P --> G
    P --> O
```

| Tier | Component | Role |
|------|-----------|------|
| **Top** | Pipeline | Runs the design loop; calls Generator (returns SMILES) and Oracle (returns phase probs, alerts, recs); applies filters and optional condition to Generator. |
| **Below** | Generator | Tokenizer + Transformer; produces SMILES list. |
| **Below** | Oracle | For each SMILES: graph → GNN → 22-D → Phase I/II/III, plus structural alerts and recommender. |

**Data flow (inside Oracle):** SMILES → Graph → GNN → Pool → 22 heads → 22-D → Phase I/II/III, Alerts, Recs.

---

## Legend

| Shape / term | Meaning |
|--------------|---------|
| **Phase 1** | Train one GNN model that predicts 22 ADMET properties from a molecular graph. |
| **Phase 2** | Train a small model that maps those 22 numbers to Phase I / II / III success probabilities. |
| **Phase 3** | Train (and optionally RL fine-tune) a transformer that generates SMILES. |
| **Design loop** | At runtime: generate candidates → score with Oracle → pick best → repeat until target or max iterations. |
| **Oracle feedback** | If the best molecule fails safety, the next iteration can use a condition and filters derived from the Oracle (e.g. avoid certain substructures). |
| **Architecture** | Pipeline owns Generator and Oracle; Oracle uses ADMET internally; RDKit turns SMILES into graphs for ADMET. |