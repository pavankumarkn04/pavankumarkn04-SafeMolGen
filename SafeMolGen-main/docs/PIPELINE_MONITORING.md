# Pipeline run and monitoring

## Full pipeline (run_full_pipeline.sh)

The one-shot script runs these steps in order:

1. Prepare ADMET data from admet_group
2. Preprocess ADMET to .pt
3. Train ADMET
4. Prepare clinical_trials.csv
5. Train Oracle
6. Generator pretrain (30 epochs)
7. Post-pretrain eval (generate_samples)
8. Generator RL (optional, 10 epochs)
9. Run pipeline (design_molecule) -> outputs/design_result.json

## Optional commands (already run if needed)

- `python scripts/download_structural_alerts.py` – refreshes structural alerts (PAINS, etc.); uses built-in if download fails.
- `python scripts/download_chembl_smiles.py` – downloads ChEMBL SMILES to data/processed/generator/smiles.tsv for richer generator data.

## Monitoring a running pipeline

If you started `bash scripts/run_full_pipeline.sh` in the background:

- Check process: `ps aux | grep -E "train_|run_full|run_pipeline" | grep -v grep`
- Generator pretrain (step 6) prints progress like `Pretrain Epoch X/30: ...` and can take ~15 min per epoch (~7.5 h for 30 epochs).
- When step 9 finishes, `outputs/design_result.json` will exist.

## Current run status

A full pipeline run was started. Steps 1–5 completed; step 6 (Generator pretrain, 30 epochs) runs next, then steps 7–9. Let it finish in the same terminal or monitor via the process list above.
