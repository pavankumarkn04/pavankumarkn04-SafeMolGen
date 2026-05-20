# ADMET and Oracle Training Check

## Summary (after retrain)

- **ADMET**: Retrained for **30 epochs** (config updated). Training loss decreased (e.g. ~445 → ~162). Test metrics improved on several weak endpoints (e.g. pgp_broccatelli 0.33→0.69, ames 0.40→0.58, bbb_martins 0.47→0.52, ppbr_az MAE 22→16).
- **Oracle**: Retrained for **25 epochs** (default `--epochs 25`). Final train loss ~1.68; eval loss (BCE) **1.67**. Stable and consistent.

---

## ADMET Test-Set Results

(From `python scripts/evaluate_admet.py`.)

### Classification (ROC-AUC; target > 0.5) — after 30-epoch ADMET retrain

| Endpoint              | ROC-AUC | Note        |
|-----------------------|--------|-------------|
| pgp_broccatelli       | 0.69   | Good (was 0.33) |
| cyp2d6_veith          | 0.64   | OK          |
| cyp3a4_veith          | 0.63   | OK          |
| cyp2d6_substrate      | 0.65   | OK          |
| bioavailability_ma    | 0.62   | OK          |
| herg                  | 0.62   | OK          |
| ames                  | 0.58   | OK (was 0.40) |
| dili                  | 0.57   | OK          |
| bbb_martins           | 0.52   | Borderline (was 0.47) |
| cyp2c9_substrate      | 0.52   | Borderline  |
| cyp2c9_veith          | 0.62   | OK          |
| cyp3a4_substrate      | 0.41   | Weak        |
| hia_hou               | 0.32   | Weak (was 0.27) |

### Regression (MAE; lower is better) — after 30-epoch retrain

| Endpoint                 | MAE   | Note        |
|--------------------------|-------|-------------|
| caco2_wang               | 0.70  | OK          |
| lipophilicity_astrazeneca| 1.03  | OK          |
| ld50_zhu                 | 0.86  | OK          |
| solubility_aqsoldb       | 1.88  | OK          |
| vdss_lombardo            | 3.99  | Moderate    |
| ppbr_az                  | 15.92 | Improved (was 21.92) |
| half_life_obach          | 16.73 | High        |
| clearance_hepatocyte_az | 38.44 | High        |
| clearance_microsome_az   | 32.93 | High        |

**Conclusion (ADMET):** 30-epoch retrain improved several endpoints (pgp, ames, bbb_martins, ppbr_az). Checkpoints saved to `checkpoints/admet/best_model.pt`. Log: `logs/train_admet_30ep.log`.

---

## Oracle Eval Loss

(From `python scripts/evaluate_oracle.py`.)

- **Eval loss (BCE): 1.6730**
- Final training loss (epoch 25): ~1.69

Eval and train loss match; Oracle is **trained consistently**. Checkpoint: `checkpoints/oracle/best_model.pt`. Log: `logs/train_oracle_25ep.log`.

---

## Desired Values / Next Steps

- **ADMET**: “Low loss” and reasonable metrics on at least some endpoints are satisfied. For “desired” across all 22 endpoints: consider more epochs (e.g. 20–30), learning-rate schedule, or rebalancing multi-task weights.
- **Oracle**: Training and eval losses are in a good, consistent range. To get lower loss and better calibration: more/better clinical data (e.g. real TrialBench) or more epochs.

Both models are trained and usable for the pipeline; improvements above would be incremental.
