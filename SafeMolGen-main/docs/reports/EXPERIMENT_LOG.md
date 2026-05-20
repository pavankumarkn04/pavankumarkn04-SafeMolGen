# Experiment Log

Use this log to record training runs, metrics, and configs.

| Date | Phase | Experiment | Dataset | Key Params | Metrics | Notes |
|------|-------|------------|---------|-----------|---------|-------|
| 2026-02-01 | Phase 1 | ADMET baseline (22 endpoints) | TDC ADMET Group | epochs=10, batch=64, device=cpu | See detailed metrics below | Torch 2.2.2 |
| 2026-02-01 | Phase 3 | SafeMolGen pretrain | ChEMBL chemreps (2,854,815 SMILES) | epochs=5, batch=64, device=cpu | Final loss=1.0253 | Output: checkpoints/generator/ |

## Phase 1 Baseline Metrics (Test)
- caco2_wang: rmse=0.7570, mae=0.6011, spearman=-0.3092
- hia_hou: roc_auc=0.6720, auprc=0.8722
- pgp_broccatelli: roc_auc=0.6796, auprc=0.5953
- bioavailability_ma: roc_auc=0.6334, auprc=0.8426
- lipophilicity_astrazeneca: rmse=1.2134, mae=1.0040, spearman=0.1822
- solubility_aqsoldb: rmse=2.3675, mae=1.8456, spearman=0.3080
- bbb_martins: roc_auc=0.4661, auprc=0.8307
- ppbr_az: rmse=24.8108, mae=23.0539, spearman=0.0287
- vdss_lombardo: rmse=5.7523, mae=4.0551, spearman=0.3248
- cyp2c9_veith: roc_auc=0.3781, auprc=0.2558
- cyp2d6_veith: roc_auc=0.6418, auprc=0.2690
- cyp3a4_veith: roc_auc=0.3936, auprc=0.3791
- cyp2c9_substrate: roc_auc=0.4771, auprc=0.3101
- cyp2d6_substrate: roc_auc=0.3658, auprc=0.2567
- cyp3a4_substrate: roc_auc=0.5824, auprc=0.6592
- half_life_obach: rmse=21.9543, mae=12.5215, spearman=0.1294
- clearance_hepatocyte_az: rmse=48.8531, mae=35.9162, spearman=-0.0785
- clearance_microsome_az: rmse=43.1097, mae=30.7591, spearman=-0.2870
- ld50_zhu: rmse=1.1481, mae=0.8286, spearman=0.1208
- herg: roc_auc=0.6163, auprc=0.8084
- ames: roc_auc=0.5783, auprc=0.6894
- dili: roc_auc=0.5572, auprc=0.5475
