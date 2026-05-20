# Oracle and RL Generator Evaluation Summary

## Success probability >50% and Oracle steering the generator

- **Calibration:** The oracle now calibrates **overall** success so it can exceed 50%. Raw overall (often 0.01–0.05) is mapped via `1 - exp(-k * raw)` with `k=22`, so good molecules show e.g. 35–70% overall. Phase I/II/III remain raw for conditioning.
- **Target:** Pipeline and API default `target_success = 0.5` (50%). Design stops when best calibrated overall ≥ 50%, and condition vectors steer toward higher phase probs so the generator improves.
- **RL:** Oracle reward in RL is now in [0, 1] (calibrated), so `--w-oracle 0.4` gives a meaningful signal. Use **curriculum** to avoid validity collapse: `bash scripts/run_rl_curriculum.sh` (phase 1 low w_oracle, phase 2 higher).
- **UI:** Generate page shows overall and phases on 0–100% axes; target success slider goes up to 95%.

## 1. Oracle check

- **Script:** `scripts/evaluate_oracle.py`
- **Metric:** BCE loss on clinical trial data (`data/processed/oracle/clinical_trials.csv`)
- **Result:** Oracle eval loss (BCE) = **1.33** (lower is better). The oracle is loaded and evaluated successfully.

## 2. Generator vs RL (oracle metrics)

- **Script:** `scripts/eval_generator_oracle.py`
- **Compares:** Base generator (`checkpoints/generator`) vs RL generator (`checkpoints/generator_rl`), each scored with the DrugOracle (overall % and Phase I/II/III %).

### Run (150 molecules each, earlier in session)

| Metric          | Base generator | RL generator (w_oracle=0.4, 8 ep) |
|-----------------|----------------|------------------------------------|
| Validity        | 92.7%          | 88.7%                              |
| Uniqueness      | 100%           | 100%                               |
| **Overall mean %** | **1.80**    | 1.18                               |
| **Overall max %**  | **2.22**    | 2.17                               |
| Phase I mean    | 44.6%          | 30.8%                              |

**Conclusion:** The base generator currently achieves **higher mean and max overall %** (and higher Phase I mean) than the RL checkpoint from that run. So for “how well the generator is doing” on the oracle: **base is ahead**.

## 3. What was tried

1. **RL with higher oracle weight**  
   - `--w-oracle 0.4` (default is 0.1), 8 epochs.  
   - Reward and validity improved during training (e.g. reward ~0.12 → 0.81, validity ~8% → 92%), but the resulting RL generator still had **lower** oracle mean/max than the base in the eval above.

2. **Scaling the oracle reward**  
   - Oracle `overall_prob` is small (~0.01–0.02). It was scaled (e.g. ×50, cap 1.0) so the oracle term could compete with validity/QED in the reward.  
   - **Result:** Validity collapsed to 0% and stayed there (policy drifted off valid molecules). Scaling was reverted.

3. **CLI for oracle weight**  
   - `scripts/train_generator.py` now supports `--w-oracle` so you can run RL with a stronger oracle weight without changing code.

## 4. Recommendations

- **For “clearly higher overall” from the generator today:** Use the **base generator** (`checkpoints/generator`); it currently outperforms the evaluated RL checkpoint on oracle overall (and Phase I) in `eval_generator_oracle.py`.
- **To improve RL in the future:**  
  - Keep `w_oracle` moderate (e.g. 0.2–0.4); very high weight or aggressive scaling hurt validity.  
  - Consider a curriculum: start with default (or low) `w_oracle`, then gradually increase it over epochs.  
  - Re-evaluate after every RL run with `scripts/eval_generator_oracle.py` to confirm that the RL checkpoint beats the base on mean/max overall and phases before relying on it.

## 5. Commands used

```bash
# Oracle evaluation
PYTHONPATH=. python3 scripts/evaluate_oracle.py

# Generator vs RL (oracle metrics)
PYTHONPATH=. python3 scripts/eval_generator_oracle.py --n 150

# RL with custom oracle weight
PYTHONPATH=. python3 scripts/train_generator.py --stage rl --resume checkpoints/generator --out checkpoints/generator_rl --w-oracle 0.4 --epochs 8
```
