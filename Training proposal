# Training AI-ECG with Data

## Overview
Goal: train a 1D-CNN with a tabular branch later to predict CTRCD from **8-lead** ECG waveforms.  
**Model input:** `(B, 8, L)` float32  
**Model output:** `(B, 2)` logits (CTRCD / non-CTRCD)  
**Primary metrics:** AUROC and AUPRC

---

## 1) Data Curation & Splits (No Leakage)
- **Manifest file (CSV):**  
  `patient_id, path_xml, label` where `label ∈ {0,1}`.
- **Split by patient** (not by recording) to prevent information leakage.
- **Stratified** split: Train/Val/Test = **70/15/15**. Each `patient_id` appears in **one** split.
- **De-identification:** store only waveforms + minimal metadata (sampling rate, lead names/order).


---

## 2) Parsing & Preprocessing (Deterministic)
- **Canonical 8-lead order** (document & enforce):  
  `["I","II","V1","V2","V3","V4","V5","V6"]`  
- **Parse XML → tensor** `x_raw ∈ ℝ^{8×L_raw}` with lead order.
- **Resample** to a fixed sampling rate (e.g., **500 Hz**) and **fix length `L`**:  
  - Too long → *center-crop* to `L`  
  - Too short → *zero-pad symmetrically* to `L`
- **Normalization (train-stats only):**
  - Per-lead z-score on the **train** split:  
    `x = (x − μ_train[lead]) / σ_train[lead]`
  - Save stats and reuse unchanged for Val/Test.
- **Missing leads:** either  
  - *(a)* drop the sample, **or**  
  - *(b)* impute zeros **plus a mask channel** (then model in-channels = 9).  


---

## 3) Train-Time Augmentations (Shape-Safe)
Apply **only** on the training split; output must remain `(8, L)`:
- **Gaussian noise:** σ ≈ 0.01–0.03 (relative to normalized scale)
- **Amplitude scaling:** ±10%
- **Time shift:** ≤ 10 samples  
For physiologic coherence, time shifts should be **lead-synchronous** when specified.

---

## 4) Model & Objective
- **Input:** `(B, 8, L)` float32.
- **Output:** `(B, 2)` logits (CTRCD / non-CTRCD).
  - **Loss:** `CrossEntropyLoss` (supports class weights).
- **Class imbalance:** compute **class weights** (CE) or **pos_weight** (BCE) from the **train** split label distribution.

---

## 5) Optimization, Scheduling, Early Stopping
- **Optimizer:** AdamW (start `lr=1e-3`, `weight_decay=1e-4`).
- **Batch size:** 16–64 (depends on `L` and memory). Enable **AMP/mixed precision** on GPU if available.
- **Scheduler:**  
  - `ReduceLROnPlateau` on **Val AUPRC**, *or*  
  - Cosine decay (optionally with warmup).
- **Early stopping:** monitor **Val AUPRC**, patience **10–15** epochs; **restore best** checkpoint.

---

## 6) Reproducibility & Logging
- Fix seeds (NumPy, PyTorch); enable deterministic CuDNN
- Log training/validation curves (loss, AUROC, AUPRC) with TensorBoard or Weights & Biases.
- Save artifacts:
  - `checkpoints/best.pt` – best model (by Val AUPRC)
  - `norm_stats.json` – per-lead train means/standard deviations
  - `config.yaml` – hyperparameters/seeds used for the run
  - `metrics_val.json`, `metrics_test.json` – final metrics

---

## 7) Evaluation & Thresholding
- Select a **decision threshold** on the **validation** split (maximize F1 or Youden’s J).
- Report on the **held-out test** set:
  - **Primary:** AUROC, AUPRC
  - **Secondary:** Accuracy, F1, Sensitivity/Specificity
- If multiple recordings per patient exist, compute **patient-level** outputs (e.g., mean of logits) before thresholding.

---

## 8) Minimal CLI Example
```bash
python ecg_train.py \
  --patient data/manifest.csv \
  --split_seed 42 \
  --leads 8 --hz 500 --length 5000 \
  --batch 32 --epochs 60 \
  --loss ce --class-weights 1.0 2.2 \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 \
  --aug noise,scale,shift \
  --val-metric auprc --early-stop 12 \
  --outdir runs/ctrcd_baseline
```
-- 
## 9) Sanity Checklist (Before Long Runs)
- All tensors seen by the model are finite and shaped (B, 8, L).
- Lead order exactly matches your documented canonical order.
- Normalization uses train-only stats across the whole experiment.
- Augmentations do not change length or desynchronize leads unintentionally.
