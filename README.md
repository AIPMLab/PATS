# PATS: Prompt-Adaptive Test-Time Segmentation

Official code for the paper "PATS: Prompt-Adaptive Test-Time Segmentation".

PATS is a label-free, plug-and-play wrapper for SAM-based test-time adaptation (TTA) in
medical image segmentation. It combines (i) **PAL**, a prompt-adaptive loop that re-derives
bounding-box candidates from the model's own predictions and re-ranks them by predicted IoU,
and (ii) **E-SBCT**, an edge-aware self-adaptive Bezier input transform. All experiments use a
shared-box, multi-seed protocol: identical pre-generated perturbed boxes and identical seeds for
every method, with paired comparisons.

## 1. Environment

Python >= 3.10, CUDA GPU (tested on a single NVIDIA L40, torch 2.5.1 + cu121):

```bash
pip install -r requirements.txt
```

## 2. Data and checkpoints

| Item | Expected path |
| --- | --- |
| BraTS-SSA T2W / T2F slices | `data/BraTS_SSA_t2w_2D/`, `data/BraTS_SSA_t2f_2D/` |
| BraTS-PED T2W / T2F slices | `data/BraTS_PED_t2w_2D/`, `data/BraTS_PED_t2f_2D/` |
| Prostate MRI (HDF5) | `data/Prostate/` (cases under `data/Prostate/data/*.h5`) |
| BUSI ultrasound | `data/Dataset_BUSI/Dataset_BUSI_with_GT/` |
| SAM ViT-B weights | `checkpoints/sam_vit_b_01ec64.pth` (from the [SAM release](https://github.com/facebookresearch/segment-anything#model-checkpoints)) |
| MedSAM weights (optional) | `checkpoints/medsam_vit_b.pth` (from the [MedSAM release](https://github.com/bowang-lab/MedSAM)) |

## 3. Shared perturbed boxes (evaluation protocol)

Every method is evaluated on the *same* perturbed boxes, pre-generated per
(dataset, offset, seed):

```bash
python protocol/gen_boxes.py --offsets 0.25 --seeds 1337 1338 1339
```

This writes `protocol/boxes/<dataset>_off<offset>_seed<seed>.csv`
(e.g. `protocol/boxes/BraTS_SSA_t2w_2D_off25_seed1337.csv`).

## 4. Running experiments

Common arguments for every script:
`--cfg configs/config_brats.py --prompt box --gpu_ids 0 --dataset <DS> --seed <SEED>
--box_csv protocol/boxes/<DS>_off25_seed<SEED>.csv --out_dir output/runs/<name>/<DS>`

Datasets `<DS>`: `BraTS_SSA_t2w_2D`, `BraTS_SSA_t2f_2D`, `BraTS_PED_t2w_2D`,
`BraTS_PED_t2f_2D`, `Prostate`, `BUSI`. Main results use seeds 1337/1338/1339.

```bash
# Full PATS (E-SBCT + PAL) on the SAM-TTA backbone
python adaptation_token.py --cfg configs/config_brats.py --prompt box --gpu_ids 0 \
  --dataset BraTS_SSA_t2w_2D --seed 1337 \
  --box_csv protocol/boxes/BraTS_SSA_t2w_2D_off25_seed1337.csv \
  --out_dir output/runs/pats_s1337/BraTS_SSA_t2w_2D \
  --loss_box selected --pal_select argmax --skip_rule fullbox,pred_empty

# TTA baselines (same shared boxes): CoTTA / GraTa / TENT / TACT
python adaptation_cotta.py  ... --out_dir output/runs/cotta_s1337/<DS>
python adaptation_grata.py  ... --out_dir output/runs/grata_s1337/<DS>
python adaptation_tent.py   ... --out_dir output/runs/tent_s1337/<DS>
python adaptation_tact.py   ... --out_dir output/runs/tact_s1337/<DS>

# Baseline + PATS wrapper
python adaptation_cotta_add.py ... --out_dir output/runs/cotta_pal_s1337/<DS>
python adaptation_grata_add.py ... --out_dir output/runs/grata_pal_s1337/<DS>
python adaptation_tent_add.py  ... --out_dir output/runs/tent_pats_s1337/<DS>
python adaptation_tact_add.py  ... --out_dir output/runs/tact_pats_s1337/<DS>
```

### Ablations (Tables 4 and 5)

```bash
# SAM-TTA baseline; E-SBCT only
python adaptation_baseline.py --transform B     ... --out_dir output/runs/b_baseline_s1337/<DS>
python adaptation_baseline.py --transform esbct ... --out_dir output/runs/b_esbct_s1337/<DS>

# PAL only (B), and single E-SBCT components on top of PAL
python adaptation_token_ablation.py --ablation baseline    ... --out_dir output/runs/pal_B_s1337/<DS>
python adaptation_token_ablation.py --ablation sobel       ... --out_dir output/runs/pal_L_s1337/<DS>
python adaptation_token_ablation.py --ablation directional ... --out_dir output/runs/pal_D_s1337/<DS>
python adaptation_token_ablation.py --ablation structure   ... --out_dir output/runs/pal_T_s1337/<DS>
```

### Hyperparameter sweeps (LoRA rank, learning rate)

```bash
python adaptation_token_lorar.py --lora_rank 3 ... --out_dir output/runs/pats_r3_s1337/<DS>
python adaptation_token_lr.py   --lr 1e-03    ... --out_dir output/runs/pats_lr1e_03_s1337/<DS>
```

### Frozen segmenter diagnostics (no parameter update)

```bash
python adaptation_source.py     ... --out_dir output/runs/sam_s1337/<DS>        # frozen SAM
python adaptation_source_pal.py ... --out_dir output/runs/sam_pal_s1337/<DS>    # frozen SAM + PAL
# MedSAM variants: add --cfg configs/config_medsam_full.py
python adaptation_medsam.py     ... --out_dir output/runs/medsam_s1337/<DS>
python adaptation_source_pal.py --cfg configs/config_medsam_full.py ... --out_dir output/runs/medsam_pal_s1337/<DS>
```

### Batch execution

`protocol/run_queue.py` runs a JSONL job list serially, waits for free GPU memory before
each job, and skips jobs whose result CSV already exists:

```bash
JR_STATUS=protocol/status.json python protocol/run_queue.py my_jobs.jsonl
```

Each line: `{"id": "...", "cmd": ["python", "adaptation_token.py", ...], "expect_csv": "output/runs/..."}`.

## 5. Aggregation

After runs complete, collect Dice/HD95 (paper protocol: empty-GT slices excluded, empty
predictions score Dice 0, HD95 capped at 50 px) and patient-level paired tests:

```bash
python protocol/aggregate.py
```

This scans `output/runs/*/<DS>/<DS>-box-test-results.csv` and writes
`protocol/summary_runs.csv` (per run and dataset) plus Wilcoxon summaries.

## 6. Repository layout

```
adaptation_token.py           PATS (full wrapper) on SAM-TTA
adaptation_token_ablation.py  PAL-only and E-SBCT component ablations
adaptation_token_lorar.py     LoRA-rank sweep (--lora_rank)
adaptation_token_lr.py        learning-rate sweep (--lr)
adaptation_baseline.py        SAM-TTA baseline (--transform B | esbct)
adaptation_{cotta,grata,tent,tact}[_add].py  TTA backbones without/with PATS
adaptation_medsam.py          frozen MedSAM
model.py losses.py sam_lora.py      shared model, losses, LoRA
configs/                      experiment configs
datasets/                     BraTS / Prostate / BUSI loaders (shared-box CSV)
utils/                        metrics, transforms, Bezier modules
segment_anything/             vendored SAM (Meta AI, Apache-2.0)
protocol/                     box generation, job queue, result aggregation
```

## Acknowledgements

Built on [Segment Anything](https://github.com/facebookresearch/segment-anything) and
[MedSAM](https://github.com/bowang-lab/MedSAM). TTA backbones follow the original
SAM-TTA, CoTTA, GraTa, and TACT papers; the TACT and SAM-TTA keeps the official update rule,
loss, and hyperparameters.

