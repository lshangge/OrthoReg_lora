# Understanding and Enforcing Weight Disentanglement in Task Arithmetic — LoRA-ATT

[CVPR 2026] Official code for the **LoRA-ATT** and **LoRA-ATT + OrthoReg** baselines from the paper **"Understanding and Enforcing Weight Disentanglement in Task Arithmetic"**.

[[Paper](https://arxiv.org/abs/2604.17078)] &nbsp; [[OrthoReg repo](https://github.com/RL-MIND/OrthoReg)] &nbsp; [[Checkpoints](#-checkpoints)] &nbsp; [[Datasets](#-datasets)]

---

## Overview

This repository implements the **LoRA-ATT** fine-tuning paradigm: LoRA adapters are applied exclusively to the attention projection layers (QKV and output) of the ViT backbone, keeping all other weights frozen. After fine-tuning, the low-rank adapters are merged back into the base model as a dense task vector for use in task arithmetic.

We provide two modes:

| `--finetuning-mode` | Description |
|---|---|
| `loraatt` | LoRA-ATT baseline (attention-only LoRA fine-tuning) |
| `loraatt_ortho` | LoRA-ATT + OrthoReg (with orthogonality regularization on LoRA delta_W) |

> **Why a separate repo?** OpenCLIP's ViT fuses Q, K, V into a single `in_proj_weight` tensor. Patching this for LoRA requires monkey-patching the attention forward pass, which adds complexity unsuitable for the main [OrthoReg repository](https://github.com/RL-MIND/OrthoReg).

---

## The OrthoReg Loss on LoRA-ATT

The OrthoReg loss is applied to the equivalent weight update implied by each LoRA module:

$$\Delta W = \frac{\alpha}{r} \cdot B A$$

$$\mathcal{L}_{\text{ortho}} = \sum_l \left\| (\Delta W^{(l)})^\top \Delta W^{(l)} - I \right\|_F^2$$

The total training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{ortho}}$$

---

## Installation

```sh
conda env create -f environment.yml
conda activate tta_peft
```

Add the project root to `PYTHONPATH`:

```sh
cd orthoreg_lora
export PYTHONPATH="$PYTHONPATH:$PWD"
```

---

## Datasets

We evaluate on 8 image classification benchmarks: **Cars · DTD · EuroSAT · GTSRB · MNIST · RESISC45 · SUN397 · SVHN**

For dataset download and preparation, follow the instructions in the [TTA repository](https://github.com/gortizji/tangent_task_arithmetic#datasets).

> 📥 **Dataset Download:** `[TODO: cloud storage link]`

Set the root path via `--data-location /path/to/datasets/`.

---

## Quick Start

All scripts are run from the `orthoreg_lora/` directory.

### Step 0 — Generate Zero-Shot Accuracies

Before evaluating task addition/negation, run the zero-shot baseline to produce `zeroshot_accuracies.json`. This requires that standard zeroshot checkpoints exist (shared with the main OrthoReg repo or generated via `eval_single_task --finetuning-mode none`):

```bash
python src/eval_single_task.py \
    --model ViT-B-32 \
    --finetuning-mode none \
    --data-location /path/to/datasets/
```

### Step 1 — Fine-tune

```bash
python src/finetune.py \
    --model ViT-B-32 \
    --finetuning-mode loraatt \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --lr 1e-4 \
    --data-location /path/to/datasets/
```

To enable OrthoReg:

```bash
python src/finetune.py \
    --model ViT-B-32 \
    --finetuning-mode loraatt_ortho \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --ortho-lambda 1.0 \
    --lr 1e-4 \
    --data-location /path/to/datasets/
```

Checkpoints are saved to:
- `checkpoints_{seed}/loraatt_{lr}_{model}/{dataset}Val/loraatt_finetuned.pt`
- `checkpoints_{seed}/loraatt_ortho_{lr}_lambda{lambda}_{model}/{dataset}Val/loraatt_ortho_finetuned.pt`

### Step 2 — Evaluate Single-Task Accuracy

```bash
python src/eval_single_task.py \
    --model ViT-B-32 \
    --finetuning-mode loraatt \
    --lr 1e-4 \
    --data-location /path/to/datasets/
```

### Step 3 — Evaluate Task Addition

```bash
python src/eval_task_addition.py \
    --model ViT-B-32 \
    --finetuning-mode loraatt \
    --lr 1e-4 \
    --data-location /path/to/datasets/
```

### Step 4 — Evaluate Task Negation

```bash
python src/eval_task_negation.py \
    --model ViT-B-32 \
    --finetuning-mode loraatt \
    --lr 1e-4 \
    --data-location /path/to/datasets/
```

---

## Key Arguments

| Argument | Default | Description |
|---|:---:|---|
| `--model` | `ViT-B-32` | CLIP model architecture |
| `--finetuning-mode` | — | `loraatt` or `loraatt_ortho` |
| `--lora-rank` | `8` | LoRA rank $r$ |
| `--lora-alpha` | `8.0` | LoRA scaling $\alpha$ (scaling = $\alpha / r$) |
| `--ortho-lambda` | `0.0` | OrthoReg strength $\lambda$; set `0` for baseline |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `1993` | Random seed |
| `--world-size` | `1` | Number of GPUs (DDP) |
| `--data-location` | — | Dataset root directory |
| `--batch-size` | `128` | Batch size per GPU |

---

## Checkpoints

> 📥 **Checkpoint Download:** `[TODO: cloud storage link]`

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026orthoreg,
  title     = {Understanding and Enforcing Weight Disentanglement in Task Arithmetic},
  author    = {Liu, Shangge and Yin, Yuehan and Wang, Lei and Fan, Qi and
               Shi, Yinghuan and Li, Wenbin and Gao, Yang and Tao, Dacheng},
  booktitle = {CVPR},
  year      = {2026}
}
```

---

## Acknowledgements

This codebase is built on top of [Task Arithmetic](https://github.com/mlfoundations/task_vectors), [Tangent Task Arithmetic](https://github.com/gortizji/tangent_task_arithmetic), and [Attention-Only Fine-tuning](https://github.com/kyrie-23/linear_task_arithmetic). We thank the authors for releasing their code.
