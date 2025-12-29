# Learning to Communicate Across Modalities

This repository contains the official code for *Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems*, to appear at Evolang 2026.

## Overview
Emergent communication offers insight into how agents develop shared structured representa- tions, yet most research assumes homogeneous modalities or aligned representational spaces, overlooking the perceptual heterogeneity of real-world settings. We study a heterogeneous multi-step binary communication game where agents differ in modality and lack perceptual grounding. Despite perceptual misalignment, multimodal systems converge to class-consistent messages grounded in perceptual input. Unimodal systems communicate more efficiently, using fewer bits and achieving lower classification entropy, while multimodal agents require greater information exchange and exhibit higher uncertainty. Bit perturbation experiments provide strong evidence that meaning is encoded in a distributional rather than compositional manner, as each bit’s contribution depends on its surrounding pattern. Finally, interoperability analy- ses show that systems trained in different perceptual worlds fail to directly communicate, but limited fine-tuning enables successful cross-system communication. This work positions emer- gent communication as a framework for studying how agents adapt and transfer representations across heterogeneous modalities, opening new directions for both theory and experimentation.

## Repository Structure

```
heteroemecom/
└── train/
    ├── blah
    ├── blah2
    └── blah3
```

## Requirements

- Python >= 3.9
- PyTorch
- NumPy
- SciPy
- Matplotlib / Seaborn (for analysis)

Exact dependencies will be specified in `requirements.txt`.

## Running the Code

### Generate Synthetic Data
To train agents in a heterogeneous communication setting:

```bash
python experiments/train.py --config configs/heterogeneous.yaml
```

### Train

To train agents in a heterogeneous communication setting:

```bash
python experiments/train.py --config configs/heterogeneous.yaml
```

### Evaluate

To train agents in a heterogeneous communication setting:

```bash
python experiments/train.py --config configs/heterogeneous.yaml
```

