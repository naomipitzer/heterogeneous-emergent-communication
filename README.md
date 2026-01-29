# Learning to Communicate Across Modalities

This repository contains the official code for *Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems*, to appear at Evolang 2026.

## Overview
Emergent communication offers insight into how agents develop shared structured representa- tions, yet most research assumes homogeneous modalities or aligned representational spaces, overlooking the perceptual heterogeneity of real-world settings. We study a heterogeneous multi-step binary communication game where agents differ in modality and lack perceptual grounding. Despite perceptual misalignment, multimodal systems converge to class-consistent messages grounded in perceptual input. Unimodal systems communicate more efficiently, using fewer bits and achieving lower classification entropy, while multimodal agents require greater information exchange and exhibit higher uncertainty. Bit perturbation experiments provide strong evidence that meaning is encoded in a distributional rather than compositional manner, as each bit’s contribution depends on its surrounding pattern. Finally, interoperability analy- ses show that systems trained in different perceptual worlds fail to directly communicate, but limited fine-tuning enables successful cross-system communication. This work positions emer- gent communication as a framework for studying how agents adapt and transfer representations across heterogeneous modalities, opening new directions for both theory and experimentation.

## Repository Structure
This repository includes the codebase for training and evaluating the models, along with sample data and pretrained checkpoints. The repository structure is as follows:

```
heterogeneous_emergent_communication/
  Data/            ← sample synthetic datasets (.npz format)
  DataPrep/        ← synthetic data preparation scripts
  Models/          ← sample model checkpoints and test datasets
  Train/           ← training scripts
  Evaluate/        ← evaluation scripts
```
The project is based on the work of Evtimova et al. (2018), *“Emergent Communication in a Multi-Modal, Multi-Step Referential Game.”* Portions of the training code are adapted from the authors’ original GitHub implementation. To cite their work, use the following BibTeX entry:
```
@misc{evtimova2018emergentcommunicationmultimodalmultistep,
      title={Emergent Communication in a Multi-Modal, Multi-Step Referential Game}, 
      author={Katrina Evtimova and Andrew Drozdov and Douwe Kiela and Kyunghyun Cho},
      year={2018},
      eprint={1705.10369},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1705.10369}, 
}
```

## Requirements

Dependencies are specified in the `requirements.txt` file. To install dependencies, run:
```
pip install -r requirements.txt
```

## Running the Code

### Train
Training parameters are currently specified directly in the train.py file. Adjust these values before running the training script.

```bash
python Train/train.py
```

### Evaluate

To compute message similarities, entropy per conversation timestep, and more:

```bash
python Evaluate/test_models.py
```

To evaluate message semantics:
```bash
python Evaluate/message_analysis.py
```

Code to evaluate interoperability between modalities can be found in Evaluate/evaluate_interoperability.ipynb.

### Generate Data

Details are provided in the [synthetic data documentation](https://github.com/naomipitzer/heterogeneous-emergent-communication/tree/main/DataPrep/synthetic), found in the DataPrep folder.


