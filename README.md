# LASA: A Weak Supervision Method for Open-Vocabulary Scene Sketch Semantic Segmentation

## Overview

This repository accompanies our paper:
**"LASA: A Weak Supervision Method for Open-Vocabulary Scene Sketch Semantic Segmentation."**

It contains a cleaned open-source version of the core LASA code for training and evaluating scene sketch semantic segmentation models.

## Repository Status

This repository is under active cleanup for public release.

## Directory Structure

```text
LASA/
  configs/
    config.py
    train_fscoco.yaml
    test_fscoco.yaml
  models/
    modified_model.py
    prompt_extractor.py
  template/
    openai_template.txt
  utils/
    core.py
    losses.py
  dataset.py
  train.py
  test.py
  requirements.txt
  README.md
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The CLIP dependency is installed from the official OpenAI CLIP GitHub repository, so internet access is required during installation.

## Data Layout

The training code expects an FS-COCO-style layout:

```text
DATA_ROOT/
  sketches/
    ...
  text/
    ...
```

The test code expects:

```text
TEST_ROOT/
  images/
  captions/
  vector_sketches/
  classes/
  all_classes.json
```

## Training

Prepare a YAML config, then run:

```bash
python train.py --config configs/train_fscoco.yaml --dataset.root /path/to/fscoco-seg/train
```

Checkpoints are saved under the configured `output_dir`.

## Evaluation

Run:

```bash
python test.py --config configs/test_fscoco.yaml --test.root /path/to/fscoco-seg/test --checkpoint_path /path/to/model.pth
```

The script reports pixel accuracy, stroke accuracy, mIoU, and average forward time.
