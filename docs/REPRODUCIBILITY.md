# Reproducibility Guide

This document summarizes how to reproduce the main PRP-DQN-MIPSO experiment.

## Environment

The project was developed with:

- Windows 11 64-bit
- Python 3.8 or later
- PyTorch with CUDA or CPU execution
- `numpy`, `pandas`, `openpyxl`, `psutil`, and `tqdm`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Experiment

Run the complete single-run workflow:

```bash
python "Core Algorithm Implementations/PRP-DQN/code.py"
```

The script performs path grouping, sample generation, PRP-DQN training, MI-PSO optimization, and Excel result export.

## Auxiliary Scripts

Path grouping:

```bash
python "Algorithm 1pathgrouping.py"
```

Sample selection:

```bash
python "Algorithm 2sample_selection.py"
```

Test data generation:

```bash
python "Generate test data.py"
```

## Inputs and Outputs

Input path samples are stored in:

```text
path_samples/
Core Algorithm Implementations/PRP-DQN/path_samples/
```

Generated results are stored in:

```text
results/
Core Algorithm Implementations/PRP-DQN/*.csv
```

## Reproducibility Notes

The target function uses random simulation values during execution. For exact deterministic reproduction, set fixed seeds for Python, NumPy, and PyTorch before running the main scripts.

Suggested seed block:

```python
import random
import numpy as np
import torch

random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)
```

## Hardware Notes

GPU acceleration is automatically used when CUDA is available. CPU execution is supported but may be slower for training.
