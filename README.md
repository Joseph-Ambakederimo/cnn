---
title: "Simple CNN demos (numpy + PyTorch)"
author: ""
date: "`
`r Sys.Date()`"
output: html_document
---

# Description

This repository contains two small example implementations of a simple
convolutional neural network for MNIST-style digit classification.

- `cnn.py` — a tiny numpy-based CNN implementation used for learning
  and experimentation (no external DL framework required).
- `cnn_keras.py` — previously a TensorFlow/Keras script; replaced with a
  PyTorch implementation (`cnn_keras.py`) to avoid TensorFlow/Python
  compatibility issues on newer Python versions.

Both scripts include a fallback to synthetic data when the `mnist`
package cannot download the dataset (offline or remote URL issues).

# Files

- `cnn.py`       — minimal numpy CNN implementation (forward/train loops).
- `conv.py`      — convolution layer used by `cnn.py`.
- `maxpool.py`   — max pooling layer for `cnn.py`.
- `softmax.py`   — dense + softmax layer used by `cnn.py`.
- `cnn_keras.py` — PyTorch reimplementation of the original Keras demo.
- `README.rmd`   — this file.

# Quick start (Windows / PowerShell)

1. Recommended Python: 3.10 or 3.11 for best compatibility with common
   ML packages. The lightweight `cnn.py` will run on any recent Python.

2. Run the numpy demo (no additional packages required):

```powershell
python cnn.py
```

3. Run the PyTorch demo (`cnn_keras.py`):

- If PyTorch is not installed the script will print an instruction and
  exit. Install PyTorch (CPU-only) with the (example) command below.

```powershell
# CPU-only (example)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For GPU builds, visit https://pytorch.org/get-started/locally/ and
# select the correct CUDA version for your system.
```

After installing PyTorch:

```powershell
python cnn_keras.py
```

# Notes & Behavior

- The `mnist` package is used to fetch the MNIST files. If the
  environment is offline, or the remote files are unavailable, both
  scripts fall back to randomly generated synthetic images and labels so
  the code can run for demos and tests without network access.

- `cnn.py` is educational and intentionally small. It demonstrates
  forward and (partial) backward passes implemented in plain NumPy.

- `cnn_keras.py` is a compact PyTorch example that mirrors the small
  architecture used in `cnn.py` (Conv -> Pool -> FC). It is aimed at
  users who prefer a modern framework and GPU support.

# Troubleshooting

- If you see import errors for `torch`/`tensorflow`, check your Python
  version and installed packages. TensorFlow frequently lags newer
  Python releases; PyTorch often has wider availability.

- If MNIST download fails with 404/URL errors, the scripts will print a
  warning and continue with synthetic data. To use real MNIST data
  offline you can download the IDX files manually and place them in the
  location expected by the `mnist` package (see the package docs).

# License

Use these examples for learning and experimentation. No license is
included — treat them as permissively reusable for educational purposes.
