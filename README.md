# ASVSpoof Anti-Spoofing Detection Framework

This project targets Automatic Speaker Verification (ASV) anti-spoofing detection, specifically geared towards the ASVspoof datasets. It provides a robust, dual-pipeline framework featuring both Deep Learning end-to-end approaches and Classical Machine Learning baselines.

## Architecture Description

### Deep Learning Pipeline
The core deep learning approach utilizes a **ResNet Architecture integrated with a Custom Self-Attention Mechanism**.
1. **Feature Extraction:** Raw audio is padded and normalized before being transformed into specific frequency representations (MFCC, LFCC, or CQCC).
2. **ResNet Backbone:** A modified ResNet utilizing `PreActBlocks` processes the 2D spatial features to capture local structural anomalies inherent to synthetic speech.
3. **Custom Self-Attention Mechanism:** After the convolutional stages, the frequency axis is flattened. A custom self-attention mechanism computes learnable weights for the temporal sequence, dynamically identifying and aggregating the most informative frames into a compact, utterance-level representation. This module computes both the mean and standard deviation.
4. **Classification:** The vector is passed through fully connected layers, optimized using custom margins like **OCSoftmax**.

### Machine Learning Baselines
For comparative robustness, the project includes classical ML techniques that operate on offline-extracted, flattened feature representations:
* **Support Vector Machines (SVM)**
* **LightGBM** (Gradient Boosting)

## Dataset Context
The framework is pre-configured to handle the **ASVspoof 2019** dataset configurations, supporting both Logical Access (LA) and Physical Access (PA).

## Training and Validation Pipeline
1. **Config:** All hyperparameters are centralized in `configs/config.py`.
2. **Deep Learning:** Execute `python -m src.dl_train`.
3. **Machine Learning:** Execute `python -m src.ml_train` after extracting offline features via the scripts in `src/features/`.