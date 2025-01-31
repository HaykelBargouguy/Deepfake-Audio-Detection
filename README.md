# Deepfake Audio Detection: ASVspoof Anti-Spoofing Framework

## 📌 Project Overview
Automatic Speaker Verification (ASV) systems are highly vulnerable to spoofing attacks, including Voice Conversion (VC), Text-To-Speech (TTS), and replay attacks. This repository provides a robust, dual-pipeline framework (Machine Learning & Deep Learning) designed to act as a countermeasure (CM) gatekeeper. It classifies incoming audio utterances as either **Bona Fide** (genuine human speech) or **Spoof** (synthetically generated or replayed speech).

This project is configured natively for the **ASVspoof 2019** dataset, handling both **Logical Access (LA)** (synthetic/converted speech) and **Physical Access (PA)** (replay attacks in physical spaces).

---

## 🎧 Inputs & Feature Extraction (The Acoustic Front-End)
Raw audio waveforms contain vast amounts of redundant information. To detect the subtle, microscopic artifacts left behind by vocoders or recording devices, we transform the 16000 Hz audio into highly discriminative feature spaces.

### 1. Voice Activity Detection (VAD)
Before feature extraction, raw audio is processed through a custom **Voice Activity Detection** function. By calculating the rolling mean amplitude of the waveform and applying a strict threshold, we filter out absolute silence. This ensures that our downstream models are learning from actual speech characteristics, not background room noise.

### 2. Hand-Crafted Acoustic Features
* **CQCC (Constant Q Cepstral Coefficients):** Unlike traditional features that use linear or Mel scales, CQCC uses geometrically spaced frequency bins. This provides higher frequency resolution at lower frequencies and higher temporal resolution at high frequencies. It is exceptionally good at detecting the high-frequency phase anomalies often left by TTS vocoders.
* **MFCC & LFCC (Mel/Linear Frequency Cepstral Coefficients):** Standard acoustic representations capturing the vocal tract's physical characteristics. We extract the static coefficients along with their **Delta** and **Delta-Delta** derivatives to capture temporal dynamics (how the speech changes over time).

### 3. Deep Acoustic Embeddings (Transformer-based)
We utilize Self-Supervised Learning (SSL) models to extract rich, contextual acoustic representations:
* **WavLM & UniSpeech-SAT:** These transformer models, pre-trained on thousands of hours of speech, extract high-dimensional embedding vectors that capture advanced phonetic and acoustic contexts missed by traditional signal processing.

---

## 🧠 The Deep Learning Approach (End-to-End)
Our primary pipeline is an end-to-end deep neural network that treats the 2D acoustic features (Time x Frequency) similarly to a single-channel image, looking for visual-spatial anomalies that indicate spoofing.

### 1. The ResNet Backbone
We utilize a deep Residual Network (ResNet) composed of `PreActBlocks` (Pre-Activation ResNet). By applying Batch Normalization and ReLU *before* the convolutions, the network optimizes gradient flow. The ResNet acts as a localized feature extractor, capturing structural inconsistencies in the spectrograms caused by synthetic generation.

### 2. The Innovation: Custom Self-Attention Mechanism
Not all parts of an audio utterance are equally useful for detecting spoofs. Artifacts often hide in specific phonetic transitions, fricatives, or unnatural silences.

After the ResNet extracts spatial features, we flatten the frequency dimension and apply a **Custom Self-Attention Mechanism**:
* **Temporal Weighting:** The module learns a weight vector that acts as a query, assigning attention scores to different frames (time-steps) using batch matrix multiplication. It "pays attention" only to the frames most likely to contain spoofing artifacts.
* **Statistical Aggregation:** Instead of simply averaging the weighted frames, our module calculates both the **Weighted Mean** and the **Weighted Standard Deviation**. By concatenating these, the network receives a much richer representation of the utterance's acoustic distribution.
* **Noise Regularization:** We inject a microscopic amount of Gaussian noise (1e-5) directly into the standard deviation computation. This acts as a novel regularizer, preventing the attention mechanism from overfitting to specific artifacts in the training set and drastically improving generalization to unseen, "zero-day" spoofing attacks.

### 3. OCSoftmax Loss (One-Class Softmax)
Traditional Cross-Entropy Loss attempts to draw a boundary between two distinct classes. However, "Spoofs" are not a single, cohesive class—they are an infinite variety of unseen algorithms.
We use **OCSoftmax**, which treats anti-spoofing as a one-class anomaly detection problem. It forces the network to map all "Bona Fide" samples into a highly compact, tight sphere in the latent space, while pushing all "Spoof" samples as far away from this sphere as possible by applying specific margins (r_real and r_fake).

---

## 🤖 The Machine Learning Baseline
To validate the efficacy of the deep learning model, we include a classical ML pipeline. Acoustic features and deep embeddings are pooled into flattened 1D vectors.
* **Support Vector Machines (SVM):** Utilizing RBF kernels with balanced class weights to handle dataset imbalances.
* **LightGBM:** A highly optimized gradient-boosting decision tree approach, utilizing fractional feature bagging to prevent overfitting on specific acoustic bins.

---

## 📊 Evaluation Metrics
Because this system acts as a gatekeeper for an Automatic Speaker Verification (ASV) system, standard accuracy is an insufficient metric. We evaluate using:

1. **EER (Equal Error Rate):** The threshold where the False Acceptance Rate (FAR - accepting a spoof) equals the False Rejection Rate (FRR - rejecting a real human). Lower is better.
2. **t-DCF (Tandem Detection Cost Function):** A specialized metric for ASVspoof. It mathematically combines the error rates of the *Countermeasure* (our model) with the error rates of the downstream *Biometric System*, providing a single cost score representing the real-world risk of the combined system.

---

## 🚀 Quick Start & Usage

### 1. Environment Setup
We recommend using `uv` for lightning-fast dependency resolution.
```bash
pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Configuration & Data Preparation
All hyperparameters and data paths are centralized in `configs/config.py`. To fetch the ASVspoof dataset and extract offline features:
```bash
bash scripts/download_LA.bash
bash scripts/prepare_data.bash
```

### 3. Training
**Train the Deep Learning Model (ResNet + Custom Attention):**
```bash
python -m src.dl_train
```

**Train the Machine Learning Baselines (SVM / LightGBM):**
```bash
python -m src.ml_train
```