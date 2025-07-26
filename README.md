# 7643 Project – Image Captioning Architecture Evolution

## Overview

This repository contains work developed as part of **Georgia Tech’s CS7643: Deep Learning** course.  
It explores how modifications to an image captioning pipeline affect performance, starting from a baseline **ResNet‑50 + LSTM** encoder-decoder model and progressively integrating more advanced components.

> **Note:** This is an academic showcase project. While it demonstrates various model architectures and experimentation workflows, the codebase was primarily designed for coursework and research exploration rather than production use.

---

## Key Architectures

1. **Baseline** – ResNet‑50 Encoder + LSTM Decoder  
2. **R‑CNN + LSTM** – Uses region proposals instead of global CNN features  
3. **ViT + LSTM** – Vision Transformer encoder with patch embeddings  
4. **ResNet‑50 + Transformer Decoder** – CNN encoder with attention-based Transformer decoder  
5. **CLIP + LSTM (attention adaptation)** – Leverages CLIP’s pretrained embeddings for multimodal feature alignment

---

## Repository Structure & Branches

Each branch corresponds to a specific experimental pipeline:

| Branch Name                | Description                                                               |
|---------------------------|---------------------------------------------------------------------------|
| `main`                    | Central branch with merged updates and results summary.                   |
| `base_model`              | ResNet‑50 encoder + LSTM decoder (baseline).                              |
| `rcnn_no_attention`       | Region proposals from R‑CNN + LSTM decoder (no attention mechanism).      |
| `vit_encoder_lstm_decoder`| ViT encoder with patch embeddings + LSTM decoder.                         |
| `transformer_decoder`     | ResNet‑50 encoder + Transformer-based decoder (cross-attention).          |
| `CLIP-FasterRCNN`         | CLIP embeddings combined with Faster R‑CNN proposals + LSTM decoder.      |

Each branch is self-contained with its own model code, feature extraction pipeline, training scripts, and outputs.

---

## Dataset

The experiments use the **MS COCO 2017** dataset.  
Please refer to the data loading utilities within each branch to correctly structure your dataset files.

---

## Environment Requirements

- Python 3.10.16  
- Latest [PyTorch](https://pytorch.org/) release (tested on 2.x)  
- See `requirements.txt` in individual branches for additional dependencies.

---

## Credits

The baseline model is adapted from [Laurent Veyssier’s Image Captioning repository](https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model).

---

## License

For educational and research purposes only.
