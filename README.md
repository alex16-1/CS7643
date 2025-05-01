# 7643_project

## Environment requirements:

The latest PyTorch version is sufficient for this project. We have tested our scripts with Python 3.10.16 and the latest PyTorch available.

## Credits

The base model from which all branches derivated is based on Laurent Veyssier's repository, https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master.

# Image Captioning Architecture Evolution

This project explores how replacing components of a baseline image captioning model (ResNet-50 + LSTM) with more advanced modules impacts performance. We evaluate and compare five architectures on the MS COCO 2017 dataset.

## Architectures Compared

- **Baseline**: ResNet-50 + LSTM  
- **R-CNN + LSTM**  
- **ViT + LSTM**  
- **ResNet-50 + Transformer Decoder**  
- **CLIP + LSTM (with attention adaptation)**

---

## Repository Structure & Branches

Each branch corresponds to a specific model pipeline:

| Branch Name               | Description                                                             |
|--------------------------|-------------------------------------------------------------------------|
| `main`                   | Central branch. All major updates merged here.                          |
| `base_model`             | ResNet-50 encoder + LSTM decoder.                                       |
| `rcnn_no_attention`      | Region proposals from R-CNN + LSTM decoder without attention.           |
| `vit_encoder_lstm_decoder` | ViT encoder with patch embeddings + LSTM decoder.                      |
| `transformer_decoder`         | ResNet-50 encoder + Transformer-based decoder (cross-attention).        |
| `CLIP-FasterRCNN`        | Comparison using CLIP embeddings and Faster R-CNN-based proposals.      |

Each branch is self-contained with model code, feature extractors, training pipeline, and outputs.

---

## Using the COCO 2017 dataset

Please follow internal data loaders code to place your datasets in the working directory.
