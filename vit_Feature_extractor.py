import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor as TransformerEncoder

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_extractor = TransformerEncoder.from_pretrained(model_name)

        # Freeze the ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Prepare images for ViT
        inputs = self.feature_extractor(images=images, return_tensors="pt", do_rescale=False)
        device = next(self.vit.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Extract features
        outputs = self.vit(**inputs)
        # OUTPUT OF VIT torch.Size([32, 768])
        # Return the [CLS] token as image representation
        return outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
