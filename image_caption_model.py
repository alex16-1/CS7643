import torch.nn as nn
from vit_Feature_extractor import *
from lstm_decoder import *

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(ImageCaptioningModel, self).__init__()
        # self.encoder = ViTFeatureExtractor(vit_model)
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, features, captions):
        # features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, features, vocabulary, max_length=20):
        """Generate a caption for an image."""
        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            # features = self.encoder(image.unsqueeze(0))
            sampled_ids = self.decoder.sample(features, max_len=max_length)

        # Convert word indices to words
        sampled_caption = []
        for word_id in sampled_ids[0].detach().cpu().numpy():
            word_id = int(word_id)
            word = vocabulary.itos.get(word_id, "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<SOS>", "<PAD>"]:
                sampled_caption.append(word)

        return ' '.join(sampled_caption)