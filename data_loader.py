from collections import Counter
import json
import numpy as np
import nltk
import os
from PIL import Image
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define paths to COCO dataset
coco_path = 'data/data'
train_image_dir = f'{coco_path}/images/train2017'
val_image_dir = f'{coco_path}/images/val2017'
annotations_dir = f'{coco_path}/annotations'


# class COCODataset(Dataset):
#     def __init__(self, root_dir, annotation_file, transform=None, max_len=50):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.max_len = max_len

#         # Load annotations
#         with open(annotation_file, 'r') as f:
#             self.annotations = json.load(f)

#         # Process annotations to get image-caption pairs
#         self.img_ids = []
#         self.captions = []
#         self.image_paths = []

#         for ann in self.annotations['annotations']:
#             image_id = ann['image_id']
#             caption = ann['caption']

#             # Find image path
#             image_path = os.path.join(
#                 self.root_dir,
#                 f'{image_id:012d}.jpg'
#             )

#             if os.path.exists(image_path):
#                 self.img_ids.append(image_id)
#                 self.captions.append(caption)
#                 self.image_paths.append(image_path)

#         # Build vocabulary (will be implemented later)
#         self.vocab = Vocabulary()
#         self.build_vocabulary()

#     def build_vocabulary(self):
#        # Collect all captions
#         all_captions = [caption for caption in self.captions]

#         # Build vocabulary
#         self.vocab.build_vocabulary(all_captions)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         # Process caption (tokenization, etc.)

#         caption = self.captions[idx]
#         # Convert caption to numerical form
#         numerical_caption = [self.vocab.stoi["<SOS>"]]
#         numerical_caption += self.vocab.numericalize(caption)
#         numerical_caption.append(self.vocab.stoi["<EOS>"])

#         # Pad to max_len
#         if len(numerical_caption) < self.max_len:
#             numerical_caption += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(numerical_caption))
#         else:
#             numerical_caption = numerical_caption[:self.max_len]



#         return image, torch.tensor(numerical_caption)


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, caption_list):
        frequencies = Counter()
        idx = 4
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        for caption in caption_list:
            # Tokenize and count words
            for word in nltk.tokenize.word_tokenize(caption.lower()):
                frequencies[word] += 1

                # Add word to vocabulary if it meets threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, max_len=50,
                 use_precomputed=False, feature_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.use_precomputed = use_precomputed
        self.feature_dir = feature_dir

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Process annotations
        self.img_ids = []
        self.captions = []
        self.image_paths = []

        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            image_file = f'{image_id:012d}.jpg'
            image_path = os.path.join(self.root_dir, image_file)

            if os.path.exists(image_path) or use_precomputed:
                self.img_ids.append(image_id)
                self.captions.append(caption)
                self.image_paths.append(image_file)

        # Build vocabulary
        self.vocab = Vocabulary()
        self.build_vocabulary()

    def build_vocabulary(self):
        all_captions = [caption for caption in self.captions]
        self.vocab.build_vocabulary(all_captions)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file = self.image_paths[idx]
        caption = self.captions[idx]
        image_id = self.img_ids[idx]

        # Load feature or image
        if self.use_precomputed:
            feature_path = os.path.join(self.feature_dir, f'{image_id}.npy')
            image = torch.tensor(np.load(feature_path)).float()
        else:
            image_path = os.path.join(self.root_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

        # Process caption
        numerical_caption = [self.vocab.stoi["<SOS>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<EOS>"])

        # Pad or truncate
        if len(numerical_caption) < self.max_len:
            numerical_caption += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(numerical_caption))
        else:
            numerical_caption = numerical_caption[:self.max_len]

        return image, torch.tensor(numerical_caption), image_id
