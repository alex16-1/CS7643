# https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
# Credit: Laurent Veyssier for the original code, and inspired derived code for feature-extracted datasets

import os
import os.path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import pickle
from collections import Counter

from torchvision import transforms
import torch
import torch.utils.data as data

import nltk

from pycocotools.coco import COCO


class Vocabulary(object):

    # https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
    # Credit: Laurent Veyssier

    def __init__(
        self,
        vocab_threshold,
        vocab_file="./vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file="../cocoapi/annotations/captions_train2014.json",
        vocab_from_file=False,
    ):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def get_loader(
    transform=None,
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """

    # https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
    # Credit: Laurent Veyssier
    # Adapted for coco2017 with some minor changes

    # We added transform here to not pass argument each time
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    assert mode in ["train", "val"], "mode must be one of 'train' or 'val'."
    if vocab_from_file == False:
        assert (
            mode == "train"
        ), "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == "train":
        if vocab_from_file == True:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join("data", "images", "train2017")
        annotations_file = os.path.join("data", "annotations", "captions_train2017")

    if mode == "val":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join("data", "images", "val2017")
        annotations_file = os.path.join("data", "annotations", "captions_val2017")

    # COCO caption dataset.
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )

    if mode == "train":
        # We are using a custom collate function here
        # To zero pad sequences inferior to max length
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    else:
        # No collate function here because only images
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return data_loader


class CoCoDataset(data.Dataset):
    # https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
    # Credit: Laurent Veyssier
    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )
        self.img_folder = img_folder
        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [
                nltk.tokenize.word_tokenize(
                    str(self.coco.anns[self.ids[index]]["caption"]).lower()
                )
                for index in tqdm(np.arange(len(self.ids)))
            ]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == "train":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train":
            return len(self.ids)
        else:
            return len(self.paths)


# Our custom modifications for extracted features dataset:


class CoCoFeatureDataset(data.Dataset):
    # This is our dataset class. Inspired by:
    # https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
    # Credit: Laurent Veyssier

    def __init__(
        self,
        mode,
        batch_size,
        max_boxes,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        features_folder,
        device,
    ):
        self.device = device
        self.mode = mode
        self.max_boxes = max_boxes
        self.batch_size = batch_size
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )
        self.features_folder = features_folder
        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [
                nltk.tokenize.word_tokenize(
                    str(self.coco.anns[self.ids[index]]["caption"]).lower()
                )
                for index in tqdm(np.arange(len(self.ids)))
            ]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        # obtain features and caption if in training mode
        if self.mode == "train":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]

            features_path = os.path.join(self.features_folder, f"{img_id:012d}.npz")
            features = np.load(features_path)  # n_boxes x feature_dim
            features = torch.tensor(features["x"], dtype=torch.float)
            current_n_boxes = features.shape[0]

            # Truncate boxes if too many, pad if too few
            if current_n_boxes >= self.max_boxes:
                features = features[: self.max_boxes, :]

            else:
                zeros = torch.zeros(
                    self.max_boxes,
                    features.shape[1],
                    dtype=torch.float,
                )

                zeros[: features.shape[0]] = features
                features = zeros

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return features, caption

        # obtain features if in test mode
        else:
            ann_id = self.ids[index]
            img_id = self.coco.anns[ann_id]["image_id"]

            features_path = os.path.join(self.features_folder, f"{img_id:012d}.npz")
            features = np.load(features_path)  # n_boxes x feature_dim
            features = torch.tensor(features["x"], dtype=torch.float)
            current_n_boxes = features.shape[0]

            # Truncate boxes if too many, pad if too few
            if current_n_boxes >= self.max_boxes:
                features = features[: self.max_boxes, :]

            else:
                zeros = torch.zeros(
                    self.max_boxes,
                    features.shape[1],
                    dtype=torch.float,
                )

                zeros[: features.shape[0]] = features
                features = zeros

            return features

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train":
            return len(self.ids)
        else:
            return len(self.paths)


def collate_fn(data):
    # Sort the data having longest captions at start
    data = sorted(data, key=lambda x: len(x[1]), reverse=True)

    # Not touching images
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)

    return images, captions


def get_loader_features(
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    max_boxes=36,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
    device="cuda",
):
    # Inspired from:
    # https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
    # Credit: Laurent Veyssier

    assert mode in ["train", "test"], "mode must be one of 'train' or 'test'."
    if vocab_from_file == False:
        assert (
            mode == "train"
        ), "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == "train":
        if vocab_from_file == True:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        features_folder = "features_train/"
        annotations_file = os.path.join(
            "data", "annotations", "captions_train2017.json"
        )
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        features_folder = "features_val/"
        annotations_file = os.path.join("data", "annotations", "captions_val2017.json")

    # COCO caption dataset.
    dataset = CoCoFeatureDataset(
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        max_boxes=max_boxes,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        features_folder=features_folder,
        device=device,
    )

    if mode == "train":
        # We will use a custom collate function to pad to the longest
        # Caption length
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return data_loader
