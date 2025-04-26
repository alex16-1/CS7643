# https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/blob/master/3_Inference.ipynb
# Credit: Laurent Veyssier
# Adapted further for our use case

import torch
import net
from data_loader import *

from nltk.translate.bleu_score import sentence_bleu


def evaluate_valid(pth_file, sanity_check=False):
    checkpoint = torch.load(pth_file)

    valid_loader = get_loader_features(
        mode="test",
        batch_size=1,
        vocab_threshold=5,
        vocab_file="vocab.pkl",
        vocab_from_file=True,
        num_workers=12,
    )

    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    vocab = valid_loader.dataset.vocab
    vocab_size = len(vocab)

    device = "cuda"
    checkpoint = torch.load(pth_file)

    decoder = net.DecoderRNN(2048, embed_dim, hidden_dim, vocab_size)

    decoder = decoder.to(device)

    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    decoder.eval()

    bleu_score = 0.0
    total = 0

    coco = COCO("data/annotations/captions_val2017.json")

    for i, (features, img_id) in enumerate(valid_loader):

        if i % 100 == 0:
            print(f"Processing sample {i}")

        features = features.to(device)

        # Same preprocessing as in train: Mean of non zero boxes

        # Mean non zero boxes
        mask = (
            features.abs().sum(dim=2) > 0
        )  # Sum across feature dim, is dim batches x max_boxes
        box_counts = mask.sum(dim=1)
        box_counts[box_counts == 0] = 1  # In rare case where we have 0 boxes

        features = features.sum(dim=1) / box_counts.unsqueeze(1)

        with torch.no_grad():  # https://discuss.pytorch.org/t/with-torch-no-grad/130146

            word_idx = decoder.sample(features)

            pred_caption = [vocab.idx2word[i] for i in word_idx]

            temp = pred_caption

            pred_caption = []

            for word in temp:
                if word not in ["<start>", "<end>", "<pad>"]:
                    pred_caption.append(word)

            # print(pred_caption)

        # Source for API: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        # Retrieve the corresponding image ID to get ground truth captions
        ann_ids = coco.getAnnIds(
            imgIds=img_id.item()
        )  # Accessing item because it is a tensor from the loader
        anns = coco.loadAnns(ann_ids)

        references = [nltk.word_tokenize(ann["caption"].lower()) for ann in anns]

        if sanity_check == True:
            print(f"Predicted caption: {pred_caption}")
            print(f"References: {references}")
            return

        # print(references)

        bleu = sentence_bleu(references, pred_caption)

        bleu_score += bleu
        total += 1

    print(f"Avg. valid bleu: {bleu_score / total}")


def main():
    pth_file = f"caption_model_rcnn_lr_0.001_epochs_10_batch_size_512_max_boxes_36_embed_dim_128_hidden_dim_256.pth"
    evaluate_valid(pth_file, sanity_check=False)


if __name__ == "__main__":
    main()
