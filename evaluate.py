import torch
import net
from data_loader import *
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction().method1


def evaluate_valid(pth_file, sanity_check=False):
    valid_loader = get_loader(
        mode="val",
        batch_size=1,
        vocab_threshold=5,
        num_workers=12,
    )

    vocab = valid_loader.dataset.vocab
    vocab_size = len(vocab)

    device = "cuda"
    checkpoint = torch.load(pth_file)

    embed_dim = 128
    hidden_dim = 256

    encoder = net.EncoderCNN(embed_dim)
    decoder = net.DecoderRNN(embed_dim, hidden_dim, vocab_size)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    bleu_score_1 = 0.0
    bleu_score_4 = 0.0
    total = 0

    coco = COCO("data/annotations/captions_val2017.json")

    for i, data in enumerate(valid_loader):

        if i % 100 == 0:
            print(f"Processing sample {i}")

        img, caption = data
        img = img.to(device)

        with torch.no_grad():
            visual_features = encoder(img)
            word_idx = decoder.sample(visual_features)

            pred_caption = [vocab.idx2word[i] for i in word_idx]

            temp = pred_caption

            pred_caption = []

            for word in temp:
                if word not in ["<start>", "<end>", "<pad>"]:
                    pred_caption.append(word)

            # print(pred_caption)

        # Retrieve the corresponding image ID to get ground truth captions
        img_id = valid_loader.dataset.coco.anns[valid_loader.dataset.ids[i]]["image_id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        references = [nltk.word_tokenize(ann["caption"].lower()) for ann in anns]

        if sanity_check == True:
            print(f"Predicted caption: {pred_caption}")
            print(f"References: {references}")
            return

        # We are using a smoothing function
        # To be more forigiving for shorter sentences
        # https://github.com/nltk/nltk/issues/1554

        bleu1 = sentence_bleu(
            references,
            pred_caption,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoothing,
        )
        bleu4 = sentence_bleu(
            references,
            pred_caption,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )

        # Compute BLEU score
        bleu = sentence_bleu(references, pred_caption)

        bleu_score_1 += bleu1
        bleu_score_4 += bleu4
        total += 1

        if i % 100 == 0:
            print(
                f"Current bleu1: {bleu_score_1 / total} bleu4: {bleu_score_4 / total}"
            )

    print(f"Avg. valid bleu1: {bleu_score_1 / total}")
    print(f"Avg. valid bleu4: {bleu_score_4 / total}")


def plot_loss(filename):
    plt.figure(figsize=(5, 3))

    checkpoint = torch.load(filename)

    loss_history_train = checkpoint["loss_history_train"]
    loss_history_valid = checkpoint["loss_history_valid"]

    epochs = range(10)

    plt.plot(epochs, loss_history_train, label=f"Train loss")
    plt.plot(epochs, loss_history_valid, label=f"Valid loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.savefig("basemodel_learningcurve.png", dpi=300)


def main():
    pth_file = "base_model.pth"
    evaluate_valid(pth_file)
    plot_loss(pth_file)


if __name__ == "__main__":
    main()
