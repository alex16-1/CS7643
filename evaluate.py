import torch
import net
from data_loader import *
import argparse
from args import get_args
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu


def evaluate_valid(pth_file, args):
    valid_loader = get_loader(
        mode="val",
        batch_size=1,
        vocab_threshold=5,
        vocab_file="vocab.pkl",
        vocab_from_file=True,
        num_workers=12,
        shuffle=False,
    )

    vocab = valid_loader.dataset.vocab
    vocab_size = len(vocab)

    device = args.device
    checkpoint = torch.load(pth_file, map_location=device)

    embed_dim = checkpoint["embed_dim"]
    num_layers = checkpoint["num_layers"]
    num_heads = checkpoint["num_heads"]

    encoder = net.ResNetEncoderPatch(embed_dim)
    decoder = net.CaptionDecoder(embed_dim, vocab_size, num_layers=num_layers, nhead=num_heads, max_len=50)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    bleu_score = 0.0
    total = 0

    coco = COCO("data/annotations/captions_val2017.json")

    for i, data in enumerate(valid_loader):

        if i % 100 == 0:
            print(f"Processing sample {i}")

        original_img, preprocessed = data
        preprocessed = preprocessed.to(device)

        with torch.no_grad():
            visual_features = encoder(preprocessed)
            word_idx = decoder.sample(visual_features)
            pred_caption = [vocab.idx2word[i] for i in word_idx]
            pred_caption = [
                word
                for word in pred_caption
                if word not in ["<start>", "<end>", "<pad>"]
            ]

        # Retrieve the corresponding image ID to get ground truth captions
        img_id = coco.getImgIds(imgIds=[])[i]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        references = [nltk.word_tokenize(ann["caption"].lower()) for ann in anns]

        if args.visualize == True and i < 5:
            output_dir = f'pics/{pth_file}'
            visualize_predictions(img_id, original_img, pred_caption, references, output_dir)

        if args.sanity_check == True:
            print(f"Predicted caption: {pred_caption}")
            print(f"References: {references}")
            return

        bleu = sentence_bleu(references, pred_caption)

        bleu_score += bleu
        total += 1

    print(f"Avg. valid bleu: {bleu_score / total}")


def visualize_predictions(id, img, pred_caption, ref_captions, output_dir):
    to_pil = ToPILImage()
    plt.figure(figsize=(6, 4))

    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:
            img = to_pil(img[0].permute(2, 0, 1).cpu())
        else:
            img = to_pil(img.permute(2, 0, 1).cpu())

    plt.imshow(img)
    plt.axis('off')

    pred_caption = ' '.join(pred_caption)
    ref_caption = ' '.join(ref_captions[0])

    title = f"Pred:\n{pred_caption}\n\nRef:\n{ref_caption}"
    plt.title(title, fontsize=8)
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, str(id))
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the image captioning model")
    get_args(parser)
    args = parser.parse_args()

    pth_file = f"./caption_model_{args.num_heads}_{args.num_layers}_{args.embed_size}.pth"
    evaluate_valid(pth_file, args)


if __name__ == "__main__":
    main()
