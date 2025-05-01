# Very inspired from:
# https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
# Credit: Laurent Veyssier

from data_loader import *
from net import *
import nltk
import argparse
from args import get_args
import os


def train(
    encoder,
    decoder,
    train_loader,
    criterion,
    optimizer,
    num_epochs,
    vocab_size,
    device,
    sanity_check=False,
):
    """
    Train the model
    """
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        current_loss = 0.0

        for batch, (images, captions) in enumerate(train_loader):

            images, captions = images.to(device), captions.to(device)

            encoder.zero_grad()
            decoder.zero_grad()

            encoder_features = encoder(images)
            outputs = decoder(encoder_features, captions)

            # offset outputs and targets: outputs[:, 0, :] should be compared with captions[:, 1]
            loss = criterion(
                outputs[:, :-1, :].reshape(-1, vocab_size),
                captions[:, 1:].reshape(-1)
            )

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            if batch % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], batch [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            current_loss += loss.item()

            if sanity_check:  # Test on only one batch
                break

        loss_history.append(current_loss / len(train_loader))

    return loss_history


def initialize_model(args, vocab_size, device):
    model_path = f"caption_model_{args.num_heads}_{args.num_layers}_{args.embed_size}.pth"

    encoder = ResNetEncoderPatch(args.embed_size)
    decoder = CaptionDecoder(args.embed_size, vocab_size, num_layers=args.num_layers, nhead=args.num_heads)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    params = (
        # list(encoder.embed.parameters())
        # + list(encoder.bn.parameters())
            list(encoder.conv.parameters())
            + list(decoder.parameters())
    )
    print(sum(p.numel() for p in decoder.parameters()) / 1e6, 'M parameters')

    optimizer = torch.optim.Adam(
        params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08
    )

    # load pretrained model if requested and file exists
    if args.from_pretrained and os.path.isfile(model_path):
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # sanity check that args match
        assert checkpoint["num_heads"] == args.num_heads
        assert checkpoint["num_layers"] == args.num_layers
        assert checkpoint["embed_dim"] == args.embed_size

    elif args.from_pretrained:
        raise FileNotFoundError(f"Pretrained model not found at {model_path}")

    return encoder, decoder, optimizer


def main():
    parser = argparse.ArgumentParser(description="Train the image captioning model")
    get_args(parser)
    args = parser.parse_args()

    nltk.download("punkt_tab")

    train_loader = get_loader(
        mode="train",
        batch_size=args.batch_size,
        vocab_threshold=5,
        vocab_file="vocab.pkl",
        vocab_from_file=True,
        num_workers=12,
        shuffle=True,
    )

    vocab_size = len(train_loader.dataset.vocab)
    vocab = train_loader.dataset.vocab

    print(
        "Index of <PAD>  :",
        vocab("<PAD>") if "<PAD>" in vocab.word2idx else "Not defined",
    )
    print("Index of <START>:", vocab("<start>"))
    print("Index of <END>  :", vocab("<end>"))
    print("Index of <UNK>  :", vocab("<unk>"))

    device = args.device
    encoder, decoder, optimizer = initialize_model(args, vocab_size, device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.vocab("<PAD>"))
    criterion = criterion.to(device)

    print(f"Training model on {args.device}...")
    loss_history = train(
        encoder,
        decoder,
        train_loader,
        criterion,
        optimizer,
        num_epochs=args.num_epochs,
        vocab_size=vocab_size,
        device=device,
        sanity_check=args.sanity_check
    )
    if args.sanity_check:
        print('Mock training completed successfully')
    else:
        print('Training completed successfully')

    model_name = f"caption_model_{args.num_heads}_{args.num_layers}_{args.embed_size}.pth"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "embed_dim": args.embed_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        },
        model_name,
    )

    print(f"Model and training loss saved to '{model_name}'")


if __name__ == "__main__":
    main()
