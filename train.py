# Very inspired from:
# https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
# Credit: Laurent Veyssier

from data_loader import *
from net import *
import nltk


def train(
    encoder,
    decoder,
    train_loader,
    valid_loader,
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
    loss_history_train = []
    loss_history_valid = []

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

            # Compute loss
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            if batch % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], batch [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            current_loss += loss.item()

            if sanity_check:  # Test on only one batch
                break

        # Valid evaluation

        with torch.no_grad():

            encoder.eval()
            decoder.eval()
            current_loss_valid = 0.0

            for batch, (images, captions) in enumerate(valid_loader):

                images, captions = images.to(device), captions.to(device)
                encoder_features = encoder(images)
                outputs = decoder(encoder_features, captions)

                current_loss_valid += criterion(
                    outputs.view(-1, vocab_size), captions.view(-1)
                ).item()

            loss_history_train.append(current_loss / len(train_loader))
            loss_history_valid.append(current_loss_valid / len(valid_loader))

        print(f"Epoch [{epoch}/{num_epochs}] Val Loss: {loss_history_valid[-1]:.4f}")

    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history_train": loss_history_train,
            "loss_history_valid": loss_history_valid,
        },
        f"base_model.pth",
    )

    return loss_history_train, loss_history_valid


def main():
    print("##### TRAINING BASE MODEL #####")
    nltk.download("punkt_tab")

    print("##### LOADING TRAIN")

    train_loader = get_loader(
        mode="train",
        batch_size=512,
        vocab_threshold=5,
        num_workers=12,
    )
    print("##### LOADING VAL")
    valid_loader = get_loader(
        mode="val",
        batch_size=512,
        vocab_threshold=5,
        num_workers=12,
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

    embed_dim = 128
    hidden_dim = 256
    device = "cuda"

    encoder = EncoderCNN(embed_dim)
    decoder = DecoderRNN(embed_dim, hidden_dim, vocab_size)

    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.vocab("<PAD>"))

    criterion = criterion.to(device)
    params = (
        list(encoder.embed.parameters())
        + list(encoder.bn.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.Adam(
        params, lr=0.001, betas=(0.9, 0.999), eps=1e-08
    )  # "Show & Tell" paper uses SGD with no momentum

    train(
        encoder,
        decoder,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        10,
        len(vocab),
        "cuda",
    )


if __name__ == "__main__":
    main()
