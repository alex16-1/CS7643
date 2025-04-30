# Parts inspired from train pipeline from:
# https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
# Credit: Laurent Veyssier

from data_loader import *
from net import *
import nltk


def train_rcnn_no_attention(
    n_epochs=10, lr=0.001, embed_dim=128, hidden_dim=256, sanity_check=False
):

    nltk.download("punkt_tab")
    batch_size = 512
    max_boxes = 36
    device = "cuda"

    train_loader = get_loader_features(
        mode="train",
        batch_size=batch_size,
        vocab_threshold=5,
        max_boxes=max_boxes,
        vocab_file="vocab.pkl",
    )

    valid_loader = get_loader_features(
        mode="val",
        batch_size=batch_size,
        vocab_threshold=5,
        max_boxes=max_boxes,
        vocab_file="vocab.pkl",
    )

    vocab = train_loader.dataset.vocab
    vocab_length = len(vocab)

    # Sanity check
    print(
        "Index of <PAD>  :",
        vocab("<PAD>") if "<PAD>" in vocab.word2idx else "Not defined",
    )
    print("Index of <START>:", vocab("<start>"))
    print("Index of <END>  :", vocab("<end>"))
    print("Index of <UNK>  :", vocab("<unk>"))

    decoder = DecoderRNN(2048, embed_dim, hidden_dim, vocab_length)

    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = criterion.to(device)

    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-08
    )  # "Show & Tell" paper uses SGD with no momentum

    loss_history_train = []
    loss_history_valid = []

    for epoch in range(n_epochs):

        current_loss_train = 0.0

        for batch, (features, captions) in enumerate(train_loader):
            decoder.train()

            features, captions = features.to(device), captions.to(device)

            optimizer.zero_grad()

            # Features are batches x max_boxes x feature dim

            # Mean non zero boxes
            mask = (
                features.abs().sum(dim=2) > 0
            )  # Sum across feature dim, is dim batches x max_boxes
            box_counts = mask.sum(dim=1)
            box_counts[box_counts == 0] = 1  # In rare case where we have 0 boxes

            features = features.sum(dim=1) / box_counts.unsqueeze(1)

            # Forward pass
            outputs = decoder(features, captions)

            # Flatten for criterion
            outputs = outputs.reshape(-1, vocab_length)
            captions = captions.reshape(-1)

            loss = criterion(outputs, captions)

            loss.backward()

            optimizer.step()

            if batch % 100 == 0:
                print(
                    f"Epoch [{epoch}/{n_epochs}], batch [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            current_loss_train += loss.item()

            if sanity_check == True:
                break

        # Valid evaluation

        with torch.no_grad():

            decoder.eval()
            current_loss_valid = 0.0

            for batch, (features, captions) in enumerate(valid_loader):

                features, captions = features.to(device), captions.to(device)

                # Features are batches x max_boxes x feature dim

                # Mean non zero boxes
                mask = (
                    features.abs().sum(dim=2) > 0
                )  # Sum across feature dim, is dim batches x max_boxes
                box_counts = mask.sum(dim=1)
                box_counts[box_counts == 0] = 1  # In rare case where we have 0 boxes

                features = features.sum(dim=1) / box_counts.unsqueeze(1)

                # Forward pass
                outputs = decoder(features, captions)

                # Flatten for criterion
                outputs = outputs.reshape(-1, vocab_length)
                captions = captions.reshape(-1)

                current_loss_valid += criterion(outputs, captions).item()

        loss_history_train.append(current_loss_train / len(train_loader))
        loss_history_valid.append(current_loss_valid / len(valid_loader))

        print(
            f"Epoch [{epoch}/{n_epochs}] Val Loss: {current_loss_valid / len(valid_loader)}"
        )

    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": lr,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "max_boxes": max_boxes,
            "loss_history_train": loss_history_train,
            "loss_history_valid": loss_history_valid,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
        },
        f"caption_model_rcnn_default_params.pth",
    )


def main():
    # Embed / hidden dim tests
    # print(
    #     "train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=256, hidden_dim=256)"
    # )
    # train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=256, hidden_dim=256)

    # print(
    #     "train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=128, hidden_dim=128)"
    # )
    # train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=128, hidden_dim=128)

    # print(
    #     "train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=64, hidden_dim=128)"
    # )
    # train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=64, hidden_dim=128)

    # print(
    #     "train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=128, hidden_dim=64)"
    # )
    # train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=128, hidden_dim=64)

    # # Playing with lr
    # print(
    #     "train_rcnn_no_attention(n_epochs=10, lr=0.0001, embed_dim=128, hidden_dim=256)"
    # )
    # train_rcnn_no_attention(n_epochs=10, lr=0.0001, embed_dim=128, hidden_dim=256)

    # print(
    #     "train_rcnn_no_attention(n_epochs=10, lr=0.01, embed_dim=128, hidden_dim=256)"
    # )
    # train_rcnn_no_attention(n_epochs=10, lr=0.01, embed_dim=128, hidden_dim=256)

    # train_rcnn_no_attention(n_epochs=10, lr=0.001, embed_dim=128, hidden_dim=256)

    # Default parameters model
    train_rcnn_no_attention()


if __name__ == "__main__":
    main()
