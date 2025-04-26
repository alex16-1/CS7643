import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_loader import COCODataset, Vocabulary
from vit_Feature_extractor import ViTFeatureExtractor
from lstm_decoder import LSTMDecoder
from image_caption_model import ImageCaptioningModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os

# Define paths to COCO dataset
coco_path = 'data/data'
train_image_dir = f'{coco_path}/images/train2017'
val_image_dir = f'{coco_path}/images/val2017'
annotations_dir = f'{coco_path}/annotations'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and data loaders
train_dataset = COCODataset(
    root_dir=train_image_dir,
    annotation_file=f'{annotations_dir}/captions_train2017.json',
    transform=None,
    use_precomputed=True,
    feature_dir="./features/train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Initialize model
embed_size = 768
hidden_size = 512
vocab_size = len(train_dataset.vocab)

model = ImageCaptioningModel(
    embed_size=embed_size,
    hidden_size=hidden_size,
    vocab_size=vocab_size
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) #0.001

val_dataset = COCODataset(
    root_dir=val_image_dir,
    annotation_file=f'{annotations_dir}/captions_val2017.json',
    transform=None,
    use_precomputed=True,
    feature_dir="./features/val"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1):
    print("Training...")

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0

        for i, (images, captions, _) in enumerate(train_loader):
            # if i >= 2:
            #     break

            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                captions.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        torch.save(model.state_dict(), 'vit_lstm_caption_model.pth')

    print("Training Complete.")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='b', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    # Save the plot
    plt.savefig('training_loss_curve.png')  # Or use any path like 'plots/loss.png'

    # Optional: Show the plot interactively
    plt.show()
    return epoch_losses


def evaluate_model(model, data_loader, vocabulary, device):
    print("Evaluating...")
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for images, captions, _ in data_loader:
            images = images.to(device)

            # Generate captions
            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)
                predicted_caption = model.caption_image(image, vocabulary)
                actual_caption = ' '.join([vocabulary.itos[idx.item()] for idx in captions[i]
                                         if idx.item() not in [0, 1, 2]])  # Exclude special tokens

                all_predictions.append(predicted_caption.split())
                all_references.append([actual_caption.split()])

    # Calculate BLEU-4 score
    smooth = SmoothingFunction().method1
    bleu4 = corpus_bleu(all_references, all_predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    print("Evation Complete.")
    print(f'BLEU-4 Score: {bleu4:.4f}')

    return bleu4


def evaluate_over_epochs(model, val_loader, vocabulary, device, num_epochs):
    bleu_scores = []

    for epoch in range(num_epochs):
        print(f'\n--- Evaluation for Epoch {epoch+1} ---')
        score = evaluate_model(model, val_loader, vocabulary, device)
        bleu_scores.append(score)

    # Plot BLEU score
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), bleu_scores, marker='x', color='g', label='BLEU-4 Score')
    plt.title('BLEU-4 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4')
    plt.grid(True)
    plt.legend()
    plt.show()

    return bleu_scores


def evaluate_and_print_examples(model, data_loader, vocabulary, device, num_examples=5):
    model.eval()
    count = 0

    with torch.no_grad():
        for features, captions, _ in data_loader:
            features = features.to(device)
            captions = captions.to(device)

            # Only run on the first few examples
            for i in range(features.size(0)):
                feature = features[i].unsqueeze(0)  # shape: [1, 768]
                predicted_caption = model.caption_image(feature, vocabulary)

                # Convert true caption to words (remove special tokens)
                true_caption_ids = captions[i].tolist()
                true_caption = []
                for word_id in true_caption_ids:
                    word = vocabulary.itos.get(int(word_id), "<UNK>")
                    if word in ["<SOS>", "<EOS>", "<PAD>"]:
                        continue
                    true_caption.append(word)

                print(f"\nExample {count + 1}")
                print("Predicted:", predicted_caption)
                print("Reference:", ' '.join(true_caption))

                count += 1
                if count >= num_examples:
                    return


########################################################## Train model ###################################################################

# train_model(model, train_loader, criterion, optimizer, device, num_epochs=100)

############################################################ VALIDATION ##################################################################
# Load the saved model weights into the model
model.load_state_dict(torch.load('vit_lstm_caption_model.pth'))
evaluate_and_print_examples(model, val_loader, val_dataset.vocab, device, 100)