import os
import json
import random
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

images_folder = 'train2017'
annotations_file = 'annotations/captions_train2017.json'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open(annotations_file, 'r') as f:
    annotations = json.load(f)

image_captions = {}
for ann in annotations['annotations']:
    image_id = ann['image_id']
    caption = ann['caption']
    if image_id in image_captions:
        image_captions[image_id].append(caption)
    else:
        image_captions[image_id] = [caption]

image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
image_ids = [img_id for img_id in image_captions if len(image_captions[img_id]) >= 5]

num_images = 100000
selected_image_ids = random.sample(image_ids, num_images)

top1_count = 0
top5_count = 0
total_rank = 0
total_mrr = 0.0
total_cosine = 0.0

for image_id in tqdm(selected_image_ids, desc="Serious CLIP evaluation"):
    image_path = os.path.join(images_folder, image_id_to_filename[image_id])
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    true_captions = image_captions[image_id][:5]

    distractor_captions = []
    while len(distractor_captions) < 5:
        distractor_id = random.choice(image_ids)
        if distractor_id != image_id:
            distractor_captions.append(random.choice(image_captions[distractor_id]))

    all_captions = true_captions + distractor_captions
    text_tokens = clip.tokenize(all_captions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    similarities = (image_features @ text_features.T).squeeze(0)
    sorted_indices = similarities.argsort(descending=True)
    sorted_captions = [all_captions[i] for i in sorted_indices]

    best_true_rank = float('inf')
    mrr = 0.0
    cosines = []

    for i in range(5):
        caption_index = i
        rank = (sorted_indices == caption_index).nonzero(as_tuple=True)[0].item()
        best_true_rank = min(best_true_rank, rank)
        mrr += 1 / (rank + 1)
        cosines.append(similarities[caption_index].item())

    total_rank += best_true_rank + 1
    total_mrr += mrr / 5
    total_cosine += sum(cosines) / 5

    if best_true_rank == 0:
        top1_count += 1
    if best_true_rank < 5:
        top5_count += 1

avg_rank = total_rank / num_images
recall_at_1 = top1_count / num_images
recall_at_5 = top5_count / num_images
mean_mrr = total_mrr / num_images
mean_cosine = total_cosine / num_images

print(f"\n📊 CLIP Evaluation on {num_images} images (5 real captions + 5 distractors):")
print(f"Average Rank of best true caption: {avg_rank:.2f}")
print(f"Recall@1 (Top-1 Accuracy): {recall_at_1:.2%}")
print(f"Recall@5: {recall_at_5:.2%}")
print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.4f}")
print(f"Mean Cosine Similarity (true captions): {mean_cosine:.4f}")
