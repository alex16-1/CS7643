import os
import json
import random
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

images_folder = 'val2017'
annotations_file = 'annotations/captions_val2017.json'

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

num_images = 100
selected_image_ids = random.sample(list(image_captions.keys()), num_images)

total_recall_at_1 = 0
total_recall_at_5 = 0
total_rank = 0
total_mrr = 0.0
total_cosine_sim = 0.0

for image_id in tqdm(selected_image_ids, desc="Evaluation"):
    image_path = os.path.join(images_folder, image_id_to_filename[image_id])
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    correct_caption = image_captions[image_id][0]

    distractor_captions = []
    while len(distractor_captions) < 9:
        random_id = random.choice(list(image_captions.keys()))
        if random_id != image_id:
            distractor_captions.append(image_captions[random_id][0])

    all_captions = [correct_caption] + distractor_captions
    text_tokens = clip.tokenize(all_captions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    similarities = (image_features @ text_features.T).squeeze(0)
    sorted_indices = similarities.argsort(descending=True)
    rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item()

    total_rank += rank + 1
    total_mrr += 1.0 / (rank + 1)
    total_cosine_sim += similarities[0].item()

    if rank == 0:
        total_recall_at_1 += 1
    if rank < 5:
        total_recall_at_5 += 1

average_rank = total_rank / num_images
recall_at_1 = total_recall_at_1 / num_images
recall_at_5 = total_recall_at_5 / num_images
mean_reciprocal_rank = total_mrr / num_images
mean_cosine_similarity = total_cosine_sim / num_images

print(f"\nEvaluation results over {num_images} images:")
print(f"Average rank of the correct caption: {average_rank:.2f}")
print(f"Recall@1 (Top-1 Accuracy): {recall_at_1:.2%}")
print(f"Recall@5: {recall_at_5:.2%}")
print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank:.4f}")
print(f"Mean Cosine Similarity (correct caption): {mean_cosine_similarity:.4f}")
