import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

images_folder = "train2017"
annotations_file = "annotations/captions_train2017.json"
output_folder = "features_clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(output_folder, exist_ok=True)

model, preprocess = clip.load("ViT-B/32", device=device)

with open(annotations_file, 'r') as f:
    annotations = json.load(f)
image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

for image_id, filename in tqdm(image_id_to_filename.items(), desc="Extracting CLIP features"):
    image_path = os.path.join(images_folder, filename)
    output_path = os.path.join(output_folder, f"{filename.split('.')[0]}.pt")
    if os.path.exists(output_path):
        continue
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image).cpu()
        torch.save(features, output_path)
    except Exception as e:
        print(f"Error with {filename}: {e}")

