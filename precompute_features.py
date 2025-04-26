import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_loader import COCODataset, Vocabulary
from vit_Feature_extractor import ViTFeatureExtractor
import numpy as np
from tqdm import tqdm


coco_path = 'data/data'
train_image_dir = f'{coco_path}/images/train2017'
val_image_dir = f'{coco_path}/images/val2017'
annotations_dir = f'{coco_path}/annotations'
train_annotation_file=f'{annotations_dir}/captions_train2017.json'
val_annotation_file=f'{annotations_dir}/captions_val2017.json'


def precompute_features(vit_model, image_dir, annotations_file, save_dir, batch_size=32, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)

    dataset = COCODataset(image_dir, annotations_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

    vit_model.eval()
    vit_model.cuda()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting ViT features"):
            images, _, image_ids = zip(*batch)
            images = list(images)

            features = vit_model(images)  # Shape: [batch_size, 768]
            features = features.cpu().numpy()

            for i, image_id in enumerate(image_ids):
                feature_path = os.path.join(save_dir, f"{image_id}.npy")
                np.save(feature_path, features[i])



########################################################## PRECOMPUTE THE FEATURES #######################################################
vit_model = ViTFeatureExtractor()
precompute_features(vit_model, train_image_dir, train_annotation_file ,'./features/train')
precompute_features(vit_model, val_image_dir, val_annotation_file,'./features/val')
