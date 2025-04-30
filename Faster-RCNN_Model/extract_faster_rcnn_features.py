import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
import h5py
import numpy as np

data_folder = "data"
output_folder = "features_rcnn"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
data_name = 'coco_5_cap_per_img_5_min_word_freq'
use_subset = True
subset_percentage = 0.2

os.makedirs(output_folder, exist_ok=True)

class FasterRCNNFeatureExtractor:
    def __init__(self, device):
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.features = None
        self.hook = self.model.backbone.register_forward_hook(self._hook_fn)
        
    def _hook_fn(self, module, input, output):
        self.features = output['3']
        
    def extract_features(self, image_tensor):
        with torch.no_grad():
            _ = self.model([image_tensor.to(self.device)])
            return self.features.cpu()
    
    def close(self):
        self.hook.remove()

def load_image_from_hdf5(hdf5_file, idx):
    image = hdf5_file['images'][idx]
    image = torch.FloatTensor(image / 255.)
    return image

preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_hdf5(split):
    print(f"Processing {split} set...")
    hdf5_path = os.path.join(data_folder, f"{split}_IMAGES_{data_name}.hdf5")
    with h5py.File(hdf5_path, 'r') as h:
        num_images = len(h['images'])
        print(f"Found {num_images} images in {hdf5_path}")
        if use_subset:
            subset_size = int(num_images * subset_percentage)
            np.random.seed(42)
            indices = np.random.choice(num_images, subset_size, replace=False)
            print(f"Using {subset_size} images ({subset_percentage*100}% of dataset)")
        else:
            indices = range(num_images)
        feature_extractor = FasterRCNNFeatureExtractor(device)
        for idx in tqdm(indices, desc=f"Extracting features for {split}"):
            output_path = os.path.join(output_folder, f"{split}_{idx:012d}.pt")
            if os.path.exists(output_path):
                continue
            try:
                image = load_image_from_hdf5(h, idx)
                image = preprocess(image)
                features = feature_extractor.extract_features(image)
                torch.save(features, output_path)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
        feature_extractor.close()

def main():
    extract_features_from_hdf5('TRAIN')
    extract_features_from_hdf5('VAL')
    train_files = [f for f in os.listdir(output_folder) if f.startswith('TRAIN_') and f.endswith('.pt')]
    val_files = [f for f in os.listdir(output_folder) if f.startswith('VAL_') and f.endswith('.pt')]
    print(f"Extracted features for {len(train_files)} training images and {len(val_files)} validation images")

if __name__ == "__main__":
    main()

