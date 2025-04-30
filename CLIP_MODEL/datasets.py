import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np

class CLIPCaptionDataset(Dataset):
    def __init__(self, data_folder, data_name, split, clip_features_folder, transform=None, 
                 use_subset=True, subset_percentage=0.2):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.clip_features_folder = clip_features_folder
        self.data_folder = data_folder
        self.data_name = data_name

        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        try:
            with h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r') as h:
                self.cpi = h.attrs['captions_per_image']
        except:
            self.cpi = 5
            print(f"Warning: Could not find captions_per_image attribute, defaulting to {self.cpi}")
            
        num_images = len(self.captions) // self.cpi
        
        if use_subset and subset_percentage < 1.0:
            subset_size = int(num_images * subset_percentage)
            np.random.seed(42)
            selected_indices = np.random.choice(num_images, subset_size, replace=False)
            caption_mask = np.zeros(len(self.captions), dtype=bool)
            for idx in selected_indices:
                start_idx = idx * self.cpi
                end_idx = start_idx + self.cpi
                caption_mask[start_idx:end_idx] = True
            self.captions = [self.captions[i] for i in range(len(self.captions)) if caption_mask[i]]
            self.caplens = [self.caplens[i] for i in range(len(self.caplens)) if caption_mask[i]]
            self.img_indices = [i // self.cpi for i in range(len(caption_mask)) if caption_mask[i]]
            print(f"Using {subset_size} images ({subset_percentage*100:.1f}% of dataset)")
            print(f"Selected {len(self.captions)} captions")
        else:
            self.img_indices = [i // self.cpi for i in range(len(self.captions))]
            
        self.dataset_size = len(self.captions)

        self.clip_files = os.listdir(clip_features_folder)
        self.clip_files = [f for f in self.clip_files if f.endswith('.pt')]
        print(f"Found {len(self.clip_files)} CLIP feature files in {clip_features_folder}")
        
        if len(self.clip_files) > 0:
            print(f"Examples of CLIP feature filenames: {self.clip_files[:3]}")

    def __getitem__(self, i):
        img_index = i // self.cpi
        formats_to_try = [
            f"{img_index:012d}.pt",
            f"{img_index}.pt",
            f"COCO_train2014_{img_index:012d}.pt",
            f"COCO_val2014_{img_index:012d}.pt",
        ]
        clip_features = None
        for fmt in formats_to_try:
            feature_path = os.path.join(self.clip_features_folder, fmt)
            try:
                clip_features = torch.load(feature_path, weights_only=True)
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {feature_path}: {e}")
                continue
        if clip_features is None:
            if len(self.clip_files) > 0:
                file_idx = img_index % len(self.clip_files)
                feature_path = os.path.join(self.clip_features_folder, self.clip_files[file_idx])
                try:
                    clip_features = torch.load(feature_path, weights_only=True)
                except Exception as e:
                    print(f"Error loading fallback file {feature_path}: {e}")
                    clip_features = torch.randn(1, 512)
            else:
                print(f"Warning: No CLIP feature files found. Using random features for image index {img_index}.")
                clip_features = torch.randn(1, 512)
        clip_features = clip_features.squeeze(0)
        clip_features = clip_features.float()
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.split == 'TRAIN':
            return clip_features, caption, caplen
        else:
            all_captions = []
            start_idx = (img_index * self.cpi)
            end_idx = start_idx + self.cpi
            for j in range(start_idx, end_idx):
                if j < len(self.captions):
                    all_captions.append(self.captions[j])
            all_captions = torch.LongTensor(all_captions)
            return clip_features, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

