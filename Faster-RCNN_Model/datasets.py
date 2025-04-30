import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np

class CaptionDataset(Dataset):
    def __init__(self, data_folder, data_name, split, transform=None, use_subset=True, subset_percentage=0.2):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        self.transform = transform
        if use_subset and subset_percentage < 1.0:
            num_images = len(self.captions) // self.cpi
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
        else:
            self.img_indices = None
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        if self.img_indices is not None:
            img_index = self.img_indices[i]
        else:
            img_index = i // self.cpi
        img = torch.FloatTensor(self.imgs[img_index] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            if self.img_indices is not None:
                all_indices = [j for j, idx in enumerate(self.img_indices) if idx == img_index]
                all_captions = torch.LongTensor([self.captions[j] for j in all_indices])
            else:
                start_idx = (img_index * self.cpi)
                end_idx = start_idx + self.cpi
                all_captions = torch.LongTensor(self.captions[start_idx:end_idx])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

