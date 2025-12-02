import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class DrywallDataset(Dataset):
    def __init__(self, csv_path, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Filter by split
        self.data = df[df['split'] == split].reset_index(drop=True)
        
        # Define specific augmentations
        self.crack_aug = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomCrop(height=400, width=400, p=0.5), # Zoom crop simulation
            A.CLAHE(p=0.5),
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.taping_aug = A.Compose([
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.base_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Paths are relative to root_dir
        img_path = os.path.join(self.root_dir, row['image_path'])
        mask_path = os.path.join(self.root_dir, row['mask_path'])
        prompt = row['prompt']
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None:
             # Fallback or error
             print(f"Warning: Could not load image {img_path}")
             return self.__getitem__((idx + 1) % len(self))
             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
             print(f"Warning: Could not load mask {mask_path}")
             return self.__getitem__((idx + 1) % len(self))
        
        # Ensure mask is 0 or 1
        mask = np.where(mask > 127, 1.0, 0.0).astype(np.float32)
        
        original_size = image.shape[:2]
        
        # Select transform based on prompt/category
        if "crack" in prompt:
            transform = self.crack_aug if self.split == 'train' else self.base_transform
            # Augment prompt synonyms
            if self.split == 'train':
                prompt = random.choice(["segment crack", "segment wall crack"])
        else:
            transform = self.taping_aug if self.split == 'train' else self.base_transform
            if self.split == 'train':
                prompt = random.choice(["segment taping area", "segment drywall seam", "segment joint/tape"])
            
        if transform:
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if isinstance(mask, torch.Tensor):
             if mask.ndim == 2:
                 mask = mask.unsqueeze(0)
        else:
             mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            'image': image,
            'mask': mask,
            'prompt': prompt,
            'original_size': original_size,
            'id': os.path.splitext(os.path.basename(img_path))[0]
        }
