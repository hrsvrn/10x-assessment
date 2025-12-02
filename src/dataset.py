import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class DrywallDataset(Dataset):
    def __init__(self, cracks_root, taping_root, transform=None, split='train'):
        self.cracks_root = cracks_root
        self.taping_root = taping_root
        self.transform = transform
        self.split = split
        self.samples = []
        
        # Load Cracks Dataset
        self._load_dataset(cracks_root, 'crack')
        
        # Load Taping Dataset
        self._load_dataset(taping_root, 'taping')
        
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

    def _load_dataset(self, root, category):
        if not os.path.exists(root):
            print(f"Warning: Dataset root {root} does not exist. Skipping.")
            return

        # Roboflow export structure handling
        # 1. Check for split folder (train/valid/test)
        split_dir = os.path.join(root, self.split)
        if not os.path.exists(split_dir):
            # Fallback: maybe the root IS the split folder or it's flat
            split_dir = root
            
        # 2. Check for images/masks subdirectories vs flat structure
        # Roboflow 'png-mask-semantic' often puts everything in the split folder
        # or uses 'images' and 'masks' folders.
        
        img_dir = os.path.join(split_dir, 'images')
        mask_dir = os.path.join(split_dir, 'masks')
        
        flat_structure = False
        if not os.path.exists(img_dir):
            # Assume flat structure in split_dir
            img_dir = split_dir
            mask_dir = split_dir
            flat_structure = True

        if not os.path.exists(img_dir):
             return

        valid_exts = ['.jpg', '.jpeg', '.png']
        # List all files
        files = os.listdir(img_dir)
        
        for f in files:
            if any(f.lower().endswith(ext) for ext in valid_exts):
                # Skip mask files if we are in a flat structure
                if flat_structure and '_mask' in f:
                    continue
                    
                img_path = os.path.join(img_dir, f)
                
                # Determine mask path
                # Roboflow semantic mask usually: filename_mask.png
                basename = os.path.splitext(f)[0]
                mask_name = f"{basename}_mask.png"
                mask_path = os.path.join(mask_dir, mask_name)
                
                if not os.path.exists(mask_path):
                    # Try exact name if masks are in separate folder with same name
                    mask_path_alt = os.path.join(mask_dir, f.replace(os.path.splitext(f)[1], '.png'))
                    if os.path.exists(mask_path_alt):
                        mask_path = mask_path_alt
                    else:
                        # Try just .png extension
                         mask_path_alt2 = os.path.join(mask_dir, basename + '.png')
                         if os.path.exists(mask_path_alt2):
                             mask_path = mask_path_alt2
                
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'category': category,
                        'id': basename
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['image'])
        if image is None:
             # Handle broken images
             return self.__getitem__((idx + 1) % len(self))
             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
             # Handle broken masks
             return self.__getitem__((idx + 1) % len(self))
        
        # Ensure mask is 0 or 1
        mask = np.where(mask > 127, 1.0, 0.0).astype(np.float32)
        
        original_size = image.shape[:2]
        
        # Select prompt
        if sample['category'] == 'crack':
            prompt = random.choice(["segment crack", "segment wall crack"])
            transform = self.crack_aug if self.split == 'train' else self.base_transform
        else:
            prompt = random.choice(["segment taping area", "segment drywall seam", "segment joint/tape"])
            transform = self.taping_aug if self.split == 'train' else self.base_transform
            
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
            'id': sample['id']
        }
