import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torch.optim import AdamW
from tqdm import tqdm
import argparse

class DrywallDataset(Dataset):
    def __init__(self, csv_path, root_dir, processor, split='train'):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert("RGB")
        
        # Load mask
        mask_path = os.path.join(self.root_dir, row['mask_path'])
        mask = Image.open(mask_path).convert("L")
        
        # Get prompt
        prompt = row['prompt']
        
        # Process inputs
        # CLIPSegProcessor handles resizing and normalization
        inputs = self.processor(
            text=[prompt], 
            images=[image], 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # Process target mask
        # We need to resize mask to match model output or input size?
        # CLIPSeg output is 352x352 usually. Processor resizes image to 352x352.
        # We should resize mask to 352x352 to match logits.
        mask = mask.resize((352, 352), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        # mask_tensor shape: (352, 352)
        
        # Add mask to inputs (for loss calculation if we were using a trainer, but we do manual loop)
        inputs['labels'] = mask_tensor
        
        # Remove batch dimension added by processor
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        return inputs

import numpy as np

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize processor and model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    
    # Create datasets
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust if csv_path is relative
    if not os.path.isabs(args.csv_path):
        csv_path = os.path.join(root_dir, args.csv_path)
    else:
        csv_path = args.csv_path
        
    train_dataset = DrywallDataset(csv_path, root_dir, processor, split='train')
    valid_dataset = DrywallDataset(csv_path, root_dir, processor, split='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                pixel_values=pixel_values, 
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids, 
                    pixel_values=pixel_values, 
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        model.save_pretrained(os.path.join(args.checkpoint_dir, f"checkpoint-epoch-{epoch+1}"))
        processor.save_pretrained(os.path.join(args.checkpoint_dir, f"checkpoint-epoch-{epoch+1}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="processed_datasets/dataset.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    train(args)
