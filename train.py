import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.dataset import DrywallDataset
from src.model import SEEMFinetuner
from src.losses import CombinedLoss
from src.utils import compute_metrics

def train(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset and Dataloader
    # Root dir is project root, assuming CSV paths are relative to it
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_dataset = DrywallDataset(
        csv_path=args.csv_path,
        root_dir=root_dir,
        split='train'
    )
    
    val_dataset = DrywallDataset(
        csv_path=args.csv_path,
        root_dir=root_dir,
        split='valid'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, # Validate one by one for accurate metrics
        shuffle=False, 
        num_workers=2
    )

    # Model
    model = SEEMFinetuner(config_path=args.config_path).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss
    criterion = CombinedLoss().to(device)
    
    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images, prompts)
                # Ensure outputs match mask shape. 
                if outputs.shape != masks.shape:
                    outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_metrics = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation: {val_metrics}")
        
        # Save checkpoint
        if val_metrics['iou'] > best_miou:
            best_miou = val_metrics['iou']
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print("Saved best model.")
            
        # Save last
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'last_model.pth'))

def validate(model, loader, device):
    model.eval()
    metrics = {'iou': [], 'dice': [], 'pixel_acc': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']
            
            outputs = model(images, prompts)
            
            if outputs.shape != masks.shape:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            batch_metrics = compute_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics[k].append(v)
                
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='processed_datasets/dataset.csv', help='Path to dataset CSV')
    parser.add_argument('--config_path', type=str, default=None, help='Path to SEEM config')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--all', action='store_true', help='Run full pipeline: download -> process -> train')
    
    args = parser.parse_args()

    if args.all:
        print("="*60)
        print("Running Full Pipeline")
        print("="*60)
        
        # 1. Download Data
        print("\n[Step 1/3] Downloading Data...")
        try:
            from download_data import download_datasets
            download_datasets()
        except ImportError:
            print("Error: Could not import download_data.py")
            exit(1)
            
        # 2. Process Data
        print("\n[Step 2/3] Processing Data...")
        try:
            from process_data import main as process_datasets
            process_datasets()
        except ImportError:
            print("Error: Could not import process_data.py")
            exit(1)
            
        print("\n[Step 3/3] Starting Training...")

    train(args)
