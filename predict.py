import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import SEEMFinetuner
from src.utils import postprocess_mask

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = SEEMFinetuner(config_path=args.config_path)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocessing
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Load Image
    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {args.image_path}")
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2] # H, W
    
    augmented = transform(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor, [args.prompt])
        # Output is logits (1, 1, 512, 512) or similar
        
    # Postprocess
    mask = postprocess_mask(output, original_size)
    
    # Save
    filename = os.path.basename(args.image_path)
    file_id = os.path.splitext(filename)[0]
    prompt_slug = args.prompt.replace(" ", "_")
    save_name = f"{file_id}__{prompt_slug}.png"
    save_path = os.path.join(args.output_dir, save_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(save_path, mask)
    print(f"Saved mask to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None, help='Path to SEEM config')
    parser.add_argument('--output_dir', type=str, default='predictions')
    
    args = parser.parse_args()
    predict(args)
