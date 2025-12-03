import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    # If checkpoint provided, load from there, else load base model
    if args.checkpoint_path:
        print(f"Loading model from {args.checkpoint_path}")
        model = CLIPSegForImageSegmentation.from_pretrained(args.checkpoint_path)
        processor = CLIPSegProcessor.from_pretrained(args.checkpoint_path)
    else:
        print("Loading base CLIPSeg model")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        
    model.to(device)
    model.eval()
    
    # Load image
    image = Image.open(args.image_path).convert("RGB")
    
    # Process inputs
    prompts = args.prompts
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    preds = outputs.logits.unsqueeze(1)
    
    # Visualize
    fig, ax = plt.subplots(1, len(prompts) + 1, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    for i, prompt in enumerate(prompts):
        # Resize prediction to image size
        pred = torch.sigmoid(preds[i][0])
        pred = pred.cpu().numpy()
        
        # Resize to original image size for visualization
        # Note: CLIPSeg outputs 352x352. We might want to resize back to original.
        # For visualization, we can just show the output.
        
        ax[i+1].imshow(pred, cmap='viridis')
        ax[i+1].set_title(prompt)
        ax[i+1].axis('off')
        
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "prediction.png")
    plt.savefig(output_path)
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--prompts", nargs="+", default=["segment crack", "segment taping area"], help="List of prompts")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="predictions")
    args = parser.parse_args()
    
    predict(args)
