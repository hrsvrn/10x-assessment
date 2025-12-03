import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from config import Config

def predict():
    args = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    # If checkpoint provided, load from there, else load base model
    if args.CHECKPOINT_PATH:
        print(f"Loading model from {args.CHECKPOINT_PATH}")
        model = CLIPSegForImageSegmentation.from_pretrained(args.CHECKPOINT_PATH)
        processor = CLIPSegProcessor.from_pretrained(args.CHECKPOINT_PATH)
    else:
        print("Loading base CLIPSeg model")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        
    model.to(device)
    model.eval()
    
    # Load image
    image = Image.open(args.IMAGE_PATH).convert("RGB")
    
    # Process inputs
    prompts = args.PROMPTS
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
        
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(args.OUTPUT_DIR, "prediction.png")
    plt.savefig(output_path)
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    predict()
