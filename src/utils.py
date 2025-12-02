import torch
import numpy as np
import cv2

def compute_metrics(pred_mask, gt_mask):
    """
    Compute mIoU, Dice, and Pixel Accuracy.
    Args:
        pred_mask: Binary prediction mask (0 or 1), shape (H, W) or (B, H, W)
        gt_mask: Binary ground truth mask (0 or 1), shape (H, W) or (B, H, W)
    Returns:
        dict: {'iou': float, 'dice': float, 'pixel_acc': float}
    """
    # Ensure inputs are boolean or 0/1
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5

    intersection = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred_mask.float().sum() + gt_mask.float().sum() + 1e-6)
    pixel_acc = (pred_mask == gt_mask).float().mean()

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'pixel_acc': pixel_acc.item()
    }

def postprocess_mask(logits, original_size, threshold=0.0):
    """
    Post-process logits to binary mask and resize to original size.
    Args:
        logits: Raw logits from model, shape (1, H, W)
        original_size: Tuple (H, W)
        threshold: Threshold for logits
    Returns:
        np.array: Binary mask (0 or 255), shape (H, W)
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    
    # Sigmoid
    probs = 1 / (1 + np.exp(-logits))
    
    # Threshold
    mask = (probs > 0.5).astype(np.uint8) * 255
    
    # Squeeze channel dim if present
    if mask.ndim == 3:
        mask = mask[0]
        
    # Resize to original size
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask
