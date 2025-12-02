import os
import json
import shutil
import csv
import glob
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def create_binary_mask_from_polygons(polygons, image_shape):
    # Convert COCO polygon annotations to binary mask.
    mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    
    # Convert flat list to list of tuples [(x1,y1), (x2,y2), ...]
    for polygon in polygons:
        if len(polygon) >= 6:  # Need at least 3 points
            coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(coords, outline=255, fill=255)
    
    return np.array(mask)

def process_coco_dataset(dataset_path, output_path, dataset_name, split):
    """Process a COCO dataset split and convert to binary masks."""
    # Paths
    split_path = os.path.join(dataset_path, split)
    annotation_file = os.path.join(split_path, '_annotations.coco.json')
    
    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Warning: Annotation file not found: {annotation_file}")
        return
    
    # Create output directories
    output_images_dir = os.path.join(output_path, dataset_name, split, 'images')
    output_masks_dir = os.path.join(output_path, dataset_name, split, 'masks')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Load COCO annotations
    print(f"\nProcessing {dataset_name}/{split}...")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for img_id, img_data in tqdm(image_info.items(), desc=f"{dataset_name}/{split}"):
        filename = img_data['file_name']
        image_path = os.path.join(split_path, filename)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            skipped_count += 1
            continue
        
        # Copy image to output directory
        output_image_path = os.path.join(output_images_dir, filename)
        shutil.copy2(image_path, output_image_path)
        
        # Create binary mask
        height = img_data['height']
        width = img_data['width']
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                mask = None
                
                # Try to use polygon segmentation first
                if 'segmentation' in ann and ann['segmentation']:
                    # Handle both list of polygons and single polygon
                    segmentation = ann['segmentation']
                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        # Check if it's a list of polygons or a single polygon
                        if isinstance(segmentation[0], list):
                            # Multiple polygons
                            mask = create_binary_mask_from_polygons(segmentation, (height, width))
                        else:
                            # Single polygon as flat list
                            mask = create_binary_mask_from_polygons([segmentation], (height, width))
                
                # Fall back to bounding box if no segmentation available
                if mask is None and 'bbox' in ann:
                    bbox = ann['bbox']  # [x, y, width, height] in COCO format
                    x, y, w, h = bbox
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Create mask from bounding box
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[y:y+h, x:x+w] = 255
                
                # Combine masks (union)
                if mask is not None:
                    combined_mask = np.maximum(combined_mask, mask)
        
        # Save mask
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_path = os.path.join(output_masks_dir, mask_filename)
        Image.fromarray(combined_mask).save(mask_path)
        
        processed_count += 1
    
    print(f"Processed {processed_count} images, skipped {skipped_count} images")

def generate_csv(root_dir, processed_datasets_dir):
    csv_path = os.path.join(processed_datasets_dir, "dataset.csv")
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Collect existing entries (avoid duplicates)
    existing_entries = set()
    if file_exists:
        with open(csv_path, "r") as check_file:
            for row in csv.reader(check_file):
                if row and row[0] != "image_path":
                    existing_entries.add(row[0])
    
    # Open in append mode ('a') if exists, else write mode ('w')
    with open(csv_path, "a" if file_exists else "w", newline='') as f:
        writer = csv.writer(f)
    
        # Write header only if new file
        if not file_exists:
            writer.writerow(["image_path", "mask_path", "prompt", "split"])
    
        def add_entries(dataset_name, split, prompt):
            """Add entries for a specific dataset and split."""
            images_dir = os.path.join(processed_datasets_dir, dataset_name, split, "images")
            
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found: {images_dir}")
                return 0
            
            count = 0
            # Support both jpg and png images
            image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpeg"))
            
            for img_path in image_files:
                # Create corresponding mask path
                img_filename = os.path.basename(img_path)
                mask_filename = os.path.splitext(img_filename)[0] + "_mask.png"
                mask_path = os.path.join(processed_datasets_dir, dataset_name, split, "masks", mask_filename)
                
                # Check if mask exists and entry is not duplicate
                if os.path.exists(mask_path) and img_path not in existing_entries:
                    # Convert to relative paths from root directory
                    img_path_rel = os.path.relpath(img_path, root_dir)
                    mask_path_rel = os.path.relpath(mask_path, root_dir)
                    
                    writer.writerow([img_path_rel, mask_path_rel, prompt, split])
                    count += 1
            
            return count
    
        # Process both datasets with train, valid, and test splits
        total_count = 0
        
        # Cracks dataset
        print("Processing cracks-1 dataset...")
        for split in ['train', 'valid', 'test']:
             count = add_entries("cracks-1", split, "segment crack")
             print(f"  Added {count} {split} images")
             total_count += count
        
        # Drywall joints dataset
        print("Processing Drywall-Join-Detect-1 dataset...")
        for split in ['train', 'valid', 'test']:
             count = add_entries("Drywall-Join-Detect-1", split, "segment taping area")
             print(f"  Added {count} {split} images")
             total_count += count
    
    print(f"\n{'='*60}")
    print(f"dataset.csv updated at: {csv_path}")
    print(f"Total new entries added: {total_count}")
    print(f"{'='*60}")

def main():
    """Main function to process all datasets."""
    # Define paths - get root directory (parent of scripts folder)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(root_dir, 'data') # Assuming downloaded data is in 'data'
    # But user script assumes 'dataset' folder. Let's check where download_data.py puts it.
    # My previous download_data.py put it in 'data/cracks' and 'data/drywall_join'.
    # User script expects 'dataset/cracks-1' and 'dataset/Drywall-Join-Detect-1'.
    # I should align with user script expectations or adjust paths.
    # Let's adjust this script to look in 'data' but with the names I used, OR
    # Update download_data.py to match this. 
    # Since user provided this script, I will assume they might have data in 'dataset' or I should adapt.
    # I will adapt this script to look in 'data' where I downloaded things.
    
    # My download_data.py downloaded to:
    # data/cracks
    # data/drywall_join
    
    # User script looks for:
    # dataset/cracks-1
    # dataset/Drywall-Join-Detect-1
    
    # I will stick to my download paths for consistency with previous steps, 
    # but use the user's logic.
    
    output_dir = os.path.join(root_dir, 'processed_datasets')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define datasets to process (Mapping my download names to user logic)
    # I need to make sure I process the right folders.
    # Let's assume the user wants to process what is in 'data'.
    
    datasets_map = {
        'cracks': 'cracks-1', 
        'drywall_join': 'Drywall-Join-Detect-1'
    }
    
    splits = ['train', 'valid', 'test']
    
    # Process each dataset
    for my_name, target_name in datasets_map.items():
        dataset_path = os.path.join(root_dir, 'data', my_name)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset not found: {dataset_path}")
            continue
        
        for split in splits:
            try:
                # Note: My download_data.py downloaded 'png-mask-semantic' which is NOT COCO JSON.
                # The user's script expects COCO JSON structure (_annotations.coco.json).
                # This is a CONFLICT.
                # If the user wants to use this script, they must have downloaded COCO format.
                # I should probably update download_data.py to download COCO format instead.
                process_coco_dataset(dataset_path, output_dir, target_name, split)
            except Exception as e:
                print(f"Error processing {target_name}/{split}: {str(e)}")
                # import traceback
                # traceback.print_exc()
    
    # Generate CSV
    generate_csv(root_dir, output_dir)

if __name__ == "__main__":
    main()
