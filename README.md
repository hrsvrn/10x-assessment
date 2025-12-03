# Prompted Segmentation for Drywall QA using CLIPSeg

This project implements a text-conditioned segmentation pipeline for drywall quality assurance, specifically targeting "cracks" and "taping areas" using **CLIPSeg** (CIDAS/clipseg-rd64-refined).

## Project Structure

```
drywall_seem_project/
├── data/                  # Raw downloaded datasets
├── processed_datasets/    # Processed binary masks and CSV
├── train.py               # Training script (CLIPSeg)
├── predict.py             # Inference script (CLIPSeg)
├── download_data.py       # Dataset download script (Roboflow)
├── process_data.py        # Data processing script (COCO -> Binary Masks)
├── requirements.txt       # Dependencies
└── report.md              # Project report
```

## Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. **Set API Key**:
   Export your Roboflow API key (or create a `.env` file):
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   ```

2. **Download Datasets**:
   Run the download script to fetch "Drywall-Join-Detect" and "Cracks" datasets:
   ```bash
   python download_data.py
   ```

3. **Process Datasets**:
   Convert the downloaded annotations to binary masks and generate a CSV index:
   ```bash
   python process_data.py
   ```
   This will create `processed_datasets/dataset.csv` which is used for training.

## Training

To train the CLIPSeg model:

```bash
python train.py \
  --csv_path processed_datasets/dataset.csv \
  --epochs 5 \
  --batch_size 4 \
  --lr 1e-5 \
  --checkpoint_dir checkpoints
```

**Key Details**:
- **Model**: CIDAS/clipseg-rd64-refined
- **Input**: Image + Text Prompt ("segment crack", "segment taping area")
- **Output**: Binary Mask (352x352)

## Inference

To run inference on a single image:

```bash
python predict.py \
  --image_path /path/to/image.jpg \
  --prompts "segment crack" "segment taping area" \
  --checkpoint_path checkpoints/checkpoint-epoch-5 \
  --output_dir predictions
```

Output will be saved as `prediction.png` showing the original image and heatmaps for each prompt.
