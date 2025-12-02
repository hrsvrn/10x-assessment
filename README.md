# Prompted Segmentation for Drywall QA using SEEM

This project implements a text-conditioned segmentation pipeline for drywall quality assurance, specifically targeting "cracks" and "taping areas" using the Segment Everything Everywhere Model (SEEM).

## Project Structure

```
drywall_seem_project/
├── data/                  # Raw downloaded datasets
├── processed_datasets/    # Processed binary masks and CSV
├── src/
│   ├── dataset.py         # Dataset loader (reads from CSV)
│   ├── losses.py          # Custom CombinedLoss (BCE+Dice+Focal)
│   ├── model.py           # SEEM Finetuning Wrapper
│   └── utils.py           # Metrics and helpers
├── train.py               # Training script
├── predict.py             # Inference script
├── download_data.py       # Dataset download script (Roboflow)
├── process_data.py        # Data processing script (COCO -> Binary Masks)
├── setup_seem.sh          # SEEM installation script
├── requirements.txt       # Dependencies
└── report.md              # Project report
```

## Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install SEEM/X-Decoder**:
   Run the setup script to clone and install the SEEM repository:
   ```bash
   bash setup_seem.sh
   ```
   **IMPORTANT**: After running the script, you must add the repo to your PYTHONPATH as instructed by the script output.

## Dataset Preparation

1. **Set API Key**:
   Export your Roboflow API key:
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   ```

2. **Download Datasets**:
   Run the download script to fetch "Drywall-Join-Detect" and "Cracks" datasets in COCO format:
   ```bash
   python download_data.py
   ```

3. **Process Datasets**:
   Convert the downloaded COCO annotations to binary masks and generate a CSV index:
   ```bash
   python process_data.py
   ```
   This will create `processed_datasets/dataset.csv` which is used for training.

## Full Pipeline

To run the entire pipeline (Download -> Process -> Train) in one command:

```bash
python train.py --all
```

## Training

To train the model manually (assuming data is processed):

```bash
python train.py \
  --csv_path processed_datasets/dataset.csv \
  --epochs 40 \
  --batch_size 4 \
  --lr 1e-4 \
  --checkpoint_dir checkpoints
```

**Key Training Details**:
- **Image Size**: 512x512
- **Optimizer**: AdamW
- **Loss**: 0.4*BCE + 0.4*Dice + 0.2*Focal
- **Augmentations**: Brightness, Zoom, CLAHE (Cracks); Rotation, Flip, Color Jitter (Taping).

## Inference

To run inference on a single image:

```bash
python predict.py \
  --image_path /path/to/image.jpg \
  --prompt "segment crack" \
  --checkpoint checkpoints/best_model.pth \
  --output_dir predictions
```

Output will be saved as `{id}__{prompt}.png` (0 or 255 values).

## Reproducibility

- **Seeds**: Random seeds are not explicitly fixed in the provided code but can be added to `train.py` using `torch.manual_seed()`, `np.random.seed()`, etc. for strict determinism.
- **Hardware**: Trained on NVIDIA GPU with Mixed Precision (Amp).
