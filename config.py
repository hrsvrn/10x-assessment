class Config:
    # Data
    CSV_PATH = "processed_datasets/dataset.csv"
    
    # Training
    EPOCHS = 5
    BATCH_SIZE = 4
    LR = 1e-5
    CHECKPOINT_DIR = "checkpoints"
    
    # Prediction
    IMAGE_PATH = "path/to/image.jpg" # Default or placeholder
    PROMPTS = ["segment crack", "segment taping area"]
    CHECKPOINT_PATH = None # None for base model, or path to checkpoint
    OUTPUT_DIR = "predictions"
