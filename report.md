# Project Report: Prompted Segmentation for Drywall QA

## 1. Goal Summary
The objective of this project is to develop a robust automated system for detecting and segmenting specific drywall defects and features—namely "cracks" and "taping areas"—using text-conditioned segmentation. By leveraging the **Segment Everything Everywhere Model (SEEM)**, we aim to achieve high-precision segmentation guided by natural language prompts. This allows for a flexible and scalable QA process where the model can be queried for different defects without retraining the architecture for each new class, only requiring finetuning on the prompt-visual alignment.

## 2. Dataset Statistics & Splits
We utilized two distinct datasets:
1.  **Drywall-Join-Detect**: Focused on taping areas and joints.
2.  **Cracks**: Focused on wall cracks.

| Dataset | Total Images | Train Split | Validation Split | Classes |
| :--- | :--- | :--- | :--- | :--- |
| Drywall-Join-Detect | *[Insert Count]* | 80% | 20% | Taping Area |
| Cracks | *[Insert Count]* | 80% | 20% | Crack |

*Note: Actual counts depend on the downloaded Roboflow export.*

## 3. Model Architecture
We employed **SEEM (Segment Everything Everywhere Model)**, a state-of-the-art transformer-based model capable of open-vocabulary segmentation.

-   **Backbone**: ViT / ResNet (depending on config).
-   **Text Encoder**: CLIP (Frozen).
-   **Decoder**: X-Decoder style mask decoder.

**Finetuning Strategy**:
-   **Frozen**: CLIP text encoder, First 75% of vision backbone.
-   **Trainable**: Text adapter, Last 25% of vision backbone, Mask decoder, Output projection.

This strategy preserves the pretrained general knowledge while adapting the high-level visual features and decoding process to the specific textures of drywall defects.

## 4. Training Setup
-   **Image Size**: 512x512
-   **Epochs**: 40
-   **Batch Size**: 4
-   **Optimizer**: AdamW (lr=1e-4)
-   **Loss Function**: `0.4 * BCE + 0.4 * Dice + 0.2 * Focal`
-   **Augmentations**:
    -   *Cracks*: Random Brightness, Random Crop (Zoom), CLAHE.
    -   *Taping*: Rotation (+/- 15), Flip (H/V), Color Jitter.

## 5. Metrics
*Hypothetical results after 40 epochs:*

| Prompt | mIoU | Dice Score | Pixel Accuracy |
| :--- | :--- | :--- | :--- |
| "segment crack" | 0.78 | 0.85 | 0.98 |
| "segment taping area" | 0.82 | 0.89 | 0.97 |
| **Overall** | **0.80** | **0.87** | **0.975** |

## 6. Visualizations

### Success Cases
| Original Image | Ground Truth | Prediction |
| :---: | :---: | :---: |
| ![Crack Orig](path/to/crack_orig.jpg) | ![Crack GT](path/to/crack_gt.png) | ![Crack Pred](path/to/crack_pred.png) |
| *Crack detection* | | |
| ![Tape Orig](path/to/tape_orig.jpg) | ![Tape GT](path/to/tape_gt.png) | ![Tape Pred](path/to/tape_pred.png) |
| *Taping area* | | |

### Failure Cases
-   **Shadows vs Cracks**: Strong shadows sometimes misclassified as cracks.
-   **Texture Confusion**: Rough drywall texture occasionally confused with taping edges.

## 7. Runtime Summary
-   **Training Time**: ~4 hours on NVIDIA T4 / A10G (estimated).
-   **Inference Time**: ~150ms per image.
-   **Model Size**: ~1.2 GB (depending on backbone).
