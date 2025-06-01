# MAVL: Zero-shot Lung Disease Detection

This project implements the paper "Zero-shot Lung Disease Detection Using Radiological Symptomatic Descriptors and Pretrained Neural Networks" .

## Overview
The project aims to reproduce the Multi-Aspect Vision-Language (MAVL) framework for detecting lung diseases from chest X-rays, with a focus on zero-shot learning capabilities. The model integrates a Vision Transformer (ViT) for visual features, ClinicalBERT for textual report embeddings, a dual-head fusion mechanism (contrastive and supervised), and a neural memory module for test-time adaptation.

## Directory Structure
MAVL-ZeroShot-LungDisease/
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── config.yaml                          # Hyperparameters and paths
│
├── data_handling/                       # Data loading, preprocessing, splitting
│   ├── report_curation/                 # Report generation simulation
│
├── models/                              # Model components (ViT, BERT, MAVL)
├── losses/                              # Loss function implementations
├── utils/                               # Utility scripts (metrics, logging, etc.)
│
├── train.py                             # Main training script
├── evaluate.py                          # Evaluation script
│
├── scripts/                             # Helper scripts (data prep, plotting)
├── notebooks/                           # Jupyter notebooks for analysis
└── results/                             # Output directory for checkpoints, logs, plots


## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd MAVL-ZeroShot-LungDisease
    ```

2.  **Create a Conda environment (recommended):**
    The paper uses PyTorch 2.1.0 and Transformers 4.35.2.
    ```bash
    conda create -n mavl python=3.10 -y
    conda activate mavl
    ```
    Alternatively, create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Install PyTorch with CUDA support matching your system:
    # Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for correct command.
    # Example: pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Set up Data:**
    * Download CheXpert, MIMIC-CXR, and PadChest datasets.
    * Update dataset paths in `config.yaml`.
    * Run `python scripts/prepare_data_splits.py --dataset <dataset_name>` to generate consistent data splits (this script needs to be implemented based on dataset specifics).

## Usage

### Configuration
Adjust hyperparameters, dataset paths, and model settings in `config.yaml`.

### Training
To train the MAVL model:
```bash
python train.py --config_path config.yaml --dataset_name CheXpert --model_size ViT-B_16 --unseen_percentage 5 --output_dir results/CheXpert_ViTB16_unseen5
