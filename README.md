# MAVL

MAVL-ZeroShot-LungDisease/
│
├── requirements.txt                 # Python dependencies and versions
├── environment.yml                  # Conda environment specification (optional)
├── README.md                       # Project overview, installation, and usage instructions
├── config.yaml                     # Model hyperparameters and dataset configurations
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch Dataset classes for CheXpert/MIMIC/PadChest
│   ├── data_loader.py              # DataLoader implementations with train/val/test splits
│   ├── preprocessing.py            # Image preprocessing and normalization utilities
│   └── report_generator.py         # DeepSeek-based report generation from findings
│
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py           # ViT-based visual encoder implementation
│   ├── text_encoder.py             # ClinicalBERT-based text encoder
│   ├── dual_head_fusion.py         # Contrastive and supervised heads with fusion
│   ├── neural_memory.py            # Neural memory module for test-time adaptation
│   └── mavl_model.py               # Complete MAVL model integrating all components
│
├── losses/
│   ├── __init__.py
│   ├── contrastive_loss.py         # InfoNCE contrastive loss implementation
│   ├── supervised_loss.py          # Multi-label BCE loss for supervised branch
│   └── combined_loss.py            # Weighted combination of losses
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # AUC, F1, precision, recall, accuracy calculations
│   ├── memory_utils.py             # Helper functions for memory retrieval/update
│   ├── augmentations.py            # Data augmentation strategies
│   ├── seed.py                     # Reproducibility utilities
│   ├── logger.py                   # TensorBoard/WandB logging utilities
│   └── prompt_engineering.py       # Prompt templates for report generation
│
├── train.py                        # Main training script with argument parsing
├── evaluate.py                     # Evaluation script for test sets and unseen diseases
├── inference.py                    # Single image inference with visualization
│
├── scripts/
│   ├── prepare_datasets.sh         # Download and prepare CheXpert/MIMIC/PadChest
│   ├── plot_results.py             # Generate figures and tables from results
│   └── run_experiments.sh          # Batch experiment runner
│
├── notebooks/
│   ├── data_exploration.ipynb     # Dataset analysis and visualization
│   ├── model_analysis.ipynb        # Model performance analysis and ablations
│   └── results_visualization.ipynb # Generate paper figures and tables
│
└── results/
    ├── checkpoints/                # Saved model checkpoints
    ├── logs/                       # Training logs and metrics
    └── figures/                    # Generated plots and visualizations
