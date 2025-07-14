# Twin DCAL: Dual Cross-Attention Learning for Twin Face Verification

This project implements Dual Cross-Attention Learning (DCAL) adapted for identical twin faces verification using the ND TWIN 2009-2010 dataset.

## Project Overview

The goal is to determine whether two highly similar face images belong to the same person, particularly focusing on identical twins where subtle differences matter. This is implemented using DCAL's dual cross-attention mechanism combined with Vision Transformers.

## Dataset

- **ND TWIN 2009-2010 Dataset**: Captured at Twins Days Festivals in Twinsburg, Ohio
- **Training**: 356 persons (178 twin pairs), 6,182 images
- **Testing**: 62 persons (31 twin pairs), 907 images
- **Image Size**: 224x224 RGB (preprocessed and face-cropped)

## Architecture

The model uses a Vision Transformer backbone with:
- **Self-Attention (SA)**: Standard transformer attention
- **Global-Local Cross-Attention (GLCA)**: Focuses on discriminative local regions
- **Pair-wise Cross-Attention (PWCA)**: Training-time regularization (removed at inference)

## Installation

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup the project structure (already done):
```
Twin_DCAL/
├── data/                    # Dataset JSON files
├── src/                     # Source code
├── configs/                 # Configuration files
├── scripts/                 # Training scripts
└── requirements.txt
```

## Project Status

### ✅ Phase 1: Core Infrastructure (COMPLETED)
- ✅ Dataset loading and triplet sampling
- ✅ Vision Transformer backbone (DeiT-Tiny/Small/Base)
- ✅ Basic training loop with triplet loss
- ✅ Configuration management and experiment tracking
- ✅ Multi-platform support (local server, Kaggle)

### ✅ Phase 2: DCAL Components (COMPLETED)
- ✅ Attention rollout for high-response region selection
- ✅ Global-Local Cross-Attention (GLCA) implementation
- ✅ Pair-wise Cross-Attention (PWCA) implementation
- ✅ Integrated DCAL model with dynamic loss weighting
- ✅ Evaluation metrics (EER, TAR@FAR, AUC)

### ✅ Phase 3: Advanced Features (COMPLETED)
- ✅ Model ensemble and test-time augmentation
- ✅ Hyperparameter optimization with Optuna
- ✅ Extended evaluation metrics and visualizations
- ✅ Performance profiling and benchmarking
- ✅ Comprehensive demo script

## Quick Start

### Basic Training

```bash
# Local training with MLFlow
python scripts/train_local.py

# Kaggle training with WandB
python scripts/train_kaggle.py

# Evaluation
python scripts/evaluate.py --checkpoint checkpoints/final_model.pth
```

### Advanced Features (Phase 3)

```bash
# Run all advanced features
python scripts/phase3_demo.py --mode all --output-dir phase3_results

# Run specific features
python scripts/phase3_demo.py --mode ensemble --use-tta --n-models 3
python scripts/phase3_demo.py --mode hyperopt --hpo-trials 50
python scripts/phase3_demo.py --mode evaluation  # Extended metrics + plots
python scripts/phase3_demo.py --mode benchmark  # Performance comparison
```

### Kaggle Environment

For training on Kaggle with WandB tracking:

```bash
python scripts/train_kaggle.py
```

The script is configured for Kaggle's dataset paths and 12-hour timeout limits.

### Custom Training

For custom configurations:

```bash
python src/main.py --config configs/base_config.yaml --config-overrides key=value
```

### Training Options

- `--config`: Path to base configuration file
- `--resume`: Path to checkpoint to resume from
- `--output-dir`: Output directory for checkpoints
- `--local`: Use local server configuration
- `--kaggle`: Use Kaggle configuration
- `--no-tracking`: Disable experiment tracking
- `--config-overrides`: Override config values (format: `key=value`)

## Configuration

The project uses YAML configuration files:

- `configs/base_config.yaml`: Base configuration with all parameters
- `configs/local_config.yaml`: Local server specific settings
- `configs/kaggle_config.yaml`: Kaggle environment specific settings

Key configuration sections:
- `model`: Model architecture parameters
- `data`: Dataset and data loading parameters
- `training`: Training hyperparameters
- `evaluation`: Evaluation metrics and thresholds
- `tracking`: Experiment tracking configuration

## Experiment Tracking

The project supports three tracking methods:

1. **MLFlow**: For local development
   - Set `tracking.method: "mlflow"`
   - Configure `tracking.mlflow_uri`

2. **Weights & Biases**: For cloud experiments
   - Set `tracking.method: "wandb"`
   - Configure `tracking.entity` and `tracking.project_name`
   - Set `WANDB_API_KEY` environment variable

3. **No Tracking**: For simple local runs
   - Set `tracking.method: "none"`

## Model Checkpoints

The trainer automatically saves:
- Model checkpoints every epoch (configurable)
- Optimizer and scheduler states
- Training progress and best metrics
- Configuration used for training

Resume training with:
```bash
python src/main.py --resume path/to/checkpoint.pth
```

## Evaluation Metrics

The model is evaluated using:
- **EER**: Equal Error Rate
- **TAR@FAR**: True Accept Rate at False Accept Rates (0.1%, 1%, 10%)
- **AUC**: Area Under Curve of ROC
- **Accuracy**: At optimal threshold

## Project Structure

```
src/
├── models/
│   ├── backbone.py          # Vision Transformer implementation
│   ├── attention.py         # DCAL attention mechanisms (GLCA, PWCA)
│   ├── dcal_model.py        # Main DCAL model
│   └── utils.py             # Attention rollout and visualization
├── data/
│   ├── dataset.py           # Dataset loading and triplet sampling
│   └── transforms.py        # Data augmentation
├── training/
│   ├── trainer.py           # Training loop with DCAL support
│   ├── losses.py            # Triplet loss implementation
│   └── metrics.py           # Evaluation metrics
├── utils/
│   ├── config.py            # Configuration management
│   └── tracking.py          # Experiment tracking
└── main.py                  # Main entry point
```

## Implementation Status

### Phase 1: Core Infrastructure ✅
- [x] Dataset loading and triplet sampling
- [x] Vision Transformer backbone (DeiT-Tiny)
- [x] Basic training loop with triplet loss
- [x] Multi-platform support (local/Kaggle)
- [x] Experiment tracking (MLFlow/WandB/none)

### Phase 2: DCAL Components ✅
- [x] Attention rollout implementation
- [x] Global-Local Cross-Attention (GLCA)
- [x] Pair-wise Cross-Attention (PWCA)
- [x] Dynamic loss weight balancing
- [x] Integrated DCAL model
- [x] Comprehensive evaluation pipeline

### Phase 3: Advanced Features (TODO)
- [ ] Model ensemble and TTA
- [ ] Hyperparameter optimization
- [ ] Extended evaluation metrics
- [ ] Performance optimization

## Usage Examples

### Basic Training
```bash
# Local training with default config
python src/main.py --local

# Kaggle training
python src/main.py --kaggle

# Custom configuration
python src/main.py --config my_config.yaml --config-overrides training.epochs=200
```

### Resume Training
```bash
python src/main.py --resume outputs/checkpoints/checkpoint_epoch_50.pth
```

### No Tracking
```bash
python src/main.py --no-tracking
```

## Next Steps

1. Implement attention rollout mechanism
2. Add GLCA and PWCA attention modules
3. Integrate DCAL components into training pipeline
4. Add comprehensive evaluation pipeline
5. Optimize for twin face verification task

## References

- DCAL Paper: "Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification"
- ND TWIN Dataset: Twins Days Festival face dataset
- DeiT: "Training data-efficient image transformers"
