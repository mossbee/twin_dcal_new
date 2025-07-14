# Twin DCAL Implementation Plan

## Project Overview
Adapt Dual Cross-Attention Learning (DCAL) for identical twin faces verification using ND TWIN 2009-2010 dataset. The task is to determine if two highly similar face images belong to the same person.

## Dataset Analysis
- **Training**: 356 persons (178 twin pairs), 6,182 images, 4-68 images per person
- **Testing**: 62 persons (31 twin pairs), 907 images
- **Image Size**: 224x224 RGB (preprocessed and cropped faces)
- **Data Structure**: 
  - `train_dataset_infor.json`: {person_id: [image_paths]}
  - `train_twin_pairs.json`: [[twin_id1, twin_id2], ...]
  - Same structure for test sets

## Architecture Design

### Core Components to Implement
1. **Vision Transformer Backbone** (use DeiT/ViT from refer_codebase)
2. **Self-Attention (SA)** - Standard transformer attention
3. **Global-Local Cross-Attention (GLCA)** - Custom implementation needed
4. **Pair-wise Cross-Attention (PWCA)** - Custom implementation needed
5. **Attention Rollout** - Available in refer_codebase/attention_rollout

### Model Architecture
```
Input: 224x224x3 face images
├── Patch Embedding (16x16 patches = 196 patches)
├── Position Embedding + CLS token
├── 12 Transformer Blocks with:
│   ├── Self-Attention (SA) - L=12 blocks
│   ├── Global-Local Cross-Attention (GLCA) - M=1 block (at layer 12)
│   └── Pair-wise Cross-Attention (PWCA) - T=12 blocks (training only)
└── Classification Head
```

### GLCA Implementation Details
- Use attention rollout to find top R=30% high-response regions
- Compute cross-attention between local query and global key-value pairs
- Only used during inference (small parameter increase ~8-9%)

### PWCA Implementation Details
- Sample image pairs during training (both intra-class and inter-class)
- Concatenate key-value from both images: K_c=[K1;K2], V_c=[V1;V2]
- Compute attention between Q1 and combined K_c, V_c
- Acts as regularization - removed during inference

## Project Structure

```
Twin_DCAL/
├── data/                           # Dataset info files
├── refer_codebase/                 # Reference implementations
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py            # ViT/DeiT backbone
│   │   ├── attention.py           # SA, GLCA, PWCA implementations
│   │   ├── dcal_model.py          # Main DCAL model
│   │   └── utils.py               # Attention rollout, etc.
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Twin dataset loader
│   │   ├── transforms.py          # Data augmentation
│   │   └── samplers.py            # Pair sampling for PWCA
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── losses.py              # Triplet loss implementation
│   │   ├── metrics.py             # EER, TAR@FAR, AUC
│   │   └── callbacks.py           # Checkpointing, tracking
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── tracking.py            # MLFlow, WandB, no tracking
│   │   └── checkpoints.py         # Model checkpointing
│   └── main.py                    # Entry point
├── configs/
│   ├── base_config.yaml           # Base configuration
│   ├── local_config.yaml          # Local server config
│   └── kaggle_config.yaml         # Kaggle config
├── scripts/
│   ├── train_local.py             # Local training script
│   ├── train_kaggle.py            # Kaggle training script
│   └── evaluate.py               # Evaluation script
├── requirements.txt
└── README.md
```

## Implementation Priority

### Phase 1: Core Infrastructure
1. **Dataset Loading** (`src/data/dataset.py`)
   - Load twin face images from JSON files
   - Handle train/test splits
   - Basic transforms (resize, normalize)

2. **Basic ViT Backbone** (`src/models/backbone.py`)
   - Adapt DeiT-Tiny from refer_codebase
   - 224x224 input, 16x16 patches
   - Standard self-attention only

3. **Training Loop** (`src/training/trainer.py`)
   - Basic triplet loss
   - Simple evaluation metrics
   - Model checkpointing

### Phase 2: DCAL Components
1. **Attention Rollout** (`src/models/utils.py`)
   - Adapt from refer_codebase/attention_rollout
   - Calculate accumulated attention scores
   - Select top-R high-response regions

2. **GLCA Implementation** (`src/models/attention.py`)
   - Cross-attention between local query and global key-value
   - Integration with ViT backbone

3. **PWCA Implementation** (`src/models/attention.py`)
   - Pair sampling strategy (random from same dataset)
   - Cross-attention with concatenated key-value pairs
   - Training-only component

### Phase 3: Advanced Features ✅ COMPLETED
1. **Multi-Platform Support** ✅
   - Local Ubuntu server (MLFlow tracking)
   - Kaggle environment (WandB tracking)
   - No tracking option

2. **Advanced Training Features** ✅
   - Model ensemble and test-time augmentation (`src/training/ensemble.py`)
   - Hyperparameter optimization with Optuna (`src/training/hyperopt.py`)
   - Dynamic loss weights (from FairMOT reference)
   - Resume from checkpoint capability
   - Performance profiling and benchmarking

3. **Extended Evaluation & Metrics** ✅
   - EER (Equal Error Rate)
   - TAR@FAR (True Accept Rate at False Accept Rate)
   - AUC (Area Under Curve)
   - Precision-Recall analysis
   - ROC curve analysis
   - Embedding visualization (t-SNE)
   - Confusion matrix analysis
   - Performance benchmarking

## Training Strategy

### Loss Function
- **Triplet Loss**: Learn similarity within same person, differences between twins
- **Dynamic Weight Balancing**: Between SA, GLCA branches (no weight for PWCA)

### Data Sampling for PWCA
- Random sampling from same dataset (natural distribution)
- Mix of intra-class and inter-class pairs
- Both positive (same person) and negative (different person) pairs

### Training Configuration
- **Image Size**: 224x224 (efficient, fast training)
- **Batch Size**: 16-32 (depending on GPU memory)
- **Learning Rate**: Cosine decay, starting from 5e-4
- **Optimizer**: Adam with weight decay 0.05
- **Epochs**: 100-120
- **Local Query Selection**: R=30% (based on Re-ID experiments)

### Multi-Platform Considerations
- **Local Server**: 2x RTX 2080Ti, MLFlow tracking
- **Kaggle**: T4/P100, WandB tracking, 12h timeout → checkpoint every epoch
- **Resume Training**: Load model, optimizer, scheduler state

## Key Implementation Challenges

1. **GLCA Cross-Attention**: Need to implement cross-attention mechanism between selected local queries and global key-value pairs

2. **PWCA Pair Sampling**: Design effective strategy for sampling image pairs during training without significant overhead

3. **Attention Rollout**: Adapt the implementation for Vision Transformers and integrate with GLCA

4. **Multi-Head Integration**: Properly integrate GLCA and PWCA with multi-head self-attention

5. **Dynamic Loss Weights**: Implement collaborative optimization between SA and GLCA branches

## Success Metrics
- **Baseline Target**: Match or exceed standard ViT performance on twin verification
- **DCAL Improvement**: Demonstrate consistent improvement over SA-only baseline
- **Computational Efficiency**: GLCA adds <10% params/FLOPs, PWCA zero inference cost
- **Evaluation Metrics**: Strong performance on EER, TAR@FAR, AUC

## Next Steps
1. Set up basic project structure
2. Implement dataset loading and basic ViT backbone
3. Add triplet loss and basic training loop
4. Progressively add GLCA and PWCA components
5. Optimize for both local and Kaggle environments
