# Training Guide: Fine-tune ResNet-50 on Disaster Data

## Prerequisites
1. **Better computer with GPU** (NVIDIA with CUDA support recommended)
2. **10k labeled disaster images** in `archive (1)/train_another/` (damage/ and no_damage/ folders)
3. **Python environment** with PyTorch + torchvision

## Quick Start (GPU Recommended)

### 1. Navigate to CLEAN_PROJECT
```bash
cd files/CLEAN_PROJECT
```

### 2. Install dependencies (if needed)
```bash
pip install torch torchvision matplotlib numpy pillow
```

### 3. Verify data exists
```bash
# Windows PowerShell
(Get-ChildItem -Path '..\..\archive (1)\train_another\damage' -File).Count
(Get-ChildItem -Path '..\..\archive (1)\train_another\no_damage' -File).Count

# Should show ~5000 images each
```

### 4. Run training
```bash
python finetune_resnet.py
```

**Expected time:**
- **CPU**: 30-90 minutes
- **GPU (CUDA)**: 5-15 minutes

## What Happens During Training
1. Loads 10k images (80% train, 20% validation)
2. Fine-tunes ResNet-50 for 10 epochs
3. Saves best model to `finetuned_resnet50.pt` (based on validation accuracy)
4. Creates `training_history.png` with loss/accuracy curves

## After Training

### 1. Verify model was saved
```bash
ls finetuned_resnet50.pt
```

### 2. Update analyzer to use fine-tuned model
The analyzer needs to be modified to load the fine-tuned weights. Add a `--finetuned` flag.

### 3. Generate new results
```bash
# Copy hurricane images to data folder first
python analyze_real_images.py --images ./data --finetuned
```

## Expected Improvements
- **Accuracy**: 70-90% (vs ~25% random baseline)
- **Confidence**: 50-80% average (vs ~28% pretrained)
- **Grad-CAM**: Should highlight actual damage features (cracks, collapsed roofs) instead of random textures

## Troubleshooting

### Out of Memory (OOM)
Edit `finetune_resnet.py` line 224:
```python
batch_size=16,  # Reduced from 32
```

### Training too slow on CPU
Reduce epochs in line 223:
```python
num_epochs=5,  # Reduced from 10
```

### CUDA not available
Install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Files Generated
- `finetuned_resnet50.pt` - Trained model weights (~100MB)
- `training_history.png` - Loss/accuracy plots
- Console logs showing progress

## Next: Compare Before/After
Once training completes, you can generate side-by-side comparisons:
1. Baseline (ImageNet): Already done in `analysis_results/`
2. Fine-tuned: New results showing improved accuracy and confidence
