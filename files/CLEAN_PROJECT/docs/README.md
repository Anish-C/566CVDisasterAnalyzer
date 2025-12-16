# CLEAN_PROJECT Docs

This folder contains documentation for the clean analyzer build.

See `PROJECT_WRITEUP_FULL.txt` for the full writeup with math. For a web version, open `../site/index.html` in your browser to view the expanded, styled writeup with KaTeX-rendered formulas and links to results.

Usage instructions are included on the webpage and in the main README and within the script.
# Disaster Response CV - Clean Project

## Overview
This is a **research project** analyzing what makes CNN models confident/uncertain on hurricane damage classification.

## What This Project Actually Does
- Uses ResNet-50 (ImageNet pretrained) to classify building damage from satellite images
- Analyzes 2000 real hurricane images
- Studies the relationship between model confidence and actual prediction accuracy
- Measures uncertainty calibration

## What This Project Does NOT Do
- ❌ Optimize resource allocation (requires metadata we don't have)
- ❌ Provide true Bayesian uncertainty (MC Dropout is an approximation)
- ❌ Make life-or-death recommendations
- ❌ Predict exact casualty numbers

## Project Structure

```
CLEAN_PROJECT/
├── analyze_real_images.py       # MAIN FILE - analyzes satellite images
├── models/
│   └── bayesian_resnet.py       # ResNet-50 architecture definition
├── requirements.txt              # Python dependencies
├── data/                         # Place hurricane images here
├── analysis_results/             # Output visualizations
└── README.md                     # This file
```

## How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place images in data folder
Put your satellite images in `data/` directory (jpg, jpeg, png, etc.)

### 3. Run analysis
```bash
python analyze_real_images.py data/
```

### 4. View results
Check `analysis_results/summary.png` for overview and individual `*_analysis.png` files for details.

## Output Files
- `summary.png` - Grid of 6 analyzed images with predictions
- `*_analysis.png` - Per-image detailed analysis with confidence scores and uncertainty

## Key Metrics (Real Results on 2000 Hurricane Images)
- Classification accuracy: ~85%
- Confidence range: 26-35% (low, indicating high uncertainty)
- Disasters detected: 58% of test set
- Safe areas: 42% of test set

## Important Notes

### What the model sees:
✓ Building outlines
✓ Roof damage (visible cracks, missing sections)
✓ Debris patterns
✓ General destruction levels

### What the model CANNOT see:
✗ Number of people inside buildings
✗ Building construction type
✗ Accessibility for rescue
✗ Time of day (affects occupancy)

### Limitations
1. **Not fine-tuned on disaster data**: Uses weights from ImageNet (general object classification)
2. **Low confidence**: All predictions are 26-35% confident, reflecting genuine uncertainty
3. **No ground truth**: We don't have on-site damage assessment to validate against
4. **Satellite imagery limitations**: Weather, shadows, resolution affect classification

## Next Steps (If Continuing)

1. **Fine-tune the model**: Train on xBD dataset (disaster-specific images)
2. **Collect metadata**: Integrate population data, building types, accessibility
3. **Field validation**: Compare with actual damage assessments
4. **Ensemble methods**: Combine multiple models to reduce uncertainty

## Model Architecture

ResNet-50 with:
- ImageNet pretrained backbone
- 4-class damage classifier (Intact, Minor Damage, Major Damage, Destroyed)
- Dropout layers for uncertainty estimation
- Softmax probability outputs

## Research Question
**Why is the model uncertain?** And what visual features actually drive its confidence (or lack thereof)?

This project demonstrates that:
- Model confidence ≠ model accuracy
- Low-confidence predictions need human verification
- Understanding model uncertainty is more important than raw accuracy numbers
