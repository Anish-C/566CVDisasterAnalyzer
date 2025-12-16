# Updates: Honest Uncertainty Approach

## Changes Made

### 1. Removed Predictive Distribution Math
**Old Approach:** The writeup included full Bayesian predictive distribution formulas and MC Dropout mathematical framework, but none of it was actually implemented in code.

**New Approach:** Removed the complex predictive integral and MC Dropout formulas. Kept only what we actually use: **softmax confidence**.

---

### 2. Added Temperature Scaling for Calibration
**What it does:** Adjusts softmax probabilities to be softer (less confident) without retraining.

**Formula:**
$$P_T(y=c | x) = \frac{\exp(z_c / T)}{\sum_k \exp(z_k / T)}$$

- `T > 1`: Lowers confidence (spreads probability mass) — useful for domain-shifted models
- `T = 1`: No change (vanilla softmax)
- `T < 1`: Raises confidence (sharpens probabilities)

**Usage:**
```powershell
python analyze_real_images.py --images .\data --temperature 2.0
```

---

### 3. Added Confidence Thresholds for Triage
**What it does:** Automatically flags images below a confidence threshold for human review.

**Why it matters:**
- Low confidence (26–30% in this project) indicates the model is uncertain
- With a threshold, you can automatically escalate uncertain cases
- E.g., `--threshold 0.30` flags all images below 30% confidence

**Usage:**
```powershell
python analyze_real_images.py --images .\data --threshold 0.40
```

**Output:**
- Images below threshold are marked `[REVIEW]` in the console output
- Summary reports how many images need manual inspection

---

### 4. Updated Analyzer Script
**New command-line interface:**
```
--images, -i        Directory with satellite images
--out, -o           Output directory (default: analysis_results)
--temperature, -t   Softmax calibration (default: 1.0, range: 0.1–5.0)
--threshold, -th    Confidence threshold (default: 0.0, range: 0.0–1.0)
```

**Example with both flags:**
```powershell
python analyze_real_images.py --images .\data --temperature 2.0 --threshold 0.30 --out .\my_results
```

---

### 5. Updated Documentation

#### `PROJECT_WRITEUP_FULL.txt`
- Replaced predictive distribution section with practical explanation of softmax calibration and thresholds
- Moved Bayesian formulas to "Framework for Future Work" section (references only, not implemented)

#### `site/index.html` (Webpage)
- Expanded "Uncertainty and Calibration" section with clear explanations of:
  - Softmax confidence
  - Temperature scaling formula and intuition
  - Threshold-based triage
  - Future Bayesian uncertainty (road map)
- Added comprehensive "Run It" section with command examples

#### `docs/README.md`
- Points to the new static website at `../site/index.html`

---

## Test Results

Running on 12 real hurricane satellite images with `--temperature 2.0 --threshold 0.30`:

```
Total images: 12
Disasters detected: 9 (75.0%)
Safe areas: 3 (25.0%)

[REVIEW] Images below confidence threshold: 12 (100.0%)
Confidence statistics:
  Mean: 27.4%
  Min:  25.9%
  Max:  29.4%

Temperature: 2.0 (softmax calibration applied)
```

**Interpretation:** All 12 images flagged for review because all confidences are below 30%. This is **honest and appropriate**—the ImageNet-pretrained model is genuinely uncertain on satellite imagery. Temperature scaling made probabilities softer but didn't eliminate the underlying uncertainty.

---

## What's Next (Roadmap)

1. **MC Dropout / Ensembles** (planned)
   - Add multiple stochastic forward passes
   - Compute predictive entropy and mutual information per image
   - Distinguish epistemic (model parameter) vs. aleatoric (data) uncertainty

2. **Fine-tuning on xBD**
   - Train on disaster-specific dataset with ground truth labels
   - Measure class-wise accuracy and calibration (ECE)

3. **Grad-CAM Overlays**
   - Visualize which regions the model focuses on

4. **Calibration Metrics**
   - Expected Calibration Error (ECE)
   - Reliability diagrams

---

## Why This Approach Is Better

| Aspect | Old | New |
|--------|-----|-----|
| **Honesty** | Theory not implemented | Only implemented features documented |
| **Usability** | One fixed configuration | Flexible via command-line flags |
| **Calibration** | No control | Temperature scaling included |
| **Triage** | Manual thresholding | Automatic flagging with `--threshold` |
| **Code Match** | Math ≠ Code | Math = Code |

---

## Files Modified

- `analyze_real_images.py` — Added temperature, threshold, argparse interface
- `docs/PROJECT_WRITEUP_FULL.txt` — Simplified uncertainty section, moved theory to roadmap
- `site/index.html` — Expanded uncertainty/calibration section with new approach
- `docs/README.md` — Points to website
- This file (`UPDATES.md`) — Summary of changes
