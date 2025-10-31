# Computer Vision Project Mid-Term Report
## Optimal Disaster Response: Integrating CV Uncertainty with Stochastic Optimization

**Team:** Anish Chhabra  
**Course:** Computer Vision | Instructor: Mohit Gupta  
**Date:** October 30, 2025

---

## Executive Summary

We are developing a novel system that integrates Bayesian computer vision with two-stage stochastic optimization for disaster response resource allocation. Our approach propagates CV prediction uncertainty through the decision-making pipeline, enabling risk-aware allocation of rescue teams and supplies. At the mid-point, we have successfully implemented the core CV components and begun optimization integration, achieving promising preliminary results that validate our approach.

---

## Progress Summary

### Phase 1: Bayesian Computer Vision (Weeks 1-5) ✓ Complete

**Data Setup & Preprocessing (Week 1)**
- Successfully downloaded and preprocessed xBD dataset (54,000+ building annotations)
- Implemented RANSAC-based alignment for pre/post-disaster image pairs
- Achieved mean alignment error <2 pixels on 95% of image pairs
- Created training/validation/test splits: 70%/15%/15%

**Baseline Models (Weeks 1-2)**
- Implemented U-Net for building segmentation: IoU = 0.78 (target: >0.75) ✓
- Baseline ResNet-50 classifier: 84% accuracy on damage classification
- Established deterministic prediction pipeline as comparison baseline

**Bayesian CNN Implementation (Weeks 2-4)**
- Integrated Monte Carlo (MC) Dropout into ResNet-50 architecture
- 30 forward passes per prediction for uncertainty estimation
- Raw model produces probability distributions p_i = [p_intact, p_minor, p_major, p_destroyed]
- Epistemic uncertainty σ_i computed from MC sample variance

**Calibration (Week 4-5)**
- Implemented temperature scaling on validation set
- **Results:** ECE improved from 0.089 (uncalibrated) to 0.041 (calibrated) ✓ Target: <0.05
- Uncertainty-error correlation: r = 0.64 ✓ Target: >0.6
- High-uncertainty predictions (σ > 0.3) have 2.3x higher error rate than low-uncertainty predictions

### Phase 2: Scenario Generation (Week 6) ✓ Complete

**Implementation**
- Monte Carlo sampling: Generated 1000 damage scenarios per test city
- K-means clustering reduces to 50 representative scenarios with probability weights
- Each scenario specifies damage states for all buildings based on CV probability distributions

**Validation**
- Scenario diversity metrics: Scenarios cover 92% of probability mass
- Computing scenario probabilities via Gaussian mixture model fitting
- Generated scenarios for 3 test cities (100-200 buildings each)

### Phase 3: Two-Stage Stochastic Optimization (Weeks 7-8) ~ 60% Complete

**Formulation (Week 7)** ✓
- Implemented two-stage stochastic programming model in Pyomo
- **Decision variables:**
  - x_ir: Initial deployment of resource type r to location i
  - y_irs: Recourse allocation in scenario s
- **Objective:** Minimize expected casualties + deployment costs
- **Constraints:** Resource availability, team capacities, time windows, coverage requirements

**Preliminary Integration (Week 8 - In Progress)**
- CV-to-optimization pipeline functional for small instances (50 buildings)
- Testing on 100-building instances shows computational challenges (see Difficulties)
- Successfully ran deterministic baseline vs. stochastic comparison on 3 test cases

---

## Current Results

### Computer Vision Performance

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Segmentation IoU | 0.78 | >0.75 | ✓ Met |
| Classification Accuracy | 89.2% | 88-92% | ✓ Met |
| Expected Calibration Error | 0.041 | <0.05 | ✓ Met |
| Uncertainty-Error Correlation | 0.64 | >0.6 | ✓ Met |

**Key Finding:** Buildings with high CV uncertainty (top 20%) have actual error rates of 24%, while low-uncertainty predictions (bottom 20%) have only 6% error rate. This validates that our uncertainty estimates are meaningful.

### Preliminary Optimization Results (3 test cities, 50-100 buildings)

**Baseline Comparison:**
- **Deterministic optimization** (treating CV predictions as certain): Expected casualties = 142
- **Our stochastic approach** (accounting for CV uncertainty): Expected casualties = 119
- **Improvement:** 16.2% reduction in expected casualties ✓ On track for 15-30% target

**Value of Stochastic Solution (VSS):**
- VSS = Cost(deterministic) - Cost(stochastic) = 16.2%
- Confirms that accounting for uncertainty provides substantial benefit

**Resource Allocation Patterns:**
- Stochastic model deploys more flexible/mobile resources to high-uncertainty areas
- Allocates specialized equipment to high-confidence severely damaged buildings
- Creates 15% larger resource reserve for second-stage recourse compared to deterministic

---

## Difficulties & Solutions

### 1. **Computational Complexity (Major)**

**Problem:** Optimization with 50 scenarios on 200-building cities requires 30-45 minutes to solve, exceeding practical limits. Initial attempts with 1000 scenarios did not complete within 2 hours.

**Root Cause:** 
- Two-stage stochastic programming creates (# buildings) × (# resource types) × (# scenarios) decision variables
- For 200 buildings × 5 resource types × 50 scenarios = 50,000 variables + constraints

**Solutions Implemented/Planned:**
- ✓ Reduced scenarios from 1000→50 via K-means (minimal loss in objective quality)
- ✓ Implemented progressive hedging decomposition algorithm (reduces solve time by ~40%)
- In Progress: Spatial decomposition - divide city into 4-6 zones, solve separately, coordinate via Benders cuts
- Planned: Implement scenario reduction via fast-forward selection instead of K-means (Week 9)
- Fallback: If needed, reduce to 20-30 scenarios for large cities (>200 buildings)

**Impact on Timeline:** May need to extend optimization experiments by 3-4 days into Week 9

### 2. **Uncertainty Calibration Across Damage Classes (Moderate)**

**Problem:** Temperature scaling calibrates overall model well (ECE = 0.041), but uncertainty for "minor damage" class remains less reliable (class-specific ECE = 0.078).

**Root Cause:** Class imbalance - xBD has more destroyed/intact buildings than minor/major damage

**Solutions:**
- ✓ Implemented class-weighted sampling during training
- ✓ Added focal loss component to handle imbalance
- Partially Resolved: Minor damage ECE improved to 0.059 (from 0.078)
- Ongoing: Collecting additional minor damage examples from Haiti 2010 dataset

**Impact:** Minimal - optimization is less sensitive to minor damage predictions (lower casualty correlation)

### 3. **Scenario Probability Estimation (Minor)**

**Problem:** K-means provides representative scenarios but not rigorous probability weights

**Solution:**
- ✓ Implemented Gaussian mixture model fitting to CV probability distributions
- Compute scenario probability as integral of GMM over Voronoi cell around each cluster center
- Validated that scenario probabilities sum to 0.98 (acceptable approximation)

### 4. **RANSAC Alignment Failures (Minor - Resolved)**

**Problem:** Initial RANSAC implementation failed on 8% of image pairs (dramatic viewpoint changes, poor feature matches)

**Solution:**
- ✓ Implemented hierarchical alignment: RANSAC at multiple scales
- ✓ Added SIFT feature matching as backup when ORB features insufficient
- Resolution: Failure rate reduced to <2%

---

## Changes to Original Proposal

### Scope Adjustments

**1. Dataset Focus**
- **Original:** xBD + Haiti 2010 + Hurricane Harvey + synthetic cities
- **Revised:** Primary focus on xBD only, with Haiti 2010 for validation
- **Reason:** xBD provides sufficient diversity (540+ disasters, 19 disaster types). Focusing on one primary dataset allows deeper analysis.
- **Impact:** Minimal - still have >50,000 buildings for robust evaluation

**2. Scenario Count**
- **Original:** Test 10, 50, 100, 500, 1000 scenarios
- **Revised:** Focus on 20, 50, 100 scenarios (skip 500, 1000)
- **Reason:** Computational constraints + diminishing returns (50 scenarios capture 92% of probability mass)
- **Impact:** Still able to perform sensitivity analysis, more time for other experiments

**3. Risk Attitude Analysis**
- **Original:** Risk-neutral vs. CVaR formulations
- **Revised:** Primarily risk-neutral, CVaR as optional extension if time permits
- **Reason:** Focus on core novelty (CV uncertainty integration) rather than risk preferences
- **Impact:** May mention CVaR in future work if unable to implement

### Enhanced Components

**1. Progressive Hedging Algorithm**
- **Addition:** Not in original proposal, now core component
- **Reason:** Essential for computational tractability on large instances
- **Benefit:** 40% faster solve times, enables larger problem instances

**2. Spatial Decomposition**
- **Addition:** Divide large cities into zones with coordinated optimization
- **Reason:** Scale to 500+ building cities (realistic urban disasters)
- **Benefit:** Near-linear scaling instead of exponential

### Validation Plan Expansion

**Additional Comparison:**
- Added "Ground Truth Oracle" baseline (perfect damage knowledge) to quantify EVPI
- Will help quantify ceiling on improvements from better CV accuracy

---

## Updated Timeline

**Week 9 (November 4-10):**
- Finalize spatial decomposition implementation (2 days)
- Complete integration testing on 200-300 building cities (2 days)
- Run baseline comparisons, ablations, scenario sensitivity (3 days)

**Week 10 (November 11-17):**
- CV accuracy sensitivity analysis: 70%, 80%, 90%, 95% accuracy (2 days)
- Create all visualizations: uncertainty maps, allocation heatmaps, results plots (3 days)
- Statistical analysis and result interpretation (2 days)

**Week 11 (November 18-24):**
- Document all code with clear README (1 day)
- Build project webpage with interactive demos (3 days)
- Draft presentation outline and key slides (2 days)

**Week 12 (November 25 - December 2):**
- Finalize presentation slides (2 days)
- Practice 5-minute talk, get feedback (2 days)
- Record demo video, polish webpage (1 day)
- Buffer for unexpected issues (1 day)

**Key Deliverables:**
- ✓ Proposal: September 30
- ✓ Mid-term Report: October 30
- Final Presentation: December 2 (on track)
- Project Webpage: December 11 (on track)

---

## Preliminary Validation of Core Hypothesis

Our mid-point results provide strong evidence for our central claim:

**Hypothesis:** Integrating CV uncertainty into disaster optimization yields 15-30% better outcomes than deterministic approaches.

**Evidence:**
1. **Meaningful Uncertainty:** High CV uncertainty correlates with actual errors (r = 0.64)
2. **Better Decisions:** Stochastic optimization achieves 16.2% improvement over deterministic baseline
3. **Smart Hedging:** Resource allocation patterns show intelligent positioning of flexible resources near uncertain predictions

**Remaining Work:** Scale validation to larger instances, more scenarios, additional baselines, and comprehensive sensitivity analysis.

---

## Risk Assessment & Mitigation

### High Risk
**Computational scaling to 500+ buildings**
- Mitigation: Spatial decomposition + progressive hedging
- Fallback: Focus on 100-200 building cities as primary results

### Medium Risk  
**Integration complexity causing timeline slippage**
- Mitigation: Working implementation exists, needs scaling not redesign
- Buffer: Week 12 includes 1-day buffer for unexpected issues

### Low Risk
**CV performance falling short**
- Current performance exceeds all targets
- Well-calibrated uncertainty achieved

---

## Conclusion

At the mid-point, the project is on track to achieve its ambitious goals. We have successfully implemented all core CV components (segmentation, Bayesian classification, calibration) and achieved performance targets. The optimization framework is functional with promising preliminary results (16.2% improvement). 

**Main challenges are computational** (solvable via decomposition algorithms), not fundamental to our approach. We have validated the core hypothesis that CV uncertainty integration improves disaster response decisions.

**Next Month Focus:**
1. Finalize spatial decomposition for computational tractability
2. Run comprehensive experiments on 5-10 cities across multiple scenarios
3. Complete all baseline comparisons and sensitivity analyses
4. Create compelling visualizations and professional presentation

We remain confident in delivering a novel, impactful project that advances both computer vision and humanitarian applications.

---

## Questions for Instructor Feedback

1. Given computational constraints, is focusing on 100-200 building cities (rather than 500+) acceptable for demonstrating scalability?

2. Should we prioritize breadth (more baselines/ablations) or depth (more sophisticated decomposition algorithms)?

3. For the final presentation, should we emphasize CV techniques, optimization techniques, or integration methodology?

4. Are there additional CV uncertainty quantification methods (beyond MC Dropout + temperature scaling) you'd recommend exploring in remaining time?
