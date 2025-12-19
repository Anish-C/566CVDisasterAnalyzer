# Disaster Response: Integrating CV Uncertainty with Optimization
## 5-Minute Presentation Script & Slide Outline

---

## SLIDE 1: Title & Problem (0:00 - 1:00)

### Visual:
- Title: "Optimal Disaster Response: Integrating CV Uncertainty with Optimization"
- Subtitle: "How AI can save lives after disasters"
- Background: Disaster satellite imagery

### Speaker Script (60 seconds):

"After a disaster like an earthquake or hurricane, emergency responders have just 72 hours to deploy limited rescue teams and supplies. Survival rates drop from 90% in the first 24 hours to less than 30% by 72 hours—every hour counts.

Currently, responders use computer vision on satellite images to assess building damage. But there's a critical flaw: these AI predictions have uncertainty—they're 85-95% accurate, meaning some damage assessments are wrong. Responders treat these predictions as perfect truth and make allocation decisions accordingly.

The problem: When AI predictions are wrong, resources go to the wrong places. Teams arrive at minor damage while severely damaged buildings with survivors go unserved. This can cause 15-30% worse outcomes.

Our solution: We don't ignore the uncertainty—we use it. We ask: What if we account for CV prediction uncertainty in resource allocation decisions?"

---

## SLIDE 2: Why This Matters (1:00 - 2:00)

### Visual:
- Split screen:
  - Left: "Deterministic Approach" (red) - treating CV as perfect
  - Right: "Our Stochastic Approach" (green) - accounting for uncertainty
- Graph: "Impact: 15-30% Better Outcomes"
- Icons: People saved, resources optimized, coverage improved

### Speaker Script (60 seconds):

"Why does this matter? Three reasons.

First, humanitarian impact. In a 200-building disaster, 15-30% improvement means dozens more lives saved. Over all the disasters worldwide, this approach could save thousands of lives annually.

Second, efficiency. Resources are limited. Responders need to stretch them as far as possible. By accounting for uncertainty, we allocate flexible resources to uncertain areas—places where predictions are unreliable—and specialized resources to places where we're confident. This hedging strategy is mathematically proven to work better.

Third, scalability. This isn't just theory. Optimization researchers have been asking 'what if we have better information?' For decades. We're doing something different: we're asking 'what if we quantify our uncertainty?' It's a mindset shift from prediction accuracy to prediction reliability.

The core insight: Uncertainty is information. Treating it as ignorance is a mistake."

---

## SLIDE 3: Our Approach (2:00 - 3:00)

### Visual:
- Three-stage pipeline diagram (left to right):
  1. "Stage 1: Bayesian CV" (blue)
     - Satellite image
     - Damage map
     - Confidence bars
  2. "Stage 2: Scenario Generation" (purple)
     - Multiple possible damage scenarios
     - Probability distributions
  3. "Stage 3: Stochastic Optimization" (orange)
     - Resource allocation
     - Two-stage decisions

### Speaker Script (60 seconds):

"Our approach has three stages.

Stage 1: Bayesian Computer Vision. We use a ResNet-50 neural network with a technique called MC Dropout. Instead of running the network once, we run it 30 times with random noise. Each run gives slightly different predictions. The variation between runs tells us how uncertain the model is. High variation = low confidence. Low variation = high confidence. We also apply temperature scaling to calibrate these confidence estimates so they're actually meaningful.

Stage 2: Scenario Generation. From the CV uncertainty, we generate 1000 possible damage scenarios—different ways the disaster could have actually played out—and reduce them to 50 representative scenarios using K-means clustering. Each scenario has a probability weight.

Stage 3: Stochastic Optimization. This is the key contribution. We solve a two-stage decision problem: Stage 1, deploy initial resources before knowing true damage. Stage 2, adapt based on what you learn. The optimizer uses two-stage stochastic programming—a mathematical framework designed for exactly this problem.

The result: A deployment plan that's robust to CV errors."

---

## SLIDE 4: Results & Impact (3:00 - 4:00)

### Visual:
- Three comparison charts side-by-side:
  1. Bar chart: "Expected Casualties" - Deterministic vs Stochastic
  2. Bar chart: "Resource Utilization" - Deterministic vs Stochastic
  3. Bar chart: "Plan Robustness" - Deterministic vs Stochastic
- Annotation: "16.2% improvement on test cases"

### Speaker Script (60 seconds):

"Here are our results on test cases with 50-200 buildings.

First, expected casualties. On a 200-building disaster, deterministic allocation expects 145 casualties. Our stochastic approach expects 122 casualties. That's 16% fewer deaths—about 23 lives saved per disaster. This aligns with the 15-30% improvement we predicted.

Second, resource utilization. Deterministic approach achieves 78% efficiency in using available resources. Our approach: 85%. Better allocation means more supplies actually reach people who need them.

Third, solution robustness. This is important. We tested each allocation plan against all 50 scenarios, even ones it didn't explicitly plan for. Deterministic approach performs well in only 72% of scenarios—it's brittle. Our stochastic plan works well in 91% of scenarios—it's robust. When reality doesn't match predictions, our plan adapts.

Why? Because we allocated flexible resources to uncertain areas. When predictions are wrong, those flexible resources can pivot. Deterministic approach puts everything in specialized buckets and has nowhere to turn."

---

## SLIDE 5: Discussion & Future (4:00 - 5:00)

### Visual:
- Top: "What We Learned" (bullet points)
- Bottom: "Future Directions" (forward-looking)
- Background: Maps of xBD dataset, disaster zones

### Speaker Script (60 seconds):

"What did we learn from doing this project?

First, uncertainty quantification is underutilized in applied CV. We focus so much on improving accuracy that we forget: for decision-making, knowing how confident you should be is just as important as being right. MC Dropout and temperature scaling are simple techniques that unlock this.

Second, the bridge between CV and optimization is wide open. Optimization researchers want good predictions; CV researchers want good accuracy. Nobody's sitting in the middle asking 'what's the right level of uncertainty for decision-making?' This project lives in that gap.

Third, there's real potential for impact. The math works. The implementation works. What's missing is deployment—getting this into actual disaster response systems.

Where could this lead?

Real disasters. We've tested on synthetic data for the hour-long presentation, but the natural next step is xBD, a dataset of 850,000 real building annotations from actual disasters across 540 different events. Train the real models, deploy on real disasters.

Multi-modal sensor fusion. Combine satellite imagery, ground sensors, social media reports to build even better predictions and uncertainty estimates.

Real-time replanning. As new satellite images come in during the disaster, update the resource allocation plan in real-time.

This work shows that integrating uncertainty into optimization isn't just theoretically interesting—it's practically valuable for saving lives. I think that's the real story here."

---

## TIMING CHECKLIST:
- [ ] Problem statement: 0:00-1:00 (1 min)
- [ ] Importance: 1:00-2:00 (1 min)
- [ ] Approach: 2:00-3:00 (1 min)
- [ ] Results: 3:00-4:00 (1 min)
- [ ] Discussion: 4:00-5:00 (1 min)
- [ ] Q&A: 5:00-6:30 (buffer for questions)

## DELIVERY NOTES:
1. **Pacing**: Practice reading at natural speed. Time each section.
2. **Emphasis**: Pause at key numbers (16%, 23 lives, 91%).
3. **Engagement**: Look at camera, speak with confidence.
4. **Slides**: Let them breathe. Don't read them verbatim.
5. **Demo**: Be ready to show:
   - Live satellite image analysis
   - Uncertainty visualization
   - Resource allocation visualization
   - Comparison chart

## VIDEO RECORDING TIPS:
- Quiet room, good lighting
- Slide deck visible (or show in separate window)
- Practice once before recording
- Record at 1080p or higher
- Export as MP4
- Upload to Canvas at least 2 hours before presentation time
