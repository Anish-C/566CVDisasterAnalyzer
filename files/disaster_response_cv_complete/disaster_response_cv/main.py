"""
Main Integration Script for Disaster Response CV Project
Demonstrates the full pipeline from CV predictions to resource allocation
"""

import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bayesian_resnet import get_bayesian_resnet50
from utils.scenario_generation import ScenarioGenerator
from optimization.stochastic_optimizer import DisasterResponseOptimizer


def run_full_pipeline(n_buildings=50, n_scenarios=20, visualize=True):
    """
    Run the complete pipeline:
    1. Simulate CV predictions with uncertainty
    2. Generate damage scenarios
    3. Solve stochastic optimization
    4. Compare with deterministic baseline
    
    Args:
        n_buildings: Number of buildings in test case
        n_scenarios: Number of scenarios to generate
        visualize: Whether to create visualizations
    """
    
    print("="*70)
    print("DISASTER RESPONSE CV + OPTIMIZATION PIPELINE")
    print("="*70)
    
    # ========================
    # Step 1: CV Predictions
    # ========================
    print("\n[STEP 1] Simulating CV Predictions...")
    
    # In practice, this would come from your trained model
    # For now, we simulate realistic predictions
    predictions, uncertainties = simulate_cv_predictions(n_buildings)
    
    print(f"  ✓ Generated predictions for {n_buildings} buildings")
    print(f"  - Mean uncertainty: {uncertainties.mean():.4f}")
    print(f"  - High uncertainty buildings: {(uncertainties > 0.15).sum()}")
    
    # Print damage distribution
    predicted_damage = np.argmax(predictions, axis=1)
    print(f"  - Predicted damage distribution:")
    for i, label in enumerate(['No damage', 'Minor', 'Major', 'Destroyed']):
        count = (predicted_damage == i).sum()
        print(f"    {label:12s}: {count:3d} buildings ({count/n_buildings*100:.1f}%)")
    
    # ========================
    # Step 2: Scenario Generation
    # ========================
    print(f"\n[STEP 2] Generating Damage Scenarios...")
    
    generator = ScenarioGenerator(n_samples=1000, n_scenarios=n_scenarios)
    scenarios, probabilities, scenario_info = generator.generate_scenarios(
        predictions, uncertainties
    )
    
    print(f"  ✓ Generated {n_scenarios} representative scenarios")
    print(f"  - Coverage: {scenario_info['coverage']:.1%}")
    
    # ========================
    # Step 3: Stochastic Optimization
    # ========================
    print(f"\n[STEP 3] Solving Stochastic Optimization...")
    
    # Create building info
    building_info = {
        'population': dict(enumerate(np.random.randint(20, 150, size=n_buildings)))
    }
    
    # Resource constraints
    resource_constraints = {
        'capacity': dict(enumerate([50, 40, 25, 60, 45])),
        'costs': dict(enumerate([100, 150, 300, 50, 120]))
    }
    
    optimizer = DisasterResponseOptimizer(n_buildings=n_buildings, solver='glpk')
    stochastic_solution = optimizer.solve(
        scenarios, probabilities, building_info, resource_constraints
    )
    
    # ========================
    # Step 4: Deterministic Baseline
    # ========================
    print(f"\n[STEP 4] Solving Deterministic Baseline...")
    
    # Deterministic: use only most likely scenario
    deterministic_scenario = predicted_damage.reshape(1, -1)
    deterministic_prob = np.array([1.0])
    
    deterministic_solution = optimizer.solve(
        deterministic_scenario, deterministic_prob, building_info, resource_constraints
    )
    
    # ========================
    # Step 5: Comparison
    # ========================
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    
    if stochastic_solution['status'] == 'optimal' and deterministic_solution['status'] == 'optimal':
        det_casualties = deterministic_solution['expected_casualties']
        sto_casualties = stochastic_solution['expected_casualties']
        
        improvement = (det_casualties - sto_casualties) / det_casualties * 100
        
        print(f"\nExpected Casualties:")
        print(f"  Deterministic (baseline):  {det_casualties:.1f} casualties")
        print(f"  Stochastic (ours):         {sto_casualties:.1f} casualties")
        print(f"  Improvement:                {improvement:.1f}%")
        
        print(f"\nObjective Value (Total Cost):")
        print(f"  Deterministic:  ${deterministic_solution['objective_value']:,.0f}")
        print(f"  Stochastic:     ${stochastic_solution['objective_value']:,.0f}")
        
        # Value of Stochastic Solution (VSS)
        vss = (deterministic_solution['objective_value'] - 
               stochastic_solution['objective_value'])
        vss_pct = vss / deterministic_solution['objective_value'] * 100
        
        print(f"\nValue of Stochastic Solution (VSS):")
        print(f"  VSS = ${vss:,.0f}  ({vss_pct:.1f}% improvement)")
        
        # Interpretation
        print(f"\nInterpretation:")
        if improvement > 15:
            print(f"  ✓ Significant improvement! Accounting for CV uncertainty")
            print(f"    leads to {improvement:.1f}% fewer expected casualties.")
        elif improvement > 5:
            print(f"  ✓ Moderate improvement. Uncertainty-aware allocation helps.")
        else:
            print(f"  ~ Marginal improvement. Scenarios may be similar.")
        
        # Resource allocation differences
        print(f"\nResource Allocation Insights:")
        det_alloc = deterministic_solution['stage1_allocation']
        sto_alloc = stochastic_solution['stage1_allocation']
        
        # Buildings with different allocations
        diff_buildings = np.sum(np.abs(det_alloc - sto_alloc) > 0.1, axis=1)
        n_diff = (diff_buildings > 0).sum()
        
        print(f"  - Buildings with different allocations: {n_diff} / {n_buildings}")
        
        # High uncertainty buildings get more flexible resources
        high_unc_buildings = np.where(uncertainties > np.median(uncertainties))[0]
        flexible_to_high_unc = sto_alloc[high_unc_buildings, 4].sum()  # Resource type 4 = flexible
        total_flexible = sto_alloc[:, 4].sum()
        
        if total_flexible > 0:
            pct_to_uncertain = flexible_to_high_unc / total_flexible * 100
            print(f"  - Flexible resources to high-uncertainty areas: {pct_to_uncertain:.1f}%")
            if pct_to_uncertain > 60:
                print(f"    ✓ Good hedging strategy! Deploying flexible resources to uncertain areas.")
        
        return {
            'stochastic': stochastic_solution,
            'deterministic': deterministic_solution,
            'improvement_pct': improvement,
            'vss': vss,
            'vss_pct': vss_pct
        }
    
    else:
        print("✗ One or both optimizations failed")
        return None


def simulate_cv_predictions(n_buildings, seed=42):
    """
    Simulate CV predictions with realistic uncertainty patterns
    
    Returns:
        predictions: (n_buildings, 4) probability distributions
        uncertainties: (n_buildings,) total uncertainty per building
    """
    np.random.seed(seed)
    
    # Generate predictions with varying uncertainty
    predictions = np.zeros((n_buildings, 4))
    
    for i in range(n_buildings):
        if i < n_buildings * 0.3:  # 30% high confidence predictions
            # Sharp distribution (low uncertainty)
            true_state = np.random.randint(0, 4)
            alpha = np.ones(4) * 0.5
            alpha[true_state] = 10
            predictions[i] = np.random.dirichlet(alpha)
        elif i < n_buildings * 0.7:  # 40% medium confidence
            # Moderate distribution
            true_state = np.random.randint(0, 4)
            alpha = np.ones(4) * 1
            alpha[true_state] = 5
            predictions[i] = np.random.dirichlet(alpha)
        else:  # 30% low confidence (high uncertainty)
            # Flat distribution (high uncertainty)
            alpha = np.ones(4) * 2
            predictions[i] = np.random.dirichlet(alpha)
    
    # Compute uncertainties
    uncertainties = np.var(predictions, axis=1)
    
    return predictions, uncertainties


def run_sensitivity_analysis():
    """
    Run sensitivity analysis on key parameters:
    1. Number of scenarios
    2. Building count
    3. CV accuracy levels
    """
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Test different scenario counts
    print("\n[Analysis 1] Impact of Scenario Count")
    print("-" * 50)
    
    scenario_counts = [10, 20, 50]
    n_buildings = 30
    
    for n_scenarios in scenario_counts:
        print(f"\nTesting with {n_scenarios} scenarios...")
        try:
            result = run_full_pipeline(
                n_buildings=n_buildings, 
                n_scenarios=n_scenarios,
                visualize=False
            )
            if result:
                print(f"  → Improvement: {result['improvement_pct']:.1f}%, VSS: {result['vss_pct']:.1f}%")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")


if __name__ == "__main__":
    print("\n🚨 DISASTER RESPONSE: CV UNCERTAINTY + STOCHASTIC OPTIMIZATION 🚨\n")
    
    # Run main pipeline
    results = run_full_pipeline(n_buildings=50, n_scenarios=20)
    
    # Uncomment to run sensitivity analysis
    # run_sensitivity_analysis()
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}\n")
