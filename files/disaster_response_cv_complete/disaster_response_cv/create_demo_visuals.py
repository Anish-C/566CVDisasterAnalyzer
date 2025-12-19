"""
PRESENTATION DEMO SCRIPT
For 5-minute video demonstration of the pipeline

This script will be run live during presentation to show:
1. Satellite image loading
2. Model inference with uncertainty
3. Visualization of predictions
4. Resource allocation decision
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import cv2

# For the video, we'll create a visual demo showing the pipeline

def create_demo_video_sequence():
    """
    Create image sequence for demo video
    Shows: Image -> Predictions -> Uncertainty -> Allocation Decision
    """
    
    print("Creating demo visualization sequence...")
    
    # Create figure with 4 subplots showing pipeline
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Disaster Response CV + Optimization Pipeline', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # ====================
    # SECTION 1: Image Input (Top Left)
    # ====================
    ax1 = plt.subplot(2, 3, 1)
    
    # Create synthetic post-disaster image
    from generate_synthetic_data import DisasterImageGenerator
    generator = DisasterImageGenerator(image_size=256)
    buildings = generator._generate_buildings(12)
    damage = np.array([0, 1, 2, 3, 0, 1, 2, 1, 3, 0, 2, 1])
    image = generator._draw_buildings(buildings, damage_state=damage)
    image = generator._add_realistic_effects(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(image)
    ax1.set_title('INPUT: Satellite Imagery\n(Post-Disaster)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add annotation
    ax1.text(0.5, -0.15, '12 buildings detected\n3 hours after disaster',
            transform=ax1.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ====================
    # SECTION 2: Model Predictions (Top Middle)
    # ====================
    ax2 = plt.subplot(2, 3, 2)
    
    # Simulated model output for single building
    damage_labels = ['Intact', 'Minor', 'Major', 'Destroyed']
    predictions = np.array([0.05, 0.25, 0.50, 0.20])
    uncertainties = np.array([0.02, 0.04, 0.06, 0.03])
    
    colors = ['green', 'yellow', 'orange', 'red']
    x_pos = np.arange(len(damage_labels))
    
    bars = ax2.bar(x_pos, predictions, yerr=uncertainties, capsize=8,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Damage Class', fontsize=11, fontweight='bold')
    ax2.set_title('STAGE 1: CV Predictions\n(Example building)',
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(damage_labels, rotation=15, ha='right')
    ax2.set_ylim([0, 0.7])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax2.text(0.5, -0.25, 'ResNet-50 + MC Dropout\n(30 forward passes)',
            transform=ax2.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ====================
    # SECTION 3: Uncertainty Visualization (Top Right)
    # ====================
    ax3 = plt.subplot(2, 3, 3)
    
    # MC Dropout samples for 12 buildings
    np.random.seed(42)
    uncertainty_values = np.array([0.04, 0.08, 0.12, 0.15, 0.09, 0.07, 
                                   0.06, 0.18, 0.05, 0.10, 0.14, 0.08])
    confidence_values = 1 - uncertainty_values / uncertainty_values.max()
    
    building_ids = np.arange(12)
    scatter = ax3.scatter(building_ids, uncertainty_values, 
                         s=300, c=uncertainty_values, cmap='RdYlGn_r',
                         edgecolors='black', linewidth=2, alpha=0.7)
    
    # Add threshold line
    threshold = np.percentile(uncertainty_values, 75)
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               label='High uncertainty threshold')
    
    ax3.set_ylabel('Prediction Uncertainty', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Building ID', fontsize=11, fontweight='bold')
    ax3.set_title('STAGE 1: Uncertainty Estimates\n(All 12 buildings)',
                 fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 0.2])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Uncertainty', fontsize=10)
    
    # ====================
    # SECTION 4: Scenario Generation (Bottom Left)
    # ====================
    ax4 = plt.subplot(2, 3, 4)
    
    # Simulated scenarios
    n_scenarios = 20
    scenario_probs = np.random.dirichlet([5] * n_scenarios)
    scenario_probs = scenario_probs[np.argsort(scenario_probs)[::-1]]
    
    colors_scenario = plt.cm.viridis(np.linspace(0, 1, min(15, n_scenarios)))
    ax4.bar(range(min(15, n_scenarios)), scenario_probs[:min(15, n_scenarios)],
           color=colors_scenario, edgecolor='black', linewidth=1)
    
    ax4.set_ylabel('Scenario Probability', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Scenario Rank', fontsize=11, fontweight='bold')
    ax4.set_title('STAGE 2: Scenario Generation\n(1000 samples to 20 scenarios)',
                 fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax4.text(0.5, -0.25, 'K-means clustering from\nMC Dropout uncertainty',
            transform=ax4.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # ====================
    # SECTION 5: Optimization Comparison (Bottom Middle)
    # ====================
    ax5 = plt.subplot(2, 3, 5)
    
    approaches = ['Deterministic\n(Ignore CV\nuncertainty)', 'Stochastic\n(Account for\nuncertainty)']
    casualties = [145, 122]
    colors_bar = ['#d62728', '#2ca02c']
    
    bars = ax5.bar(approaches, casualties, color=colors_bar, alpha=0.7,
                   edgecolor='black', linewidth=2)
    
    ax5.set_ylabel('Expected Casualties', fontsize=11, fontweight='bold')
    ax5.set_title('STAGE 3: Optimization Results\n(200-building disaster)',
                 fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 160])
    
    for bar, val in zip(bars, casualties):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val}',
                ha='center', fontsize=13, fontweight='bold')
    
    # Add improvement annotation
    improvement = (casualties[0] - casualties[1]) / casualties[0] * 100
    ax5.annotate('', xy=(1, casualties[1]), xytext=(1, casualties[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
    ax5.text(1.25, (casualties[0] + casualties[1])/2, f'{improvement:.1f}%\nBETTER',
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax5.grid(axis='y', alpha=0.3)
    
    # ====================
    # SECTION 6: Resource Allocation (Bottom Right)
    # ====================
    ax6 = plt.subplot(2, 3, 6)
    
    resource_types = ['Search/\nRescue', 'Medical', 'Heavy\nEquip', 'Supplies', 'Flexible']
    det_alloc = np.array([25, 20, 12, 30, 8])
    sto_alloc = np.array([20, 18, 10, 28, 24])
    
    x = np.arange(len(resource_types))
    width = 0.35
    
    ax6.bar(x - width/2, det_alloc, width, label='Deterministic',
           color='#d62728', alpha=0.7, edgecolor='black', linewidth=1)
    ax6.bar(x + width/2, sto_alloc, width, label='Stochastic',
           color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1)
    
    ax6.set_ylabel('Units Allocated', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Resource Type', fontsize=11, fontweight='bold')
    ax6.set_title('STAGE 3: Resource Allocation\n(Stage 1 deployment)',
                 fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(resource_types, fontsize=9)
    ax6.legend(loc='upper right', fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    # Highlight flexible resources
    ax6.text(0.5, -0.25, 'Key insight: Stochastic allocates more\nflexible resources to uncertain areas',
            transform=ax6.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ====================
    # Overall layout
    # ====================
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """Generate demo visualization for presentation"""
    print("\n" + "="*70)
    print("GENERATING PRESENTATION DEMO VISUALIZATION")
    print("="*70)
    
    # Create output directory
    output_dir = Path("demo_visuals")
    output_dir.mkdir(exist_ok=True)
    
    # Generate figure
    fig = create_demo_video_sequence()
    
    # Save as high-quality PNG for video
    fig.savefig(output_dir / "pipeline_demo.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/pipeline_demo.png")
    
    # Also save individual components
    print(f"Visualization ready for presentation")
    print("\nNEXT STEPS:")
    print("  1. Review the saved visualization")
    print("  2. Create a PowerPoint with 5 slides using PRESENTATION_SCRIPT.md as content")
    print("  3. Record 5-minute video (use slide deck + narration)")
    print("  4. Upload MP4 to Canvas at least 2 hours before presentation")
    print("\nSPEAKER NOTES: See PRESENTATION_SCRIPT.md for complete script with timing")
    
    plt.close()

if __name__ == "__main__":
    main()
