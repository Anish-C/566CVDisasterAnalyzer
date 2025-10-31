"""
Scenario Generation from CV Uncertainty
Generates probabilistic damage scenarios for stochastic optimization
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


class ScenarioGenerator:
    """
    Generate damage scenarios from CV model predictions and uncertainties
    
    Process:
    1. Sample N scenarios from CV probability distributions
    2. Reduce to K representative scenarios using K-means clustering
    3. Compute scenario probabilities
    """
    
    def __init__(self, n_samples=1000, n_scenarios=50, random_state=42):
        """
        Args:
            n_samples: Number of MC samples to generate
            n_scenarios: Number of representative scenarios to keep
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_scenarios = n_scenarios
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_scenarios(self, predictions, uncertainties, building_ids=None):
        """
        Generate scenarios from CV predictions
        
        Args:
            predictions: Array of shape (n_buildings, n_classes) - probability distributions
            uncertainties: Array of shape (n_buildings,) - total uncertainty per building
            building_ids: Optional building IDs
            
        Returns:
            scenarios: Array of shape (n_scenarios, n_buildings) - damage state per building per scenario
            probabilities: Array of shape (n_scenarios,) - probability of each scenario
            scenario_info: Dictionary with additional scenario information
        """
        n_buildings = predictions.shape[0]
        n_classes = predictions.shape[1]
        
        if building_ids is None:
            building_ids = np.arange(n_buildings)
        
        print(f"Generating {self.n_samples} damage scenarios for {n_buildings} buildings...")
        
        # Step 1: Sample scenarios from probability distributions
        raw_scenarios = self._sample_scenarios(predictions, n_buildings, n_classes)
        
        # Step 2: Reduce to representative scenarios using K-means
        print(f"Reducing to {self.n_scenarios} representative scenarios...")
        scenarios, cluster_labels, cluster_centers = self._reduce_scenarios(raw_scenarios)
        
        # Step 3: Compute scenario probabilities
        print("Computing scenario probabilities...")
        probabilities = self._compute_probabilities(raw_scenarios, scenarios, predictions, uncertainties)
        
        # Prepare scenario info
        scenario_info = {
            'n_buildings': n_buildings,
            'n_raw_samples': self.n_samples,
            'n_scenarios': self.n_scenarios,
            'building_ids': building_ids,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'coverage': self._compute_coverage(raw_scenarios, scenarios)
        }
        
        print(f"Scenario generation complete!")
        print(f"  - Coverage: {scenario_info['coverage']:.1%} of probability mass")
        print(f"  - Mean scenario probability: {probabilities.mean():.4f}")
        
        return scenarios, probabilities, scenario_info
    
    def _sample_scenarios(self, predictions, n_buildings, n_classes):
        """Sample scenarios from CV probability distributions"""
        scenarios = np.zeros((self.n_samples, n_buildings), dtype=int)
        
        for i in range(n_buildings):
            # Sample damage states from probability distribution
            damage_states = np.random.choice(
                n_classes, 
                size=self.n_samples, 
                p=predictions[i]
            )
            scenarios[:, i] = damage_states
            
        return scenarios
    
    def _reduce_scenarios(self, raw_scenarios):
        """Reduce scenarios using K-means clustering"""
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_scenarios, 
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(raw_scenarios)
        
        # Get representative scenarios (cluster centers rounded to nearest damage state)
        scenarios = np.round(kmeans.cluster_centers_).astype(int)
        
        # Ensure damage states are valid (0, 1, 2, 3)
        scenarios = np.clip(scenarios, 0, 3)
        
        return scenarios, cluster_labels, kmeans.cluster_centers_
    
    def _compute_probabilities(self, raw_scenarios, scenarios, predictions, uncertainties):
        """
        Compute scenario probabilities using cluster membership
        
        For each scenario (cluster), compute probability as:
        P(scenario) = (# samples in cluster) / (total samples)
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Compute distances between raw scenarios and cluster centers
        distances = euclidean_distances(raw_scenarios.astype(float), scenarios.astype(float))
        
        # Assign each raw scenario to nearest cluster
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Count samples per cluster
        probabilities = np.zeros(self.n_scenarios)
        for s in range(self.n_scenarios):
            probabilities[s] = np.sum(cluster_assignments == s)
        
        # Normalize to sum to 1
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            # Fallback: uniform distribution
            probabilities = np.ones(self.n_scenarios) / self.n_scenarios
        
        # Ensure no zero probabilities (add small epsilon)
        probabilities = np.maximum(probabilities, 1e-6)
        probabilities = probabilities / probabilities.sum()
        
        return probabilities
    
    def _compute_coverage(self, raw_scenarios, scenarios):
        """
        Compute what fraction of probability mass is covered by representative scenarios
        """
        covered = 0
        tolerance = int(0.2 * scenarios.shape[1])
        
        for raw_scenario in raw_scenarios:
            # Check if this raw scenario is close to any representative scenario
            for scenario in scenarios:
                distance = np.sum(raw_scenario != scenario)
                if distance <= tolerance:
                    covered += 1
                    break
        
        return covered / len(raw_scenarios)
    
    def visualize_scenarios(self, scenarios, probabilities, save_path=None):
        """
        Visualize scenario distributions
        
        Args:
            scenarios: Generated scenarios
            probabilities: Scenario probabilities
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scenario probability distribution
        ax = axes[0, 0]
        sorted_idx = np.argsort(probabilities)[::-1]
        ax.bar(range(len(probabilities)), probabilities[sorted_idx])
        ax.set_xlabel('Scenario (sorted by probability)')
        ax.set_ylabel('Probability')
        ax.set_title('Scenario Probability Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. Damage state distribution across scenarios
        ax = axes[0, 1]
        damage_counts = np.zeros((self.n_scenarios, 4))
        for s in range(self.n_scenarios):
            for damage_state in range(4):
                damage_counts[s, damage_state] = np.sum(scenarios[s] == damage_state)
        
        x = np.arange(4)
        width = 0.8
        bottom = np.zeros(4)
        colors = ['green', 'yellow', 'orange', 'red']
        labels = ['No Damage', 'Minor', 'Major', 'Destroyed']
        
        for s in range(min(10, self.n_scenarios)):  # Show top 10 scenarios
            if probabilities[s] > 0.01:  # Only show scenarios with >1% probability
                ax.bar(x, damage_counts[s], width, bottom=bottom, 
                      label=f'Scenario {s+1} ({probabilities[s]:.1%})', alpha=0.7)
                bottom += damage_counts[s]
        
        ax.set_xlabel('Damage State')
        ax.set_ylabel('Number of Buildings')
        ax.set_title('Damage Distribution by Scenario (Top 10)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Expected damage distribution
        ax = axes[1, 0]
        expected_damage = np.zeros(4)
        for s in range(self.n_scenarios):
            for damage_state in range(4):
                expected_damage[damage_state] += probabilities[s] * np.sum(scenarios[s] == damage_state)
        
        ax.bar(x, expected_damage, color=colors, edgecolor='black', linewidth=2)
        ax.set_xlabel('Damage State')
        ax.set_ylabel('Expected Number of Buildings')
        ax.set_title('Expected Damage Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Scenario diversity (Hamming distance heatmap)
        ax = axes[1, 1]
        # Compute pairwise Hamming distances between top scenarios
        top_n = min(20, self.n_scenarios)
        top_scenarios = scenarios[sorted_idx[:top_n]]
        distances = np.zeros((top_n, top_n))
        for i in range(top_n):
            for j in range(top_n):
                distances[i, j] = np.sum(top_scenarios[i] != top_scenarios[j]) / len(scenarios[0])
        
        im = ax.imshow(distances, cmap='viridis', aspect='auto')
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Scenario Index')
        ax.set_title(f'Scenario Diversity (Top {top_n} Scenarios)')
        plt.colorbar(im, ax=ax, label='Normalized Hamming Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test scenario generation
    print("Testing Scenario Generation...")
    
    # Simulate CV predictions for 100 buildings
    n_buildings = 100
    n_classes = 4
    
    # Create synthetic predictions (with some buildings more uncertain)
    predictions = np.random.dirichlet(np.ones(n_classes) * 2, size=n_buildings)
    # Add high uncertainty to some buildings
    predictions[20:30] = np.random.dirichlet(np.ones(n_classes) * 0.5, size=10)
    
    # Simulate uncertainties
    uncertainties = np.array([np.var(pred) for pred in predictions])
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Mean prediction: {predictions.mean(axis=0)}")
    print(f"Mean uncertainty: {uncertainties.mean():.4f}")
    
    # Generate scenarios
    generator = ScenarioGenerator(n_samples=1000, n_scenarios=50)
    scenarios, probabilities, info = generator.generate_scenarios(predictions, uncertainties)
    
    print(f"\nGenerated scenarios shape: {scenarios.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probability sum: {probabilities.sum():.6f}")
    print(f"Max scenario probability: {probabilities.max():.4f}")
    print(f"Min scenario probability: {probabilities.min():.6f}")
    
    # Visualize
    # generator.visualize_scenarios(scenarios, probabilities, save_path='scenarios_viz.png')
