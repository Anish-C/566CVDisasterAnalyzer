"""
Real Image Analysis Script
Analyzes actual satellite/disaster images using pre-trained ResNet-50
Classifies damage levels and provides uncertainty estimates
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Import our Bayesian model
from models.bayesian_resnet import get_bayesian_resnet50

class RealImageAnalyzer:
    """Analyze real images for disaster damage assessment"""
    
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize analyzer with pre-trained model
        
        Args:
            model_path: Path to trained model weights (optional)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.damage_classes = ['Intact', 'Minor Damage', 'Major Damage', 'Destroyed']
        
        # Load pre-trained ResNet-50 (ImageNet weights)
        print("Loading pre-trained ResNet-50 model...")
        import torchvision.models as models
        self.model = models.resnet50(pretrained=True)
        # Modify final layer for damage classification
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)
        
        # If you have trained weights, load them
        if model_path and os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Using ImageNet pre-trained weights (not fine-tuned on disaster data)")
            print("Note: Results will be better with fine-tuned weights on real xBD data")
        
        self.model.to(device)
        self.model.eval()
        
        # Enable dropout for MC Dropout
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.apply(enable_dropout)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_path):
        """Load and validate image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def predict_with_uncertainty(self, image_path, n_samples=30):
        """
        Predict damage class with uncertainty using MC Dropout
        
        Args:
            image_path: Path to image file
            n_samples: Number of MC dropout samples
        
        Returns:
            dict with predictions, uncertainty, and confidence
        """
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return None
        
        # Store original for visualization
        original_img = img.copy()
        
        # Preprocess
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # MC Dropout: multiple forward passes
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Forward pass with dropout enabled
                logits = self.model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions_list.append(probs.cpu().numpy()[0])
        
        predictions_array = np.array(predictions_list)  # (n_samples, 4)
        
        # Compute statistics
        mean_pred = predictions_array.mean(axis=0)
        std_pred = predictions_array.std(axis=0)
        max_class = np.argmax(mean_pred)
        confidence = mean_pred[max_class]
        
        # Total uncertainty (entropy of mean + mean entropy)
        total_uncertainty = np.var(predictions_array, axis=0).mean()
        
        return {
            'image_path': image_path,
            'original_image': original_img,
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'predicted_class': max_class,
            'predicted_damage': self.damage_classes[max_class],
            'confidence': confidence,
            'uncertainty': total_uncertainty,
            'all_samples': predictions_array,
            'is_disaster': max_class >= 2  # Major damage or destroyed
        }
    
    def analyze_directory(self, directory_path, n_samples=30):
        """
        Analyze all images in a directory
        
        Args:
            directory_path: Path to folder with images
            n_samples: MC dropout samples per image
        
        Returns:
            List of results for all images
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = set()  # Use set to avoid duplicates
        
        for ext in image_extensions:
            image_files.update(Path(directory_path).glob(f'*{ext}'))
            image_files.update(Path(directory_path).glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(image_files))  # Convert back to sorted list
        
        if not image_files:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"\nFound {len(image_files)} unique images. Analyzing...")
        print("=" * 70)
        
        results = []
        for i, img_path in enumerate(sorted(image_files), 1):
            print(f"\n[{i}/{len(image_files)}] Analyzing: {img_path.name}")
            result = self.predict_with_uncertainty(str(img_path), n_samples=n_samples)
            
            if result:
                results.append(result)
                self._print_result(result)
        
        return results
    
    def _print_result(self, result):
        """Print formatted result for single image"""
        print(f"  Predicted Damage: {result['predicted_damage']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Uncertainty: {result['uncertainty']:.4f}")
        print(f"  Disaster Detected: {'YES ⚠️' if result['is_disaster'] else 'NO ✓'}")
        
        # Show probabilities for all classes
        print(f"  Class Probabilities:")
        for i, (label, prob, std) in enumerate(zip(
            self.damage_classes,
            result['mean_predictions'],
            result['std_predictions']
        )):
            print(f"    {label:15s}: {prob:.1%} ± {std:.1%}")
    
    def visualize_results(self, result, save_path=None):
        """Create visualization of prediction with uncertainty"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Original image
        ax1 = axes[0]
        ax1.imshow(result['original_image'])
        ax1.set_title(f"Input Image: {Path(result['image_path']).name}", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Add disaster indicator
        if result['is_disaster']:
            ax1.text(0.5, -0.05, '⚠️ DISASTER DETECTED', 
                    transform=ax1.transAxes, ha='center',
                    fontsize=14, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Right: Predictions with uncertainty
        ax2 = axes[1]
        
        x_pos = np.arange(len(self.damage_classes))
        colors = ['green', 'yellow', 'orange', 'red']
        
        bars = ax2.bar(x_pos, result['mean_predictions'], 
                      yerr=result['std_predictions'],
                      capsize=8, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=2)
        
        ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax2.set_title('Damage Classification with Uncertainty\n(MC Dropout, 30 samples)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.damage_classes, rotation=15, ha='right')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Highlight predicted class
        predicted_idx = result['predicted_class']
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(3)
        
        # Add confidence text
        ax2.text(0.5, 1.05, f"Confidence: {result['confidence']:.1%} | Uncertainty: {result['uncertainty']:.4f}",
                transform=ax2.transAxes, ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization: {save_path}")
        
        return fig
    
    def create_summary_report(self, results, output_dir='analysis_results'):
        """Create summary report of all analyzed images"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Count results by disaster status
        disasters = [r for r in results if r['is_disaster']]
        safe = [r for r in results if not r['is_disaster']]
        
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nTotal Images Analyzed: {len(results)}")
        print(f"Disasters Detected: {len(disasters)} ({len(disasters)/len(results)*100:.1f}%)")
        print(f"Safe Areas: {len(safe)} ({len(safe)/len(results)*100:.1f}%)")
        
        if disasters:
            print(f"\n⚠️ DISASTER IMAGES:")
            for r in disasters:
                print(f"  • {Path(r['image_path']).name}: {r['predicted_damage']} (confidence: {r['confidence']:.1%})")
        
        if safe:
            print(f"\n✓ SAFE IMAGES:")
            for r in safe:
                print(f"  • {Path(r['image_path']).name}: {r['predicted_damage']} (confidence: {r['confidence']:.1%})")
        
        # Create visualizations for each result
        print(f"\nCreating visualizations...")
        for r in results:
            img_name = Path(r['image_path']).stem
            save_path = os.path.join(output_dir, f"{img_name}_analysis.png")
            self.visualize_results(r, save_path)
        
        # Create combined summary figure
        self._create_summary_figure(results, os.path.join(output_dir, 'summary.png'))
        
        print(f"\n✓ Results saved to: {output_dir}/")


    def _create_summary_figure(self, results, save_path):
        """Create summary figure with all results"""
        n = len(results)
        if n == 0:
            return
        
        fig, axes = plt.subplots(2, min(3, n), figsize=(15, 10))
        if n == 1:
            axes = axes.reshape(2, 1)
        elif n <= 3:
            axes = axes.T
        
        fig.suptitle('Real Image Analysis Summary - Disaster Detection', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        for idx, result in enumerate(results[:6]):  # Show up to 6 images
            row = idx // 3
            col = idx % 3
            ax = axes[row, col] if n > 1 else axes[row]
            
            # Show image
            ax.imshow(result['original_image'])
            
            # Add title with disaster status
            status = "⚠️ DISASTER" if result['is_disaster'] else "✓ SAFE"
            title = f"{result['predicted_damage']}\n{status}"
            color = 'red' if result['is_disaster'] else 'green'
            
            ax.set_title(title, fontsize=11, fontweight='bold', color=color)
            ax.axis('off')
            
            # Add confidence
            ax.text(0.5, -0.08, f"Confidence: {result['confidence']:.1%}",
                   transform=ax.transAxes, ha='center', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n, 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved summary: {save_path}")
        plt.close()


def main():
    """Main analysis pipeline"""
    
    print("\n" + "="*70)
    print("REAL IMAGE DISASTER ANALYSIS")
    print("="*70)
    
    # Check if image directory provided
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        # Default: look for images in data/test_images or data/
        image_dir = Path('data/test_images')
        if not image_dir.exists():
            image_dir = Path('data')
        
        if not image_dir.exists():
            print("\n❌ ERROR: No image directory found!")
            print("\nUsage:")
            print("  python analyze_real_images.py <path_to_image_directory>")
            print("\nExample:")
            print("  python analyze_real_images.py ./my_satellite_images/")
            print("  python analyze_real_images.py data/test_images/")
            return
    
    # Initialize analyzer
    analyzer = RealImageAnalyzer(device='cpu')
    
    # Analyze all images in directory
    results = analyzer.analyze_directory(image_dir, n_samples=20)
    
    if results:
        # Create summary report
        analyzer.create_summary_report(results, output_dir='analysis_results')
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nResults saved to: analysis_results/")
        print("\nNext steps:")
        print("  1. Review analysis_results/summary.png")
        print("  2. Check individual *_analysis.png files")
        print("  3. Use these results in your presentation")
    else:
        print("\n❌ No images could be analyzed")


if __name__ == "__main__":
    main()
