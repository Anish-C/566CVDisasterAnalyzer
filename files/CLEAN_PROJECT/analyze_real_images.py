"""
Real Hurricane Image Analysis
Analyzes satellite imagery using ResNet-50 to classify building damage
Includes temperature scaling for softmax calibration and confidence thresholds for triage
Includes Grad-CAM for interpretability: visualizes which pixels influence decisions
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2


class HurricaneImageAnalyzer:
    """Analyze real hurricane satellite images for damage assessment"""
    
    def __init__(self, device='cpu', temperature=1.0, threshold=0.0):
        """
        Initialize with pre-trained ResNet-50
        
        Args:
            device: 'cpu' or 'cuda'
            temperature: Temperature for softmax scaling (>1 = softer, <1 = sharper).
                        Typical range 1.0-3.0 for domain-shifted models.
            threshold: Confidence threshold. Images below this flagged for review (0.0-1.0).
        """
        self.device = device
        self.temperature = temperature
        self.threshold = threshold
        self.damage_classes = ['Intact', 'Minor Damage', 'Major Damage', 'Destroyed']
        
        print("Loading ResNet-50 (ImageNet pretrained)...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Adapt final layer for 4-class damage classification
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)
        
        self.model.to(device)
        self.model.eval()
        
        # Image preprocessing (ImageNet standard)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("[OK] Model loaded")
        if self.temperature != 1.0:
            print(f"Temperature scaling: T={self.temperature} (softer probabilities)")
        if self.threshold > 0:
            print(f"Confidence threshold: {self.threshold:.1%} (images below will be flagged for review)")
        print("Note: Using ImageNet weights, NOT fine-tuned on disaster data\n")
    
    def analyze_image(self, image_path):
        """
        Classify single image and return prediction with confidence
        Includes Grad-CAM for interpretability
        
        Args:
            image_path: Path to satellite image
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            original_img = np.array(img)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(img_tensor)
                # Temperature scaling
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            
            # Check if below threshold
            below_threshold = confidence < self.threshold
            
            # Compute Grad-CAM for the predicted class
            gradcam_heatmap = self._compute_gradcam(img_tensor, predicted_class)
            
            return {
                'image_path': image_path,
                'original_image': original_img,
                'predicted_class': predicted_class,
                'predicted_damage': self.damage_classes[predicted_class],
                'confidence': confidence,
                'probabilities': probs,
                'is_disaster': predicted_class in [2, 3],  # Major or Destroyed
                'needs_review': below_threshold,
                'gradcam': gradcam_heatmap
            }
        
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def _compute_gradcam(self, img_tensor, target_class):
        """
        Compute Grad-CAM heatmap for the predicted class
        Uses layer4 (final residual block) of ResNet-50
        
        Args:
            img_tensor: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class index to compute gradient for
            
        Returns:
            Heatmap array (224, 224) normalized to [0, 1]
        """
        # Target layer (last block of layer4)
        target_layer = self.model.layer4[-1]

        activations = []
        gradients = []

        def fwd_hook(module, inp, out):
            activations.append(out)

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])  # grad_out is a tuple; take tensor

        handle_fwd = target_layer.register_forward_hook(fwd_hook)
        handle_bwd = target_layer.register_backward_hook(bwd_hook)

        # Forward
        logits = self.model(img_tensor)
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Backward on target class
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)

        # Remove hooks
        handle_fwd.remove()
        handle_bwd.remove()

        if not activations or not gradients:
            return np.zeros((224, 224), dtype=np.float32)

        # Extract tensors
        act = activations[0].squeeze(0)          # (C, H, W)
        grad = gradients[0].squeeze(0)           # (C, H, W)

        # Global average pooling on gradients to get channel weights
        weights = grad.mean(dim=(1, 2))          # (C,)

        # Weighted sum of activations
        heatmap = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (H, W)

        heatmap = heatmap.detach().cpu().numpy()

        # Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap)

        # Resize to (224, 224)
        heatmap = cv2.resize(heatmap, (224, 224))
        return heatmap
    
    def analyze_directory(self, directory_path):
        """Analyze all images in a directory"""
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = set()
        
        # Find all image files (case-insensitive)
        for ext in image_extensions:
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                for file in Path(directory_path).glob(pattern):
                    if file not in image_files:
                        image_files.add(file)
        
        image_files = sorted(list(image_files))
        
        if not image_files:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"Found {len(image_files)} images. Analyzing...\n")
        
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {img_path.name:50s}", end=" ")
            result = self.analyze_image(str(img_path))
            
            if result:
                results.append(result)
                status = "[DISASTER]" if result['is_disaster'] else "[OK]"
                review_flag = " [REVIEW]" if result['needs_review'] else ""
                print(f"{result['predicted_damage']:15s} ({result['confidence']:.1%}) {status}{review_flag}")
            else:
                print("ERROR")
        
        return results
    
    def print_summary(self, results):
        """Print analysis summary"""
        if not results:
            return
        
        disasters = [r for r in results if r['is_disaster']]
        safe = [r for r in results if not r['is_disaster']]
        review_needed = [r for r in results if r['needs_review']]
        
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total images: {len(results)}")
        print(f"Disasters detected: {len(disasters)} ({len(disasters)/len(results)*100:.1f}%)")
        print(f"Safe areas: {len(safe)} ({len(safe)/len(results)*100:.1f}%)")
        
        if review_needed:
            print(f"\n[REVIEW] Images below confidence threshold: {len(review_needed)} ({len(review_needed)/len(results)*100:.1f}%)")
            print(f"  Threshold: {self.threshold:.1%}")
            for r in review_needed:
                print(f"    - {Path(r['image_path']).name}: {r['confidence']:.1%}")
        
        print(f"\nConfidence statistics:")
        confidences = [r['confidence'] for r in results]
        print(f"  Mean: {np.mean(confidences):.1%}")
        print(f"  Min:  {np.min(confidences):.1%}")
        print(f"  Max:  {np.max(confidences):.1%}")
        
        if self.temperature != 1.0:
            print(f"\nTemperature: {self.temperature} (softmax calibration applied)")
        
        print(f"\nIMPORTANT: Low confidence indicates the model is uncertain about")
        print(f"  its predictions. This is expected when using an ImageNet-pretrained")
        print(f"  model on satellite imagery. Use --threshold to flag borderline cases.")
    
    def visualize_results(self, results, output_dir='analysis_results'):
        """Create summary visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary grid (6 images)
        n_images = min(len(results), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hurricane Damage Analysis - ResNet-50 Classification\n(ImageNet Pretrained)', 
                     fontsize=14, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < n_images:
                result = results[idx]
                ax.imshow(result['original_image'])
                
                # Color-code by damage level
                colors = {'Intact': 'green', 'Minor Damage': 'yellow', 
                         'Major Damage': 'orange', 'Destroyed': 'red'}
                color = colors.get(result['predicted_damage'], 'gray')
                
                title = f"{result['predicted_damage']}\n{result['confidence']:.1%} confidence"
                ax.set_title(title, fontsize=11, fontweight='bold', 
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, 'summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Summary saved: {summary_path}")
        
        # Create individual analyses
        for result in results:
            self._save_individual_analysis(result, output_dir)
    
    def _save_individual_analysis(self, result, output_dir):
        """Create detailed analysis for single image with Grad-CAM"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Left: Original image
        original = result['original_image']
        axes[0].imshow(original)
        axes[0].set_title(f"Input Image\n{Path(result['image_path']).name}", 
                         fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Middle: Grad-CAM heatmap overlay
        # Resize original to match heatmap size (224, 224)
        original_resized = cv2.resize(original, (224, 224))
        img_rgb = original_resized.astype(np.float32) / 255.0
        heatmap = result['gradcam']
        
        # Create colormap heatmap overlay
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        blended = 0.6 * img_rgb + 0.4 * heatmap_colored
        blended = np.clip(blended, 0, 1)
        
        axes[1].imshow(blended)
        axes[1].set_title('Grad-CAM: Model Attention\n(Yellow/Red = High Focus)', 
                         fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # Right: Predictions bar chart
        x_pos = np.arange(len(self.damage_classes))
        colors_bar = ['green', 'yellow', 'orange', 'red']
        bars = axes[2].bar(x_pos, result['probabilities'], color=colors_bar, alpha=0.7, edgecolor='black')
        
        # Highlight predicted class
        bars[result['predicted_class']].set_linewidth(3)
        
        axes[2].set_ylabel('Probability', fontsize=11, fontweight='bold')
        axes[2].set_title('Model Predictions (Softmax Probabilities)', fontsize=11, fontweight='bold')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(self.damage_classes, rotation=15, ha='right')
        axes[2].set_ylim([0, 1.0])
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add confidence annotation
        axes[2].text(0.5, 1.05, f"Predicted: {result['predicted_damage']} ({result['confidence']:.1%})",
                    transform=axes[2].transAxes, ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # Save with image name
        img_name = Path(result['image_path']).stem
        save_path = os.path.join(output_dir, f'{img_name}_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze hurricane satellite imagery for building damage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python analyze_real_images.py --images ./data
  python analyze_real_images.py --images ./data --temperature 2.0 --threshold 0.40
  python analyze_real_images.py --images ./data --out ./my_results
        '''
    )
    parser.add_argument('--images', '-i', default='data',
                       help='Directory containing satellite images (default: data)')
    parser.add_argument('--out', '-o', default='analysis_results',
                       help='Output directory for results (default: analysis_results)')
    parser.add_argument('--temperature', '-t', type=float, default=1.0,
                       help='Softmax temperature for calibration (default: 1.0, range: 0.1-5.0)')
    parser.add_argument('--threshold', '-th', type=float, default=0.0,
                       help='Confidence threshold for review flagging (default: 0.0, range: 0.0-1.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.1 <= args.temperature <= 5.0:
        print("ERROR: temperature must be between 0.1 and 5.0")
        return
    if not 0.0 <= args.threshold <= 1.0:
        print("ERROR: threshold must be between 0.0 and 1.0")
        return
    if not os.path.exists(args.images):
        print(f"ERROR: Directory {args.images} not found")
        return
    
    print("\n" + "="*70)
    print("HURRICANE SATELLITE IMAGE ANALYSIS")
    print("="*70 + "\n")
    print(f"Image directory: {args.images}")
    print(f"Output directory: {args.out}")
    print(f"Temperature: {args.temperature}")
    print(f"Threshold: {args.threshold:.2%}\n")
    
    # Analyze images
    analyzer = HurricaneImageAnalyzer(device='cpu', temperature=args.temperature, 
                                      threshold=args.threshold)
    results = analyzer.analyze_directory(args.images)
    
    if results:
        analyzer.print_summary(results)
        analyzer.visualize_results(results, output_dir=args.out)
        
        print("\n" + "="*70)
        print(f"[OK] Analysis complete. Results in: {args.out}/")
        print("="*70 + "\n")
    else:
        print("\nNo results to display")


if __name__ == "__main__":
    main()
