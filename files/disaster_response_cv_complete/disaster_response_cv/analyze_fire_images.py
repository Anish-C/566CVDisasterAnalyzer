"""
Fire Detection and Highlighting Script
Analyzes satellite images from Kaggle fire dataset
Highlights fire regions using color segmentation and heat detection
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class FireDetector:
    """Detect and highlight fire regions in satellite imagery"""
    
    def __init__(self):
        self.fire_detected = False
        self.fire_confidence = 0.0
    
    def detect_fire(self, image_path):
        """
        Detect fire in satellite image using color-based segmentation
        Fire appears as bright red/orange/yellow in satellite imagery
        
        Args:
            image_path: Path to satellite image
        
        Returns:
            dict with fire detection results
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load {image_path}")
            return None
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original = img_rgb.copy()
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define fire color ranges in HSV
        # Red-orange-yellow colors typical of fire
        # Range 1: Red (wraps around HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Range 2: Orange
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Range 3: Yellow (bright areas)
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine all fire color masks
        fire_mask = mask_red1 | mask_red2 | mask_orange | mask_yellow
        
        # Also check for very bright regions (hotspots)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Combine color-based and brightness-based detection
        combined_mask = cv2.bitwise_or(fire_mask, bright_mask)
        
        # Remove noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of fire regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours (noise)
        min_area = img.shape[0] * img.shape[1] * 0.001  # 0.1% of image
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Calculate fire metrics
        total_fire_pixels = cv2.countNonZero(combined_mask)
        total_pixels = img.shape[0] * img.shape[1]
        fire_percentage = (total_fire_pixels / total_pixels) * 100
        
        fire_detected = len(significant_contours) > 0 and fire_percentage > 0.5
        
        # Create highlighted version
        highlighted = original.copy()
        
        if fire_detected:
            # Draw contours
            cv2.drawContours(highlighted, significant_contours, -1, (255, 0, 0), 3)
            
            # Create overlay with semi-transparent red
            overlay = original.copy()
            cv2.drawContours(overlay, significant_contours, -1, (255, 50, 50), -1)
            highlighted = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
            
            # Draw bounding boxes around fire regions
            for contour in significant_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Add label
                area = cv2.contourArea(contour)
                label = f"FIRE: {area:.0f}px²"
                cv2.putText(highlighted, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return {
            'image_path': image_path,
            'original': original,
            'highlighted': highlighted,
            'fire_mask': combined_mask,
            'fire_detected': fire_detected,
            'fire_percentage': fire_percentage,
            'num_fire_regions': len(significant_contours),
            'fire_contours': significant_contours,
            'total_fire_pixels': total_fire_pixels
        }
    
    def visualize_detection(self, result, save_path=None):
        """Create visualization of fire detection"""
        
        if result is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        ax1 = axes[0]
        ax1.imshow(result['original'])
        ax1.set_title('Original Satellite Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Fire mask
        ax2 = axes[1]
        ax2.imshow(result['fire_mask'], cmap='hot')
        ax2.set_title('Fire Detection Mask\n(Red/Orange/Yellow + Bright regions)', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Highlighted image
        ax3 = axes[2]
        ax3.imshow(result['highlighted'])
        
        if result['fire_detected']:
            title = f"🔥 FIRE DETECTED 🔥\n{result['fire_percentage']:.2f}% of image | {result['num_fire_regions']} regions"
            color = 'red'
        else:
            title = "✓ No Fire Detected"
            color = 'green'
        
        ax3.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
        
        return fig
    
    def create_summary(self, results, output_dir='fire_analysis_results'):
        """Create summary of all analyzed images"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("FIRE DETECTION SUMMARY")
        print("="*70)
        
        fire_images = [r for r in results if r['fire_detected']]
        safe_images = [r for r in results if not r['fire_detected']]
        
        print(f"\nTotal Images Analyzed: {len(results)}")
        print(f"🔥 Fire Detected: {len(fire_images)} ({len(fire_images)/len(results)*100:.1f}%)")
        print(f"✓ No Fire: {len(safe_images)} ({len(safe_images)/len(results)*100:.1f}%)")
        
        if fire_images:
            print(f"\n🔥 FIRE IMAGES:")
            for r in fire_images:
                img_name = Path(r['image_path']).name
                print(f"  • {img_name}")
                print(f"    - Fire coverage: {r['fire_percentage']:.2f}%")
                print(f"    - Fire regions: {r['num_fire_regions']}")
                print(f"    - Total fire pixels: {r['total_fire_pixels']:,}")
        
        if safe_images:
            print(f"\n✓ SAFE IMAGES:")
            for r in safe_images:
                print(f"  • {Path(r['image_path']).name}")
        
        # Create visualizations
        print(f"\nCreating visualizations...")
        for r in results:
            img_name = Path(r['image_path']).stem
            save_path = os.path.join(output_dir, f"{img_name}_fire_detection.png")
            self.visualize_detection(r, save_path)
        
        print(f"\n✓ All results saved to: {output_dir}/")


def main():
    """Main fire detection pipeline"""
    
    print("\n" + "="*70)
    print("FIRE DETECTION FROM SATELLITE IMAGERY")
    print("="*70)
    
    # Check if image directory provided
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        # Default: look for images in data/ or current directory
        possible_dirs = [
            Path('data/fire_images'),
            Path('data'),
            Path('.'),
        ]
        
        image_dir = None
        for d in possible_dirs:
            if d.exists():
                # Check if it has images
                image_files = list(d.glob('*.png')) + list(d.glob('*.jpg')) + list(d.glob('*.jpeg'))
                if image_files:
                    image_dir = d
                    break
        
        if image_dir is None:
            print("\n❌ ERROR: No images found!")
            print("\nUsage:")
            print("  python analyze_fire_images.py <path_to_image_directory>")
            print("\nExample:")
            print("  python analyze_fire_images.py ./fire_data/")
            print("  python analyze_fire_images.py data/")
            return
    
    image_dir = Path(image_dir)
    
    # Find images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"\n❌ No images found in {image_dir}")
        return
    
    # Limit to first 3 images as requested
    image_files = sorted(image_files)[:3]
    
    print(f"\nAnalyzing {len(image_files)} images from: {image_dir}")
    print("="*70)
    
    # Initialize detector
    detector = FireDetector()
    
    # Analyze each image
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Analyzing: {img_path.name}")
        result = detector.detect_fire(img_path)
        
        if result:
            results.append(result)
            status = "🔥 FIRE DETECTED" if result['fire_detected'] else "✓ No fire"
            print(f"  Status: {status}")
            if result['fire_detected']:
                print(f"  Fire Coverage: {result['fire_percentage']:.2f}%")
                print(f"  Fire Regions: {result['num_fire_regions']}")
    
    # Create summary
    if results:
        detector.create_summary(results, output_dir='fire_analysis_results')
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nResults saved to: fire_analysis_results/")
        print("\nNext steps:")
        print("  1. Review fire_analysis_results/*_fire_detection.png")
        print("  2. Use these highlighted images in your presentation")
        print("  3. Show fire detection capability as part of disaster CV")


if __name__ == "__main__":
    main()
