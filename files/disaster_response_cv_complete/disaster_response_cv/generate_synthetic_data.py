"""
Synthetic Disaster Image Generator
Creates realistic satellite imagery of buildings with damage annotations
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import os
from pathlib import Path


class DisasterImageGenerator:
    """Generate synthetic satellite images of buildings with damage"""
    
    def __init__(self, output_dir="data/synthetic_images", image_size=256):
        self.output_dir = output_dir
        self.image_size = image_size
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/pre_disaster").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/post_disaster").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/labels").mkdir(parents=True, exist_ok=True)
        
    def generate_dataset(self, num_images=100, num_buildings_per_image=15):
        """
        Generate synthetic satellite images with damage labels
        
        Args:
            num_images: Number of image pairs to generate
            num_buildings_per_image: Buildings per image
            
        Returns:
            List of (pre_image_path, post_image_path, damage_labels, uncertainty)
        """
        print(f"Generating {num_images} synthetic disaster image pairs...")
        
        dataset = []
        
        for img_idx in range(num_images):
            # Generate building locations and characteristics
            buildings = self._generate_buildings(num_buildings_per_image)
            
            # Create pre-disaster image (all intact)
            pre_disaster = self._draw_buildings(buildings, damage_state=None)
            
            # Generate random damage pattern
            damage_labels = np.random.choice(
                [0, 1, 2, 3],  # no damage, minor, major, destroyed
                size=num_buildings_per_image,
                p=[0.4, 0.3, 0.2, 0.1]  # realistic distribution
            )
            
            # Create post-disaster image
            post_disaster = self._draw_buildings(buildings, damage_state=damage_labels)
            
            # Add realistic noise and atmospheric effects
            pre_disaster = self._add_realistic_effects(pre_disaster)
            post_disaster = self._add_realistic_effects(post_disaster)
            
            # Save images
            pre_path = f"{self.output_dir}/pre_disaster/image_{img_idx:03d}.jpg"
            post_path = f"{self.output_dir}/post_disaster/image_{img_idx:03d}.jpg"
            label_path = f"{self.output_dir}/labels/image_{img_idx:03d}.npy"
            
            cv2.imwrite(pre_path, pre_disaster)
            cv2.imwrite(post_path, post_disaster)
            np.save(label_path, {
                'damage': damage_labels,
                'buildings': buildings,
                'uncertainty': self._estimate_uncertainty(damage_labels)
            }, allow_pickle=True)
            
            dataset.append((pre_path, post_path, damage_labels))
            
            if (img_idx + 1) % 20 == 0:
                print(f"  Generated {img_idx + 1}/{num_images} images")
        
        print(f"✓ Dataset complete: {num_images} images in {self.output_dir}")
        return dataset
    
    def _generate_buildings(self, num_buildings):
        """Generate random building positions and sizes"""
        buildings = []
        grid_size = int(np.sqrt(num_buildings)) + 1
        cell_size = self.image_size // (grid_size + 1)
        
        for i in range(num_buildings):
            row = i // grid_size
            col = i % grid_size
            
            # Random jitter to avoid perfect grid
            x = col * cell_size + cell_size // 2 + np.random.randint(-cell_size // 4, cell_size // 4)
            y = row * cell_size + cell_size // 2 + np.random.randint(-cell_size // 4, cell_size // 4)
            
            # Random building size
            size = np.random.randint(20, 45)
            
            buildings.append({
                'x': int(x),
                'y': int(y),
                'size': size,
                'rotation': np.random.uniform(-15, 15)
            })
        
        return buildings
    
    def _draw_buildings(self, buildings, damage_state=None):
        """
        Draw buildings on satellite image
        
        Args:
            buildings: List of building dicts
            damage_state: Array of damage labels (0=intact, 1=minor, 2=major, 3=destroyed)
        """
        # Create base image (green land background)
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8)
        img[:, :] = [34, 120, 34]  # Forest green
        
        # Add some grass texture
        for _ in range(200):
            x = np.random.randint(0, self.image_size)
            y = np.random.randint(0, self.image_size)
            cv2.circle(img, (x, y), np.random.randint(2, 5), 
                      (25 + np.random.randint(-10, 10), 100, 20), -1)
        
        # Draw buildings
        for i, building in enumerate(buildings):
            x, y, size = building['x'], building['y'], building['size']
            
            if damage_state is None:
                # Pre-disaster: all intact (light gray/white)
                color = (200, 200, 200)
                cv2.rectangle(img, (x - size // 2, y - size // 2),
                            (x + size // 2, y + size // 2), color, -1)
                # Add roof outline
                cv2.rectangle(img, (x - size // 2, y - size // 2),
                            (x + size // 2, y + size // 2), (100, 100, 100), 2)
            else:
                # Post-disaster: colored by damage
                damage = damage_state[i]
                
                if damage == 0:  # Intact
                    color = (200, 200, 200)
                    cv2.rectangle(img, (x - size // 2, y - size // 2),
                                (x + size // 2, y + size // 2), color, -1)
                    cv2.rectangle(img, (x - size // 2, y - size // 2),
                                (x + size // 2, y + size // 2), (100, 100, 100), 2)
                
                elif damage == 1:  # Minor damage
                    color = (200, 200, 100)  # Yellow
                    cv2.rectangle(img, (x - size // 2, y - size // 2),
                                (x + size // 2, y + size // 2), color, -1)
                    # Add some cracks
                    cv2.line(img, (x - size // 3, y - size // 2), 
                            (x + size // 3, y + size // 2), (100, 100, 50), 1)
                
                elif damage == 2:  # Major damage
                    color = (100, 165, 255)  # Orange
                    cv2.rectangle(img, (x - size // 2, y - size // 2),
                                (x + size // 2, y + size // 2), color, -1)
                    # Add visible damage
                    cv2.line(img, (x - size // 2, y - size // 2), 
                            (x + size // 2, y + size // 2), (50, 80, 200), 2)
                    cv2.line(img, (x + size // 2, y - size // 2), 
                            (x - size // 2, y + size // 2), (50, 80, 200), 2)
                
                else:  # Destroyed (3)
                    color = (50, 50, 200)  # Red
                    cv2.rectangle(img, (x - size // 2, y - size // 2),
                                (x + size // 2, y + size // 2), color, -1)
                    # Heavy damage pattern
                    for _ in range(3):
                        px = np.random.randint(x - size // 2, x + size // 2)
                        py = np.random.randint(y - size // 2, y + size // 2)
                        cv2.circle(img, (px, py), np.random.randint(3, 8), 
                                  (20, 20, 100), -1)
        
        return img
    
    def _add_realistic_effects(self, img):
        """Add atmospheric effects to make images look more realistic"""
        # Add slight blur (atmospheric haze)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Add noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add shadows
        h, w = img.shape[:2]
        shadow = np.ones((h, w, 3), dtype=np.uint8) * 255
        shadow = cv2.ellipse(shadow, (w // 3, h // 3), (w // 2, h // 2), 0, 0, 360, 0, -1)
        img = cv2.addWeighted(img, 0.9, shadow, 0.1, 0)
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _estimate_uncertainty(self, damage_labels):
        """Estimate CV uncertainty based on damage severity"""
        # Higher damage = more certain predictions (easier to see destruction)
        # Minor damage = more uncertain (could be shadows, damage, or unclear)
        uncertainty = np.array([0.05, 0.20, 0.12, 0.03])
        return uncertainty[damage_labels]


if __name__ == "__main__":
    # Generate synthetic dataset
    generator = DisasterImageGenerator(
        output_dir=Path(__file__).parent / "data" / "synthetic_images",
        image_size=256
    )
    
    dataset = generator.generate_dataset(num_images=100, num_buildings_per_image=15)
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Total buildings: {sum(len(labels) for _, _, labels in dataset)}")
    print(f"  Damage distribution:")
    all_damage = np.concatenate([labels for _, _, labels in dataset])
    for damage_type, label in enumerate(['Intact', 'Minor', 'Major', 'Destroyed']):
        count = np.sum(all_damage == damage_type)
        print(f"    {label}: {count} ({100*count/len(all_damage):.1f}%)")
