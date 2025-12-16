"""
Fine-tune ResNet-50 on labeled disaster damage data (binary: damage vs no_damage)
Trains on archive dataset (10k images: 5k damaged, 5k no_damage)
Should complete in 30-60 minutes on CPU
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from datetime import datetime


class DisasterDamageFineTuner:
    """Fine-tune ResNet-50 on disaster damage classification"""
    
    def __init__(self, device='cpu', num_epochs=10, batch_size=32, learning_rate=0.001):
        """
        Args:
            device: 'cpu' or 'cuda'
            num_epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
        """
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Device: {device}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}\n")
    
    def load_data(self, data_path, val_split=0.2):
        """Load ImageFolder dataset and create train/val splits"""
        
        print(f"Loading data from {data_path}...")
        
        # Image preprocessing (ImageNet standard)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load all images (expects folder structure: data_path/class1, data_path/class2, ...)
        dataset = ImageFolder(root=data_path, transform=transform)
        num_samples = len(dataset)
        val_size = int(val_split * num_samples)
        train_size = num_samples - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.class_names = dataset.classes
        self.num_classes = len(dataset.classes)
        
        print(f"Classes: {self.class_names}")
        print(f"Total samples: {num_samples}")
        print(f"Train: {train_size}, Val: {val_size}\n")
        
        return self.class_names
    
    def build_model(self):
        """Load ResNet-50 and adapt for binary/multi-class classification"""
        
        print("Loading ResNet-50 (ImageNet pretrained)...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        
        self.model.to(self.device)
        print(f"[OK] Model ready. Final layer: {in_features} -> {self.num_classes}\n")
    
    def train(self):
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        print("="*70)
        print(f"TRAINING START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        best_val_acc = 0
        
        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}] Batch {batch_idx+1} Loss: {loss.item():.4f}")
            
            train_loss /= len(self.train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(self.val_loader)
            val_acc = 100 * correct / total
            
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.2f}%\n")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'finetuned_resnet50.pt')
                print(f"  [SAVED] Best model (accuracy: {val_acc:.2f}%)\n")
            
            scheduler.step()
        
        print("="*70)
        print(f"TRAINING COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print("="*70 + "\n")
    
    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history['val_acc'], label='Val Accuracy', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Training history saved: {save_path}\n")
    
    def save_model(self, path='finetuned_resnet50.pt'):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"[OK] Model saved: {path}\n")


def main():
    """Main fine-tuning pipeline"""
    
    # Dataset path
    data_path = '../../archive (1)/train_another'
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}")
        print("Expected structure:")
        print("  archive/train_another/")
        print("    damage/        (5000 images)")
        print("    no_damage/     (5000 images)")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Initialize
    tuner = DisasterDamageFineTuner(
        device=device,
        num_epochs=10,           # Adjust based on time/accuracy tradeoff
        batch_size=32,           # Reduce to 16 if OOM
        learning_rate=0.001
    )
    
    # Load data
    classes = tuner.load_data(data_path, val_split=0.2)
    
    # Build model
    tuner.build_model()
    
    # Train
    tuner.train()
    
    # Plot and save
    tuner.plot_history()
    tuner.save_model()
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Verify 'finetuned_resnet50.pt' was created")
    print("2. Update analyze_real_images.py to load this model")
    print("3. Run: python analyze_real_images.py --images ./data --finetuned")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
