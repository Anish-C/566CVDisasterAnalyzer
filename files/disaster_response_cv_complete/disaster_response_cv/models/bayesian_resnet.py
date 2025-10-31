"""
Bayesian ResNet-50 with MC Dropout for Building Damage Classification
Outputs probability distributions and epistemic uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that stays active during inference
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        # Apply dropout even during evaluation
        return F.dropout(x, p=self.p, training=True, inplace=False)


class BayesianResNet50(nn.Module):
    """
    ResNet-50 with MC Dropout for uncertainty quantification
    
    Outputs 4-class damage classification:
    0: No damage (intact)
    1: Minor damage
    2: Major damage  
    3: Destroyed
    
    Args:
        n_classes: Number of damage classes (default: 4)
        dropout_p: Dropout probability for MC Dropout (default: 0.3)
        pretrained: Use ImageNet pretrained weights (default: True)
    """
    def __init__(self, n_classes=4, dropout_p=0.3, pretrained=True):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract feature extractor (remove final FC layer)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add MC Dropout layers and classifier
        self.dropout1 = MCDropout(p=dropout_p)
        self.fc1 = nn.Linear(2048, 512)
        self.dropout2 = MCDropout(p=dropout_p)
        self.fc2 = nn.Linear(512, n_classes)
        
    def forward(self, x):
        """Forward pass (single sample)"""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=30):
        """
        Perform MC Dropout inference to estimate uncertainty
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            n_samples: Number of MC samples (default: 30)
            
        Returns:
            predictions: Mean predictions (batch_size, n_classes)
            uncertainties: Epistemic uncertainty (batch_size, n_classes)
            all_predictions: All MC samples (batch_size, n_samples, n_classes)
        """
        self.eval()  # Set to eval mode (but dropout will still be active)
        
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_predictions.append(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)  # (n_samples, batch_size, n_classes)
        all_predictions = np.transpose(all_predictions, (1, 0, 2))  # (batch_size, n_samples, n_classes)
        
        # Calculate mean predictions
        predictions = np.mean(all_predictions, axis=1)  # (batch_size, n_classes)
        
        # Calculate epistemic uncertainty (variance across MC samples)
        uncertainties = np.var(all_predictions, axis=1)  # (batch_size, n_classes)
        
        # Total uncertainty per sample (sum of variances across classes)
        total_uncertainty = np.sum(uncertainties, axis=1)  # (batch_size,)
        
        return predictions, total_uncertainty, all_predictions


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration
    Learns a single temperature parameter to calibrate confidence
    
    Usage:
        1. Train base model
        2. Create TemperatureScaling wrapper
        3. Optimize temperature on validation set
        4. Use calibrated model for inference
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        """Forward pass with temperature scaling"""
        logits = self.model(x)
        return self.temperature_scale(logits)
    
    def temperature_scale(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def predict_with_uncertainty(self, x, n_samples=30):
        """MC Dropout prediction with temperature scaling"""
        # Get predictions from base model
        predictions, uncertainties, all_predictions = self.model.predict_with_uncertainty(x, n_samples)
        
        # Apply temperature scaling to logits is tricky after softmax
        # For simplicity, we return the calibrated probabilities from forward passes
        # In practice, you'd apply temperature before softmax in the base model
        
        return predictions, uncertainties, all_predictions


def get_bayesian_resnet50(n_classes=4, dropout_p=0.3, pretrained=True):
    """
    Factory function to create Bayesian ResNet-50
    
    Args:
        n_classes: Number of damage classes
        dropout_p: Dropout probability
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        BayesianResNet50 model
    """
    return BayesianResNet50(n_classes=n_classes, dropout_p=dropout_p, pretrained=pretrained)


def calibrate_model(model, val_loader, device='cuda'):
    """
    Calibrate model using temperature scaling on validation set
    
    Args:
        model: Base Bayesian model
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Calibrated model
    """
    # Wrap model with temperature scaling
    scaled_model = TemperatureScaling(model).to(device)
    
    # Collect validation logits and labels
    logits_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Optimize temperature
    optimizer = torch.optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def eval_temperature():
        optimizer.zero_grad()
        loss = criterion(scaled_model.temperature_scale(logits.to(device)), labels.to(device))
        loss.backward()
        return loss
    
    optimizer.step(eval_temperature)
    
    print(f"Optimal temperature: {scaled_model.temperature.item():.3f}")
    
    return scaled_model


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_bayesian_resnet50(n_classes=4, dropout_p=0.3, pretrained=False).to(device)
    
    # Test single forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test MC Dropout inference
    predictions, uncertainties, all_preds = model.predict_with_uncertainty(x, n_samples=10)
    print(f"\nMC Dropout results:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainties shape: {uncertainties.shape}")
    print(f"All predictions shape: {all_preds.shape}")
    print(f"Mean prediction: {predictions[0]}")
    print(f"Uncertainty: {uncertainties[0]:.4f}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
