import torch
import torch.nn as nn
from torchvision import models

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load pre-trained Swin-B model
        self.model = models.swin_b(pretrained=pretrained)
        
        # Replace the head with a new one for our number of classes
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_target_layer(self):
        """Returns the target layer for visualization"""
        return self.model.features[-1][-1].norm1