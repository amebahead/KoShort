# ================================
# model.py
# ================================
import torch
import torch.nn as nn
from torchvision import models

class KeypointRegressor(nn.Module):
    def __init__(self, num_keypoints=48):
        super().__init__()
        self.backbone = models.efficientnet_v2_s()
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)