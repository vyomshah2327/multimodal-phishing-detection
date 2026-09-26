# models/image_model.py

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.resnet(x)
