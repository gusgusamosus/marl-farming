import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()
    
    def forward(self, x):
        return self.model(x)  # Output embedding shape [batch, 512]
