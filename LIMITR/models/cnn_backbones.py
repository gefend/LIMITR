import torch.nn as nn
from torchvision import models as models_2d


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


def resnet_50(pretrained=True):
    model = models_2d.resnet50(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024



