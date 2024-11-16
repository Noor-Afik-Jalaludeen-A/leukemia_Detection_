import torch.nn as nn
from torchvision.models import resnet18

def build_model(num_classes=3):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
