
import torch
from torch import nn
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
except ImportError:
    weights = True

# Конечно, это можно сделать и с помощью оформления класса
class MyResNet(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()

        # подгружаем модель
        self.model = resnet18(weights=weights)
        # заменяем слой
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # замораживаем слои
        for i in self.model.parameters():
            i.requires_grad = False
        # размораживаем только последний, который будем обучать
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

# model = MyResNet()
# model.to(DEVICE);