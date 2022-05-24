import torch
import torch.nn as nn
from utils.Vit import *


#  test Vit
class MyModel(nn.Module):
    def __init__(self, img_size=28, patch_size=4):
        super(MyModel, self).__init__()
        self.vit = Vit(img_size=28, patch_size=4)

    def forward(self, x):
        x = self.vit(x)
        return x


model = MyModel()
input_data = torch.randn(1, 3, 28, 28)
output = model(input_data)
print(output.shape)
