import torch
import torch.nn as nn


class TIMMBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def get_output_channels(self):
        input = torch.zeros(size=(2, 3, 64, 64))
        output = self.backbone.forward_features(input)
        channels = output.size()[1]
        return channels

    def forward(self, x):
        return self.backbone.forward_features(x)
