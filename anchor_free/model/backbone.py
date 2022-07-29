import torch
import torch.nn as nn


class TIMMBackbone(nn.Module):
    def __init__(self, backbone, multi_output=False):
        super().__init__()
        self.backbone = backbone
        self.multi_output = multi_output

    def get_output_channels(self):
        input = torch.zeros(size=(2, 3, 128, 128))
        if self.multi_output:
            output = self.backbone.forward(input)
        else:
            output = self.backbone.forward_features(input)
        if isinstance(output, torch.Tensor):
            channels = output.size()[1]
        else:
            channels = []
            for item in output:
                channels.append(item.size()[1])
        return channels

    def forward(self, x):
        if self.multi_output:
            return self.backbone.forward(x)
        else:
            return self.backbone.forward_features(x)
