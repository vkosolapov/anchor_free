import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d


class TIMMBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        children = list(backbone.children())
        children.reverse()
        for idx, module in enumerate(children):
            sub_children = list(module.children())
            if (
                isinstance(module, SelectAdaptivePool2d)
                or len(sub_children) > 0
                and isinstance(sub_children[0], SelectAdaptivePool2d)
            ):
                backbone = nn.Sequential(*list(backbone.children())[: -(idx + 1)])
                break
        self.backbone = backbone

    def get_output_channels(self):
        channels = 0
        modules = list(self.backbone.modules())
        modules.reverse()
        for module in modules:
            if isinstance(module, nn.Conv2d):
                channels = module.out_channels
                break
        return channels

    def forward(self, x):
        return self.backbone(x)
