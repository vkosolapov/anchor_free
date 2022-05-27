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

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    import torch
    from timm import create_model

    model_configs = [
        "densenet121",
        "dpn68",
        "dla34",
        "cspresnet50",
        "vovnet39a",
        "res2next50",
        "selecsls42",
        "skresnet18",
        "resnest14d",
        "tresnet_m",
        "repvgg_a2",
        "nfnet_f0",
        "efficientnet_b0",
        "regnetx_002",
        # "hrnet_w18",
        "ghostnet_050",
        "convnext_tiny",
        # "vit_tiny_patch16_224",
        # "swin_base_patch4_window7_224",
        # "botnet26t_256",
        # "halonet26t",
        "lambda_resnet26t",
    ]
    for cfg in model_configs:
        model = create_model(cfg)
        model = TIMMBackbone(model)
        input = torch.zeros(size=(2, 3, 224, 224))
        output = model(input)
        assert (
            output.dim() == 4
            and output.size(0) == 2
            and output.size(2) == output.size(3)
        ), f"Wrong backbone output for model {cfg}"
