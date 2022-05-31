if __name__ == "__main__":
    import torch
    from timm import create_model
    from model.backbone import TIMMBackbone
    from model.centernet import CenterNet

    model_configs = [
        # "densenet121",
        # "dpn68",
        "dla34",
        "cspresnet50",
        "vovnet39a",
        "res2next50",
        # "selecsls42",
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
        input = torch.zeros(size=(2, 3, 640, 640))
        output = model(input)
        channels = model.get_output_channels()
        assert (
            output.dim() == 4
            and output.size(0) == 2
            and output.size(1) == channels
            and output.size(2) == output.size(3)
        ), f"Wrong backbone output for model {cfg}: {output.size()}"
        head = CenterNet(num_classes=10, input_channels=channels)
        output = head.decoder(output)
        assert (
            head.classification_head(output).size()[2] == 160
            and head.offset_head(output).size()[2] == 160
            and head.size_head(output).size()[2] == 160
            and head.classification_head(output).size()[3] == 160
            and head.offset_head(output).size()[3] == 160
            and head.size_head(output).size()[3] == 160
        ), "Wrong dimensions in CenterNet"
