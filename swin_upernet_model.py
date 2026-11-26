"""Swin-B UPerNet with a stride-2 residual segmentation head."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from residual_refinement import ResidualRefinement


CLASSIFIERS = ("decode_head.classifier", "auxiliary_head.classifier")


class SwinUPerNetResidual(nn.Module):
    def __init__(self, config, classes, initialize_pretrained):
        super().__init__()
        from transformers import UperNetConfig, UperNetForSemanticSegmentation

        architecture = UperNetConfig.from_dict(config["architecture"])
        core = UperNetForSemanticSegmentation(architecture)
        self.pretrained_report = None
        if initialize_pretrained:
            self.load_ade20k(core, Path(config["pretrained_dir"]) / "model.safetensors")

        self.backbone = core.backbone
        self.decode_head = core.decode_head
        self.auxiliary_head = core.auxiliary_head
        self.classifier = nn.Conv2d(self.decode_head.classifier.in_channels, classes, 1)
        self.decode_head.classifier = nn.Identity()
        self.auxiliary_head.classifier = nn.Conv2d(self.auxiliary_head.classifier.in_channels, classes, 1)
        for head in (self.classifier, self.auxiliary_head.classifier):
            nn.init.normal_(head.weight, std=architecture.initializer_range)
            nn.init.zeros_(head.bias)
        self.refinement = ResidualRefinement(architecture.hidden_size, classes)
        if config["gradient_checkpointing"]:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def load_ade20k(self, core, path):
        from safetensors.torch import load_file

        if not path.is_file():
            raise FileNotFoundError(f"Full ADE20K checkpoint required: {path}. Run prepare first.")
        weights = load_file(path, device="cpu")
        expected = core.state_dict()
        missing = sorted(expected.keys() - weights.keys())
        unexpected = sorted(weights.keys() - expected.keys())
        mismatched = {key: [list(weights[key].shape), list(expected[key].shape)]
                      for key in expected.keys() & weights.keys()
                      if weights[key].shape != expected[key].shape}
        report = dict(source=str(path), missing_keys=missing, unexpected_keys=unexpected,
                      mismatched_shapes=mismatched, replaced_classifiers=list(CLASSIFIERS),
                      checkpoint_tensors=len(weights))
        print(json.dumps(report, indent=2), flush=True)
        if missing or unexpected or mismatched:
            raise ValueError("ADE20K state dict does not match the full UPerNet architecture")
        core.load_state_dict(weights, strict=True)
        self.pretrained_report = report

    def forward(self, image):
        features = self.backbone(pixel_values=image).feature_maps
        semantic = self.decode_head(features)
        size = image.shape[-2:]
        main = F.interpolate(self.classifier(semantic), size=size, mode="bilinear", align_corners=False)
        main = main + self.refinement(image, semantic)
        if self.training:
            aux = F.interpolate(self.auxiliary_head(features), size=size, mode="bilinear", align_corners=False)
            return main, aux
        return main


def optimizer_groups(model, training):
    rates = {"encoder": training["encoder_lr"], "decoder": training["decoder_lr"],
             "new": training["new_lr"]}
    groups = {}
    seen = set()
    for module_name, module in model.named_modules():
        for leaf, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{leaf}"
            if not param.requires_grad or id(param) in seen:
                raise ValueError(f"Frozen or duplicate Swin parameter: {name}")
            seen.add(id(param))
            if name.startswith("backbone."):
                role = "encoder"
            elif name.startswith(("classifier.", "auxiliary_head.classifier.", "refinement.")):
                role = "new"
            else:
                role = "decoder"
            no_decay = (leaf in {"bias", "relative_position_bias_table"}
                        or isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)))
            group = groups.setdefault((role, no_decay), dict(
                params=[], lr=rates[role], weight_decay=0.0 if no_decay else training["weight_decay"],
                group_name=f"{role}_{'no_decay' if no_decay else 'decay'}"))
            group["params"].append(param)
    if seen != {id(param) for param in model.parameters()} or len(groups) != 6:
        raise ValueError("Optimizer groups do not partition all three Swin parameter families")
    return list(groups.values())
