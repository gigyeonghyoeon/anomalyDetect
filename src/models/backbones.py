"""Feature extractors for PatchCore (multi-scale patch features)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

XRV_WEIGHTS_URL = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)
XRV_WEIGHTS_NAME = (
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)


def _ensure_xrv_weights() -> None:
    """Download torchxrayvision CheXpert weights if missing or corrupted."""
    cache_dir = Path.home() / ".torchxrayvision" / "models_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / XRV_WEIGHTS_NAME
    if target.exists() and target.stat().st_size > 1_000_000:
        return
    if target.exists():
        target.unlink()
    print(f"Downloading CheXpert DenseNet weights to {target} ...")
    urllib.request.urlretrieve(XRV_WEIGHTS_URL, target)
    print("Download complete.")


class FeatureExtractor(nn.Module):
    """ResNet-style multi-scale extractor (layer2 + layer3)."""

    def __init__(self, backbone: str = "resnet18", in_channels: int = 3):
        super().__init__()
        self.backbone_name = backbone
        self.features: dict[str, torch.Tensor] = {}

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self._adapt_first_conv(net, in_channels)
            self.model = nn.Sequential(*list(net.children())[:-2])
            self._hook_resnet_layers(net)
        elif backbone == "wide_resnet50_2":
            net = self._load_timm_backbone("wide_resnet50_2", in_channels)
            self.model = nn.Sequential(*list(net.children())[:-2])
            self._hook_resnet_layers(net)
        elif backbone == "densenet121_chexpert":
            self.model, self._forward_fn = self._build_xrv_densenet(in_channels)
        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                "Use resnet18, wide_resnet50_2, or densenet121_chexpert."
            )

        self.eval()

    @staticmethod
    def _adapt_first_conv(net: nn.Module, in_channels: int) -> None:
        old = net.conv1
        net.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            if in_channels == 1:
                net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            elif in_channels == 3:
                net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))
            else:
                raise ValueError(f"Unsupported in_channels: {in_channels}")

    @staticmethod
    def _load_timm_backbone(name: str, in_channels: int) -> nn.Module:
        import timm

        net = timm.create_model(name, pretrained=True, features_only=False)
        FeatureExtractor._adapt_first_conv(net, in_channels)
        return net

    def _hook_resnet_layers(self, net: nn.Module) -> None:
        def hook(name: str):
            def fn(_m, _i, out):
                self.features[name] = out

            return fn

        net.layer2.register_forward_hook(hook("layer2"))
        net.layer3.register_forward_hook(hook("layer3"))

    def _build_xrv_densenet(self, in_channels: int):
        try:
            import torchxrayvision as xrv
        except ImportError as e:
            raise ImportError(
                "densenet121_chexpert requires torchxrayvision. "
                "Install with: pip install torchxrayvision"
            ) from e

        _ensure_xrv_weights()
        net = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.features = {}

        def hook(name: str):
            def fn(_m, _i, out):
                self.features[name] = out

            return fn

        net.features.denseblock2.register_forward_hook(hook("layer2"))
        net.features.denseblock3.register_forward_hook(hook("layer3"))

        if in_channels == 3:
            old = net.features.conv0
            net.features.conv0 = nn.Conv2d(
                3, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=False,
            )
            with torch.no_grad():
                net.features.conv0.weight.copy_(
                    old.weight.repeat(1, 3, 1, 1) / 3.0
                )

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            _ = net.features(x)
            f2 = self.features["layer2"]
            f3 = self.features["layer3"]
            f2 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([f2, f3], dim=1)

        return net.features, forward_fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "densenet121_chexpert":
            return self._forward_fn(x)

        _ = self.model(x)
        f2 = self.features["layer2"]
        f3 = self.features["layer3"]
        f2 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)
