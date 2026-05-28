from __future__ import annotations

import torch
import torch.nn as nn

from openprompt_rs.data.structures import generate_query_centers


class TinyBackbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class ResNetBackbone(nn.Module):
    """Pretrained ResNet-50 backbone for scaffold detector."""

    def __init__(
        self,
        output_dim: int = 256,
        pretrained: bool = True,
        freeze_bn: bool = True,
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        if output_stride == 16:
            self.layer3 = resnet.layer3
            self._layers = [self.conv1, self.bn1, self.relu, self.maxpool,
                            self.layer1, self.layer2, self.layer3]
            in_dim = 1024
        else:
            self._layers = [self.conv1, self.bn1, self.relu, self.maxpool,
                            self.layer1, self.layer2]
            in_dim = 512
        self._output_stride = output_stride
        self.proj = nn.Conv2d(in_dim, output_dim, kernel_size=1)
        self.output_dim = output_dim

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self._output_stride == 16:
            x = self.layer3(x)
        return self.proj(x)


class QueryGenerator(nn.Module):
    def __init__(self, feature_dim: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = feature_map.shape[0]
        pooled = self.proj(self.pool(feature_map))
        query_tokens = pooled.flatten(2).transpose(1, 2)
        scene_feature = feature_map.mean(dim=(-1, -2))
        query_centers = generate_query_centers(self.grid_size, batch_size=batch_size, device=feature_map.device)
        return query_tokens, scene_feature, query_centers

