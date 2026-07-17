# models.py
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet34Backbone(nn.Module):
    """
    ResNet34 split:
      - forward_to_layer1(x) -> layer1 feature map
      - forward_from_layer1(x1) -> pooled vector (512)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.out_dim = 512

    def forward_to_layer1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        return x1

    def forward_from_layer1(self, x1: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (N,512)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.forward_to_layer1(x)
        return self.forward_from_layer1(x1)


class MVCNN(nn.Module):
    """
    MVCNN-style multi-view classifier.
    - Shared ResNet34 backbone per-view
    - Pooling across views: mean (default) or max
    Exposes:
      - forward_with_feat(x) -> (logits, pooled_feat)
      - encode_views_layer1 / encode_views_from_layer1 : for S2Mix
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, pool: str = "mean"):
        super().__init__()
        assert pool in ["mean", "max"]
        self.pool = pool
        self.backbone = ResNet34Backbone(pretrained=pretrained)
        self.feat_dim = self.backbone.out_dim
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def encode_views(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,V,3,H,W) -> (B,V,512)
        B, V, C, H, W = x.shape
        xv = x.view(B * V, C, H, W)
        fv = self.backbone(xv).view(B, V, -1)
        return fv

    def aggregate(self, view_feats: torch.Tensor) -> torch.Tensor:
        # view_feats: (B,V,D) -> (B,D)
        if self.pool == "mean":
            return view_feats.mean(dim=1)
        else:
            return view_feats.max(dim=1).values

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        vf = self.encode_views(x)
        return self.aggregate(vf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encode(x)
        return self.classifier(feat)

    def forward_with_feat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode(x)
        logits = self.classifier(feat)
        return logits, feat

    # ---- for S2Mix (style mixing at early feature) ----
    def encode_views_layer1(self, x: torch.Tensor) -> torch.Tensor:
        # (B,V,3,H,W) -> (B,V,C1,H1,W1)
        B, V, C, H, W = x.shape
        xv = x.view(B * V, C, H, W)
        x1 = self.backbone.forward_to_layer1(xv)
        C1, H1, W1 = x1.shape[1], x1.shape[2], x1.shape[3]
        return x1.view(B, V, C1, H1, W1)

    def encode_views_from_layer1(self, x1: torch.Tensor) -> torch.Tensor:
        # (B,V,C1,H1,W1) -> (B,V,512)
        B, V, C1, H1, W1 = x1.shape
        x1v = x1.view(B * V, C1, H1, W1)
        fv = self.backbone.forward_from_layer1(x1v).view(B, V, -1)
        return fv
    
    # ======= (ADD) logit-only forward =======
    def forward_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """(B,D) -> (B,C)"""
        return self.classifier(feat)

    def forward_features(self, x: torch.Tensor, view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,V,3,H,W)
        view_mask: (V,) bool tensor. Pool only views where True (supports mean/max)
        return: (B,D)
        """
        vf = self.encode_views(x)  # (B,V,D)

        if view_mask is None:
            return self.aggregate(vf)

        vm = view_mask.to(vf.device).view(1, -1, 1)  # (1,V,1)

        if self.pool == "mean":
            denom = vm.float().sum(dim=1).clamp_min(1.0)
            feat = (vf * vm.float()).sum(dim=1) / denom
            return feat
        else:
            neg_inf = torch.finfo(vf.dtype).min
            vf2 = vf.masked_fill(~vm.bool(), neg_inf)
            return vf2.max(dim=1).values


# --- DA helper modules (used by methods) ---
class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return GradReverseFn.apply(x, self.lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class AuxClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)
