# focusedai/model.py
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def create_model(num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    # NOTE: torchvision now prefers 'weights' over 'pretrained', but this still works with a warning.
    m = models.mobilenet_v3_small(pretrained=pretrained)
    # The last classifier layer takes 1024 features on MobileNetV3-Small
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    if freeze_backbone:
        for p in m.features.parameters():
            p.requires_grad = False
    return m

def load_model(ckpt_path: str, device: Optional[torch.device] = None) -> nn.Module:
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = create_model(num_classes=ckpt.get("num_classes", 3), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model
