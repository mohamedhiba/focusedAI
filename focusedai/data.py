from typing import Tuple, Dict
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 224

def make_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

def make_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, Dict[int,str]]:
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=make_transforms("train"))
    val_ds   = datasets.ImageFolder(val_dir,   transform=make_transforms("val"))

    idx_to_class = {v:k for k,v in train_ds.class_to_idx.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, idx_to_class
