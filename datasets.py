import os
import re
import random
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


_VIEW_ORDER = {"B": 0, "L": 1, "R": 2, "T": 3, "F": 4}


def _infer_view_key(fname: str) -> str:
    # Expect names like ..._B.jpg; the last underscore chunk holds the view letter.
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"_([A-Za-z])$", base)
    if m:
        return m.group(1).upper()
    # Fallback: last character.
    return base[-1].upper()


def _sorted_view_files(files: List[str]) -> List[str]:
    def k(f):
        v = _infer_view_key(f)
        return _VIEW_ORDER.get(v, 999), f
    return sorted(files, key=k)


def build_default_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def build_weak_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma_range=(0.0, 0.03), p=0.3):
        super().__init__()
        self.sigma_range = sigma_range
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W), expected in [0,1] range.
        if torch.rand(1).item() > self.p:
            return x
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0.0, 1.0)


def build_strong_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        # T.RandomAutocontrast(p=0.3),
        # T.RandomEqualize(p=0.1),
        # T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.15),
        T.ToTensor(),
        AddGaussianNoise(sigma_range=(0.0, 0.02), p=0.3),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class MultiViewFolderDataset(Dataset):
    """
    root/
      images/<id>/*.jpg   (each id folder contains V images)
      labels.npy          (labels indexed by id folder name int)
    """
    def __init__(self, root: str, ids: List[int],
                 transform: Optional[T.Compose] = None,
                 return_multi_aug: bool = False,
                 weak_transform: Optional[T.Compose] = None,
                 strong1_transform: Optional[T.Compose] = None,
                 strong2_transform: Optional[T.Compose] = None):
        super().__init__()
        self.root = root
        self.ids = ids
        self.img_root = os.path.join(root, "images")
        self.labels = np.load(os.path.join(root, "labels.npy"))
        self.transform = transform if transform is not None else build_default_transform()

        self.return_multi_aug = return_multi_aug
        self.weak_t = weak_transform if weak_transform is not None else build_weak_transform()
        self.strong1_t = strong1_transform if strong1_transform is not None else build_strong_transform()
        self.strong2_t = strong2_transform if strong2_transform is not None else build_strong_transform()

    def __len__(self):
        return len(self.ids)

    def _load_views(self, obj_id: int) -> List[Image.Image]:
        folder = os.path.join(self.img_root, str(obj_id))
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = _sorted_view_files(files)
        imgs = [Image.open(f).convert("RGB") for f in files]
        return imgs

    def __getitem__(self, idx: int):
        obj_id = self.ids[idx]
        y = int(self.labels[obj_id])
        views = self._load_views(obj_id)

        if not self.return_multi_aug:
            xs = [self.transform(im) for im in views]  # list of (3,H,W)
            x = torch.stack(xs, dim=0)                 # (V,3,H,W)
            return x, y

        # MuVo-style multi-augmentation outputs.
        xs = torch.stack([self.transform(im) for im in views], dim=0)
        w = torch.stack([self.weak_t(im) for im in views], dim=0)
        s1 = torch.stack([self.strong1_t(im) for im in views], dim=0)
        s2 = torch.stack([self.strong2_t(im) for im in views], dim=0)
        x = {"default": xs, "weak": w}
        return x, y


try:
    from imagecorruptions import corrupt as imagenet_c_corrupt
except Exception:
    imagenet_c_corrupt = None


class ImageNetCCorruption:
    """
    ImageNet-C style corruption wrapper.
    corruption_name examples:
      - 'gaussian_noise'
      - 'defocus_blur'
      - 'brightness'
      - 'contrast'
      - 'transforms'  (custom pose variation: rotation + translation)
    severity: 1..5
    """
    def __init__(self, corruption_name: str, severity: int = 5, p: float = 1.0):
        self.corruption_name = str(corruption_name)
        self.severity = int(severity)
        self.p = float(p)

        if self.corruption_name.lower() == "none":
            self.corruption_name = "none"

        # 'transforms' (pose) does not use the imagecorruptions library.
        if self.corruption_name not in ("none", "transforms"):
            if imagenet_c_corrupt is None:
                raise ImportError(
                    "imagecorruptions is not installed. "
                    "Please `pip install imagecorruptions` to match ImageNet-C corruptions."
                )

        if self.corruption_name != "none":
            if not (1 <= self.severity <= 5):
                raise ValueError("severity must be in [1,5]")

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.corruption_name == "none":
            return img
        if np.random.rand() > self.p:
            return img

        # Custom pose variation (rotation + translation).
        if self.corruption_name == "transforms":
            # Strength scales with severity: sev5 -> up to 30 deg rotation, 20% translation.
            max_deg = self.severity * 6.0
            max_trans_ratio = self.severity * 0.04

            angle = random.uniform(-max_deg, max_deg)
            w, h = img.size
            tx = int(random.uniform(-max_trans_ratio * w, max_trans_ratio * w))
            ty = int(random.uniform(-max_trans_ratio * h, max_trans_ratio * h))

            return TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=0.0)

        # Standard ImageNet-C corruptions.
        arr = np.asarray(img).astype(np.uint8)
        out = imagenet_c_corrupt(arr, corruption_name=self.corruption_name, severity=self.severity)
        return Image.fromarray(out)
