import os
import json
import random
import argparse
from itertools import cycle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from models import MVCNN
from methods import build_method

from datasets import build_default_transform, build_weak_transform, build_strong_transform
from datasets import ImageNetCCorruption
import zlib
import random


# -------------------------
# utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    xs, ys = zip(*batch)
    ys = torch.tensor(ys, dtype=torch.long)
    if isinstance(xs[0], dict):
        out = {k: torch.stack([x[k] for x in xs], dim=0) for k in xs[0].keys()}
        return out, ys
    return torch.stack(xs, dim=0), ys


CLASS_NAME_TO_LABEL = {
    "OK": 0,
    "AK": 1,
    "HS": 2,
    "QS": 3,
    "ZW": 4,
}
LABEL_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_LABEL.items()}
NUM_CLASSES = len(CLASS_NAME_TO_LABEL)


@torch.no_grad()
def evaluate(model, loader, device, method=None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

    model.eval()
    if method is not None:
        method.eval()

    ys, probs = [], []
    for x, y in loader:
        if isinstance(x, dict):
            x = x["weak"]
        x = x.to(device)
        y = y.to(device)

        if method is not None and hasattr(method, "predict_logits"):
            logits = method.predict_logits(model, x)
        else:
            logits = model(x)

        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        probs.append(p)

    ys = np.concatenate(ys) if len(ys) else np.array([], dtype=np.int64)
    probs = np.concatenate(probs, axis=0) if len(probs) else np.empty((0, NUM_CLASSES), dtype=np.float32)

    if len(ys) == 0:
        return {"acc": float("nan"), "macro_auroc": float("nan"), "macro_aupr": float("nan")}

    pred = probs.argmax(axis=1)
    acc = accuracy_score(ys, pred)

    unique_classes = np.unique(ys)
    if len(unique_classes) > 1:
        try:
            auroc = roc_auc_score(
                ys,
                probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(NUM_CLASSES)),
            )
        except ValueError:
            auroc = float("nan")

        try:
            y_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[ys]
            aupr = average_precision_score(y_onehot, probs, average="macro")
        except ValueError:
            aupr = float("nan")
    else:
        auroc = float("nan")
        aupr = float("nan")

    metrics = {
        "acc": float(acc),
        "macro_auroc": float(auroc),
        "macro_aupr": float(aupr),
    }

    for cls_idx, cls_name in LABEL_TO_CLASS_NAME.items():
        cls_mask = (ys == cls_idx)
        metrics[f"acc_{cls_name}"] = float((pred[cls_mask] == ys[cls_mask]).mean()) if cls_mask.any() else float("nan")

    return metrics


# -------------------------
# Real-IAD indexing
# -------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_categories(root: str) -> List[str]:
    cats = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and not name.startswith("."):
            cats.append(name)
    cats.sort()
    return cats


def _list_image_files(folder: str) -> List[str]:
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(IMG_EXTS):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files


def build_real_iad_index(root: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    returns:
      index[category] = list of sample dicts:
        {
          "key": "category/DEFECT/S0001",
          "category": str,
          "defect_type": str,  # one of {OK, AK, HS, QS, ZW}
          "label": int,        # OK=0, AK=1, HS=2, QS=3, ZW=4
          "img_paths": [..]
        }
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    cats = list_categories(root)

    for cat in cats:
        cat_root = os.path.join(root, cat)
        samples = []
        # defect_type folders under category
        for defect_type in os.listdir(cat_root):
            droot = os.path.join(cat_root, defect_type)
            if not os.path.isdir(droot) or defect_type.startswith("."):
                continue

            if defect_type not in CLASS_NAME_TO_LABEL:
                raise ValueError(
                    f"Unknown defect_type '{defect_type}' found under category '{cat}'. "
                    f"Expected one of {sorted(CLASS_NAME_TO_LABEL.keys())}."
                )
            label = CLASS_NAME_TO_LABEL[defect_type]

            # sample folders under defect_type
            for sid in os.listdir(droot):
                sroot = os.path.join(droot, sid)
                if not os.path.isdir(sroot) or sid.startswith("."):
                    continue
                img_paths = _list_image_files(sroot)
                if len(img_paths) == 0:
                    continue

                key = f"{cat}/{defect_type}/{sid}"
                samples.append(
                    {
                        "key": key,
                        "category": cat,
                        "defect_type": defect_type,
                        "label": int(label),
                        "img_paths": img_paths,
                    }
                )

        # stable order
        samples.sort(key=lambda x: x["key"])
        index[cat] = samples

    return index


# -------------------------
# splitting
# -------------------------
def group_split(
    items: List[Dict[str, Any]],
    groups: List[str],
    seed: int,
    ratios=(0.7, 0.15, 0.15),
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split by group so that each group appears (if enough items).
    Returns indices into items list.
    """
    assert len(items) == len(groups)
    rng = np.random.default_rng(seed)

    idx_all = np.arange(len(items), dtype=int)
    groups = np.array(groups)

    tr, va, te = [], [], []
    for g in np.unique(groups):
        idx = idx_all[groups == g]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(ratios[0] * n))
        nva = int(round(ratios[1] * n))
        tr += idx[:ntr].tolist()
        va += idx[ntr: ntr + nva].tolist()
        te += idx[ntr + nva:].tolist()

    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def split_val_test_stratified(
    items: List[Dict[str, Any]],
    groups: List[str],
    seed: int,
    val_ratio: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    Split items into val/test only, stratified by `groups`.
    Returns indices into `items`: (val_idx, test_idx)
    """
    assert len(items) == len(groups)
    rng = np.random.default_rng(seed)

    idx_all = np.arange(len(items), dtype=int)
    groups = np.array(groups)

    va, te = [], []
    for g in np.unique(groups):
        idx = idx_all[groups == g]
        rng.shuffle(idx)
        n = len(idx)
        nva = int(round(val_ratio * n))
        va += idx[:nva].tolist()
        te += idx[nva:].tolist()

    rng.shuffle(va)
    rng.shuffle(te)
    return va, te


def take_target_subset_balanced(
    tgt_train_items: List[Dict[str, Any]],
    frac: float,
    seed: int,
    ok_ratio: float = 0.5,
    min_defect_per_type: int = 1,
) -> List[Dict[str, Any]]:
    """
    Reduce target train set but try not to drop classes.
    - label mapping: OK=0, AK=1, HS=2, QS=3, ZW=4
    - no unlabeled target data is used here; this selects from the labeled target-train set only.

    Strategy:
      1) pick n_total = max(1, round(frac * N))
      2) allocate about ok_ratio to OK, rest to defects
      3) ensure each defect_type (excluding OK) gets >= min_defect_per_type if possible
    """
    if frac >= 1.0:
        return tgt_train_items

    rng = np.random.default_rng(seed)
    N = len(tgt_train_items)
    n_total = max(1, int(round(N * frac)))

    ok_items = [s for s in tgt_train_items if s["defect_type"] == "OK"]
    def_items = [s for s in tgt_train_items if s["defect_type"] != "OK"]

    rng.shuffle(ok_items)
    rng.shuffle(def_items)

    n_ok = int(round(n_total * ok_ratio))
    n_ok = max(0, min(len(ok_items), n_ok))
    n_def = n_total - n_ok
    n_def = max(0, min(len(def_items), n_def))

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for s in def_items:
        by_type.setdefault(s["defect_type"], []).append(s)

    defect_types = sorted(list(by_type.keys()))
    for dt in defect_types:
        rng.shuffle(by_type[dt])

    chosen: List[Dict[str, Any]] = []

    chosen += ok_items[:n_ok]

    alloc = {dt: 0 for dt in defect_types}
    remaining = n_def

    for dt in defect_types:
        if remaining <= 0:
            break
        k = min(min_defect_per_type, len(by_type[dt]))
        if k > 0:
            alloc[dt] += k
            remaining -= k

    if remaining > 0 and len(defect_types) > 0:
        caps = {dt: len(by_type[dt]) - alloc[dt] for dt in defect_types}
        total_cap = sum(max(0, v) for v in caps.values())
        if total_cap > 0:
            for dt in defect_types:
                if remaining <= 0:
                    break
                share = int(round(remaining * (max(0, caps[dt]) / total_cap)))
                share = min(share, max(0, caps[dt]))
                alloc[dt] += share

            used = sum(alloc.values())
            leftover = n_def - used
            if leftover > 0:
                dt_order = sorted(defect_types, key=lambda d: (len(by_type[d]) - alloc[d]), reverse=True)
                for dt in dt_order:
                    if leftover <= 0:
                        break
                    can = len(by_type[dt]) - alloc[dt]
                    if can <= 0:
                        continue
                    take = min(can, leftover)
                    alloc[dt] += take
                    leftover -= take

    for dt in defect_types:
        k = min(len(by_type[dt]), alloc[dt])
        if k > 0:
            chosen += by_type[dt][:k]

    if len(chosen) < n_total:
        chosen_keys = set(s["key"] for s in chosen)
        rest = [s for s in tgt_train_items if s["key"] not in chosen_keys]
        rng.shuffle(rest)
        chosen += rest[: (n_total - len(chosen))]

    if len(chosen) > n_total:
        rng.shuffle(chosen)
        chosen = chosen[:n_total]

    return chosen


# -------------------------
# dataset
# -------------------------
class RealIADMultiViewDataset(Dataset):
    """
    One sample = one Sxxxx folder containing multiple view images.
    Returns:
      - (views_tensor, label) when return_multi_aug=False
      - ({"weak":..., "strong1":..., "strong2":...}, label) when return_multi_aug=True
    views_tensor shape: (V,3,H,W)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transform,
        num_views: int = 5,
        training: bool = False,
        return_multi_aug: bool = False,
        weak_transform=None,
        strong1_transform=None,
        strong2_transform=None,
        corruptor=None, corruption_seed=0, deterministic_corruption=True):

        self.samples = samples
        self.transform = transform
        self.num_views = int(num_views)
        self.training = bool(training)

        self.return_multi_aug = bool(return_multi_aug)
        self.weak_transform = weak_transform
        self.strong1_transform = strong1_transform
        self.strong2_transform = strong2_transform

        self.corruptor = corruptor
        self.corruption_seed = int(corruption_seed)
        self.deterministic_corruption = bool(deterministic_corruption)

        if self.return_multi_aug:
            assert self.weak_transform is not None
            assert self.strong1_transform is not None
            assert self.strong2_transform is not None

    def __len__(self):
        return len(self.samples)

    def _select_view_paths(self, paths: List[str]) -> List[str]:
        V = self.num_views
        n = len(paths)
        if n == V:
            return paths

        if n > V:
            if self.training:
                idx = torch.randperm(n)[:V].tolist()
                idx.sort()
                return [paths[i] for i in idx]
            else:
                return paths[:V]

        out = list(paths)
        while len(out) < V:
            out.append(paths[-1])
        return out

    def _load_img(self, p: str) -> Image.Image:
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _apply_corruption(self, img: Image.Image, sample_key: str, view_idx: int) -> Image.Image:
        if self.corruptor is None:
            return img

        if not self.deterministic_corruption:
            return self.corruptor(img)

        h = zlib.adler32(f"{sample_key}|{view_idx}".encode("utf-8")) & 0xFFFFFFFF
        seed = (self.corruption_seed + h) % (2**32 - 1)

        py_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)

        out = self.corruptor(img)

        random.setstate(py_state)
        np.random.set_state(np_state)
        return out

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        y = int(s["label"])
        view_paths = self._select_view_paths(s["img_paths"])

        pil_views = []
        for vi, p in enumerate(view_paths):
            im = self._load_img(p)
            im = self._apply_corruption(im, s["key"], vi)
            pil_views.append(im)

        if not self.return_multi_aug:
            views = [self.transform(im) for im in pil_views]
            x = torch.stack(views, dim=0)
            return x, y

        w = [self.weak_transform(im) for im in pil_views]
        s1 = [self.strong1_transform(im) for im in pil_views]
        s2 = [self.strong2_transform(im) for im in pil_views]
        x = {
            "weak": torch.stack(w, dim=0),
            "strong1": torch.stack(s1, dim=0),
            "strong2": torch.stack(s2, dim=0),
        }
        return x, y


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./Real-IAD")
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        help="must be supported by your methods.build_method()",
    )

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--num_views", type=int, default=5, help="Real-IAD sample folder typically has 5 images")

    ap.add_argument("--target_domain_ratio", type=float, default=0.5,
                    help="fraction of ALL samples assigned to target domain (e.g., 0.5)")
    ap.add_argument("--target_frac", type=float, default=0.2,
                    help="fraction of TARGET domain used for target-train (labeled-scarce)")

    ap.add_argument("--corruption", type=str, default="gaussian_noise", help="ImageNet-C corruption applied to target domain")
    ap.add_argument("--severity", type=int, default=5, help="corruption severity level (1..5)")
    ap.add_argument("--corrupt_p", type=float, default=1.0, help="probability to apply corruption")
    ap.add_argument("--corrupt_seed", type=int, default=1234, help="base seed for deterministic corruption")
    ap.add_argument("--corrupt_deterministic", action="store_true",
                    help="make target corruption deterministic per sample (recommended)")
    
    ap.add_argument("--tau_orc", type=float, default=1.0)
    ap.add_argument("--align_loss", type=str, default="dann")
    ap.add_argument("--multiplier", type=float, default=4.0)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # index dataset
    index = build_real_iad_index(args.data_root)
    categories = sorted(index.keys())

    # flatten all samples across all categories
    all_samples = []
    for c in categories:
        all_samples.extend(index[c])

    # ---- helper: stratified split into (A,B) by group labels
    def stratified_holdout(items, groups, seed, frac_b):
        rng = np.random.default_rng(seed)
        idx_all = np.arange(len(items), dtype=int)
        groups = np.array(groups)

        a_idx, b_idx = [], []
        for g in np.unique(groups):
            idx = idx_all[groups == g]
            rng.shuffle(idx)
            n = len(idx)
            nb = int(round(frac_b * n))

            # safety: keep at least 1 sample on each side if possible
            if n >= 2:
                nb = max(1, min(n - 1, nb))
            else:
                nb = min(n, nb)

            b_idx += idx[:nb].tolist()
            a_idx += idx[nb:].tolist()

        rng.shuffle(a_idx)
        rng.shuffle(b_idx)
        return a_idx, b_idx

    # 1) source/target domain split (balanced by category|defect_type)
    dom_groups = [f"{s['category']}|{s['defect_type']}" for s in all_samples]
    src_idx, tgt_idx = stratified_holdout(
        all_samples,
        dom_groups,
        seed=args.seed + 1,
        frac_b=args.target_domain_ratio,
    )

    src_all = [all_samples[i] for i in src_idx]   # clean source domain
    tgt_all = [all_samples[i] for i in tgt_idx]   # target domain (will be corrupted)

    # 2) source train/val/test split (category|defect_type stratified)
    ratios = (0.7, 0.15, 0.15)
    src_groups = [f"{s['category']}|{s['defect_type']}" for s in src_all]
    src_tr_i, src_va_i, src_te_i = group_split(src_all, src_groups, seed=args.seed, ratios=ratios)
    src_tr = [src_all[i] for i in src_tr_i]
    src_va = [src_all[i] for i in src_va_i]
    src_te = [src_all[i] for i in src_te_i]

    # 3) target split: labeled target-train = target_frac, rest -> val/test 50/50
    #    NOTE: unlabeled target data is NOT used anywhere.
    tgt_groups = [f"{s['category']}|{s['defect_type']}" for s in tgt_all]
    tgt_rest_i, tgt_tr_i = stratified_holdout(
        tgt_all,
        tgt_groups,
        seed=args.seed + 2,
        frac_b=args.target_frac,
    )
    tgt_tr = [tgt_all[i] for i in tgt_tr_i]
    tgt_rest = [tgt_all[i] for i in tgt_rest_i]

    if len(tgt_rest) == 0:
        tgt_va, tgt_te = [], []
    else:
        rest_groups = [f"{s['category']}|{s['defect_type']}" for s in tgt_rest]
        tgt_va_i, tgt_te_i = split_val_test_stratified(
            tgt_rest,
            rest_groups,
            seed=args.seed + 3,
            val_ratio=0.5,
        )
        tgt_va = [tgt_rest[i] for i in tgt_va_i]
        tgt_te = [tgt_rest[i] for i in tgt_te_i]

    # transforms
    default_t = build_default_transform()
    weak_t = build_weak_transform()
    strong1_t = build_strong_transform()
    strong2_t = build_strong_transform()

    return_multi_aug = (args.method == "muvo")

    # corruption (target only)
    corruptor = ImageNetCCorruption(
        corruption_name=args.corruption,
        severity=args.severity,
        p=args.corrupt_p
    ) if args.corruption != "none" else None

    ds_src_tr = RealIADMultiViewDataset(
        src_tr,
        transform=default_t,
        num_views=args.num_views,
        training=True,
        return_multi_aug=False,
        corruptor=None
    )

    ds_tgt_tr = RealIADMultiViewDataset(
        tgt_tr,
        transform=default_t,
        num_views=args.num_views,
        training=True,
        return_multi_aug=return_multi_aug,
        weak_transform=weak_t,
        strong1_transform=strong1_t,
        strong2_transform=strong2_t,
        corruptor=corruptor,
        corruption_seed=args.corrupt_seed,
        deterministic_corruption=args.corrupt_deterministic
    )

    ds_src_va = RealIADMultiViewDataset(src_va, transform=default_t, num_views=args.num_views, training=False, corruptor=None)
    ds_src_te = RealIADMultiViewDataset(src_te, transform=default_t, num_views=args.num_views, training=False, corruptor=None)

    ds_tgt_va = RealIADMultiViewDataset(
        tgt_va,
        transform=default_t,
        num_views=args.num_views,
        training=False,
        corruptor=corruptor,
        corruption_seed=args.corrupt_seed,
        deterministic_corruption=args.corrupt_deterministic
    )
    ds_tgt_te = RealIADMultiViewDataset(
        tgt_te,
        transform=default_t,
        num_views=args.num_views,
        training=False,
        corruptor=corruptor,
        corruption_seed=args.corrupt_seed,
        deterministic_corruption=args.corrupt_deterministic
    )

    assert ds_src_tr.corruptor is None
    assert ds_src_va.corruptor is None
    assert ds_src_te.corruptor is None
    print("[OK] source datasets are clean (no corruptor).")

    print("tgt_tr corruptor:", ds_tgt_tr.corruptor is not None)
    print("tgt_va corruptor:", ds_tgt_va.corruptor is not None)
    print("tgt_te corruptor:", ds_tgt_te.corruptor is not None)

    from collections import Counter

    def grp_count(samples):
        return Counter((s["category"], s["defect_type"], s["label"]) for s in samples)

    print("SRC_TR:", grp_count(src_tr))
    print("SRC_VA:", grp_count(src_va))
    print("SRC_TE:", grp_count(src_te))
    print("TGT_TR:", grp_count(tgt_tr))
    print("TGT_VA:", grp_count(tgt_va))
    print("TGT_TE:", grp_count(tgt_te))

    def keyset(samples):
        return set(s["key"] for s in samples)

    print("overlap src tr/va:", len(keyset(src_tr) & keyset(src_va)))
    print("overlap src tr/te:", len(keyset(src_tr) & keyset(src_te)))
    print("overlap src va/te:", len(keyset(src_va) & keyset(src_te)))
    print("overlap src/tgt:", len(keyset(src_all) & keyset(tgt_all)))

    # two-stream loaders (always include source+target each step)
    bs_t = max(1, int(round(args.batch_size * 0.5)))
    bs_s = max(1, args.batch_size - bs_t)

    def _drop_last(ds_len: int, bsz: int) -> bool:
        return ds_len >= bsz

    dl_src_tr = DataLoader(
        ds_src_tr,
        batch_size=min(bs_s, len(ds_src_tr)) if len(ds_src_tr) > 0 else bs_s,
        shuffle=True,
        drop_last=_drop_last(len(ds_src_tr), bs_s),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_tgt_tr = DataLoader(
        ds_tgt_tr,
        batch_size=min(bs_t, len(ds_tgt_tr)) if len(ds_tgt_tr) > 0 else bs_t,
        shuffle=True,
        drop_last=_drop_last(len(ds_tgt_tr), bs_t),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    dl_src_va = DataLoader(
        ds_src_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_tgt_va = DataLoader(
        ds_tgt_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_src_te = DataLoader(
        ds_src_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_tgt_te = DataLoader(
        ds_tgt_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # model/method
    num_classes = NUM_CLASSES
    model = MVCNN(num_classes=num_classes, pretrained=True, pool=args.pool).to(device)
    method = build_method(args, args.method, feat_dim=model.feat_dim, num_classes=num_classes).to(device)

    # optimizer
    if args.method in ["u2dan", "maa"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(method.parameters()), lr=args.lr)

    
    exp_tag = f"{args.corruption}/{args.seed}"
    print(exp_tag)
    save_dir = os.path.join("./runs_real", exp_tag)
    os.makedirs(save_dir, exist_ok=True)

    split_obj = {
        "protocol": "sample-level source/target split across all categories; target is corrupted; labeled target only (no unlabeled target data)",
        "class_name_to_label": CLASS_NAME_TO_LABEL,
        "target_domain_ratio": args.target_domain_ratio,
        "target_frac": args.target_frac,
        "corruption": {
            "name": args.corruption,
            "severity": args.severity,
            "p": args.corrupt_p,
            "deterministic": args.corrupt_deterministic,
            "seed": args.corrupt_seed,
        },
        "src_tr": [s["key"] for s in src_tr],
        "src_va": [s["key"] for s in src_va],
        "src_te": [s["key"] for s in src_te],
        "tgt_tr": [s["key"] for s in tgt_tr],
        "tgt_va": [s["key"] for s in tgt_va],
        "tgt_te": [s["key"] for s in tgt_te],
    }

    with open(os.path.join(save_dir, "split.json"), "w") as f:
        json.dump(split_obj, f, indent=2)

    history_path = os.path.join(save_dir, "history.json")
    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            json.dump([], f, indent=2)

    steps_per_epoch = len(dl_src_tr)
    if steps_per_epoch == 0:
        raise RuntimeError("Source train loader is empty. Check dataset indexing / splits.")
    if len(dl_tgt_tr) == 0:
        raise RuntimeError("Target train loader is empty. Check target split / target_frac.")

    best = -1.0
    global_step = 0

    src_iter = cycle(dl_src_tr)
    tgt_iter = cycle(dl_tgt_tr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        method.train()

        stats_sum = {}
        stats_cnt = 0

        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}", ncols=160)
        for _ in pbar:
            batch_s = next(src_iter)
            batch_t = next(tgt_iter)

            stats = method.train_step(model, optimizer, batch_s, batch_t, device, global_step)
            global_step += 1

            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    stats_sum[k] = stats_sum.get(k, 0.0) + float(v)
            stats_cnt += 1

            pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items() if isinstance(v, (int, float))})

        stats_avg = {k: (v / max(1, stats_cnt)) for k, v in stats_sum.items()}

        va_t = evaluate(model, dl_tgt_va, device, method)
        va_s = evaluate(model, dl_src_va, device, method)

        score = va_t["macro_auroc"]

        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "target_frac": args.target_frac,
            "seed": args.seed,
            "method": args.method,
            "train_avg": stats_avg,
            "val_src": va_s,
            "val_tgt": va_t,
            "best_val_tgt_acc": best,
        }
        with open(history_path, "r") as f:
            hist = json.load(f)
        hist.append(entry)
        with open(history_path, "w") as f:
            json.dump(hist, f, indent=2)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "method": method.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        if score >= best:
            best = score
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

        print(f"[epoch {epoch}] src_val={va_s}  tgt_val={va_t}  best_tgt_acc={best:.4f}")

    best_ckpt = torch.load(os.path.join(save_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    method.load_state_dict(best_ckpt["method"])

    te_s = evaluate(model, dl_src_te, device, method)
    te_t = evaluate(model, dl_tgt_te, device, method)
    print(f"[TEST] src={te_s}  tgt={te_t}")

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump({"src_test": te_s, "tgt_test": te_t, "best_val_acc": best}, f, indent=2)


if __name__ == "__main__":
    main()