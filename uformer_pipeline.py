#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 U-FORMER  |  CBCT -> synthetic CT (sCT)  |  BRAIN  |  2D slice-wise pipeline
==============================================================================
One self-contained file that holds the whole flow:

    PHASE 1  preprocess : raw NIfTI volumes  -> normalised 2D axial slices (.npy)
    PHASE 2  split      : patient-level train / val / test split (saved to JSON)
    PHASE 3  train      : train U-Former (masked L1), validate, checkpoint, log
    PHASE 4  test       : 2D slice metrics + 3D patient-level metrics, sCT NIfTI
    PHASE 5  infer      : raw CBCT NIfTI (new patient) -> sCT NIfTI

Objective (unchanged from the original notebooks)
-------------------------------------------------
Input  : CBCT slice, HU clipped to [-1000, 3000] and scaled to [0, 1]
Target : planning CT slice, same normalisation
Output : sCT slice in [0, 1] (sigmoid) -> HU = out * 4000 - 1000
Loss   : L1 inside the official body mask only
Metrics: MAE (HU), PSNR (dB), SSIM, all evaluated inside the mask

Quick start
-----------
    python uformer_pipeline.py --phase all                  # preprocess+split+train+test
    python uformer_pipeline.py --phase preprocess
    python uformer_pipeline.py --phase train --set NUM_EPOCHS=30 BATCH_SIZE=16
    python uformer_pipeline.py --phase train --set RESUME=auto      # continue last.pth
    python uformer_pipeline.py --phase test
    python uformer_pipeline.py --phase infer --input cbct.nii.gz --output sct.nii.gz

In a Jupyter notebook:
    %run uformer_pipeline.py --phase train
    # or
    import uformer_pipeline as up
    up.run(["train"], NUM_EPOCHS=30)

Expected raw layout  (Config.RAW_DIR):
    ./Task2/brain/<patient_id>/cbct.nii.gz
                              /ct.nii.gz
                              /mask.nii.gz

HOW TO MODIFY (for whoever picks this up)
-----------------------------------------
* Paths / hyper-parameters / HU window ...... `Config` dataclass (or `--set KEY=VALUE`)
* Model architecture ........................ `UFormer` (and the blocks above it)
* Loss ...................................... `build_criterion`
* Optimiser / LR schedule ................... `phase_train` (search "OPTIMISER")
* Metrics ................................... `slice_metrics`, `volume_metrics`
* If you change HU_MIN / HU_MAX / preprocessing, re-run `--phase preprocess --force`.
  A `preprocess_meta.json` guard stops you from training on data that was
  normalised with different settings than the current Config.

Known conventions
-----------------
* Volumes are read with SimpleITK -> arrays are (Z, Y, X); slicing is along Z.
* Slices are stored at native in-plane size; resizing to 256x256 (bilinear for
  images, nearest for masks) happens in the Dataset, and predictions are
  resized back to native size before 3D stacking.
* Training uses mixed precision (AMP) on GPU; validation / test / inference run
  in full fp32 so the reported HU numbers are not affected by fp16 rounding.
==============================================================================
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import time
from dataclasses import dataclass, field, fields, asdict

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ==============================================================================
# 0. CONFIGURATION  (single place to change anything)
# ==============================================================================
@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    RAW_DIR: str = "./Task2/brain"                 # <pid>/cbct.nii.gz, ct.nii.gz, mask.nii.gz
    PROCESSED_DIR: str = "./Processed_Task2_Clean"  # cbct/ ct/ mask/ *.npy slices
    CHECKPOINT_DIR: str = "./checkpoints"
    RESULTS_DIR: str = "./Results"
    SPLIT_FILE: str = "./split.json"

    # ---- data / normalisation ---------------------------------------------
    BRAIN_PREFIX: str = "2B"      # only patient ids starting with this are used
    HU_MIN: float = -1000.0       # HU window mapped to [0, 1]
    HU_MAX: float = 3000.0
    IMG_SIZE: int = 256           # model input size (H = W)

    # ---- split (patient level) --------------------------------------------
    SEED: int = 42
    VAL_FRAC: float = 0.10
    TEST_FRAC: float = 0.10
    # Optional: force a fixed test set (comma separated when used with --set).
    # Val set is then drawn from the remaining patients with the same size as test.
    # Example (test set used in the earlier notebooks):
    #   2BB041,2BB016,2BA024,2BB059,2BC046,2BC056,2BA051,2BA004,2BA090,
    #   2BB049,2BB031,2BB177,2BC038,2BC073,2BC024,2BA074,2BC048,2BC001
    TEST_IDS_OVERRIDE: list = field(default_factory=list)

    # ---- model -------------------------------------------------------------
    EMBED_DIM: int = 32
    WIN_SIZE: int = 8

    # ---- training ----------------------------------------------------------
    BATCH_SIZE: int = 8
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 60
    SAVE_EVERY: int = 5           # also keep epoch_XXX.pth every N epochs
    NUM_WORKERS: int = 4
    AMP_TRAIN: bool = True        # mixed precision for the training loop only
    RESUME: str = ""              # "" = fresh | "auto" = CHECKPOINT_DIR/last.pth | path to .pth

    # ---- evaluation / inference -------------------------------------------
    EVAL_BATCH_SIZE: int = 8
    INFER_BATCH: int = 16         # slices per forward pass in 3D reconstruction
    WEIGHTS: str = ""             # "" -> CHECKPOINT_DIR/best_model.pth
    SAVE_CBCT_NIFTI: bool = True  # also export the input CBCT next to sCT/CT

    # ---- derived helpers ---------------------------------------------------
    @property
    def HU_RANGE(self):
        return self.HU_MAX - self.HU_MIN

    def best_path(self):
        return os.path.join(self.CHECKPOINT_DIR, "best_model.pth")

    def last_path(self):
        return os.path.join(self.CHECKPOINT_DIR, "last.pth")

    def log_path(self):
        return os.path.join(self.CHECKPOINT_DIR, "training_log.csv")

    def meta_path(self):
        return os.path.join(self.PROCESSED_DIR, "preprocess_meta.json")


def apply_overrides(cfg, items):
    """Apply KEY=VALUE strings to a Config (type is taken from the current value)."""
    valid = {f.name for f in fields(cfg)}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}'. Valid keys: {sorted(valid)}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            val = raw.strip().lower() in ("1", "true", "yes", "y")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, list):
            val = [s.strip() for s in raw.split(",") if s.strip()]
        else:
            val = raw
        setattr(cfg, key, val)
    return cfg


# ==============================================================================
# 1. SMALL UTILITIES
# ==============================================================================
def banner(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hu_to_norm(hu, cfg):
    """HU -> [0, 1] (clipped)."""
    return np.clip((hu - cfg.HU_MIN) / cfg.HU_RANGE, 0.0, 1.0)


def norm_to_hu(x, cfg):
    """[0, 1] -> HU."""
    return x * cfg.HU_RANGE + cfg.HU_MIN


def z_index(path):
    """Slice number from '<pid>_<zzz>.npy' (used for strict Z ordering)."""
    m = re.search(r"_(\d+)\.npy$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot extract Z index from filename: {path}")
    return int(m.group(1))


def make_amp(device, enabled):
    """Version-tolerant (torch 2.1 ... 2.x) autocast context + GradScaler."""
    use = bool(enabled and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use)
    else:  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use)

    def ctx():
        return torch.amp.autocast(device_type=device.type, enabled=use)

    return ctx, scaler, use


def slice_files(cfg, pid):
    files = glob.glob(os.path.join(cfg.PROCESSED_DIR, "cbct", f"{pid}_*.npy"))
    return sorted(files, key=z_index)


def list_patient_ids(cfg):
    files = glob.glob(os.path.join(cfg.PROCESSED_DIR, "cbct", "*.npy"))
    return sorted({os.path.basename(f).split("_")[0] for f in files})


# ==============================================================================
# 2. PHASE 1 - SLICING + PREPROCESSING
# ==============================================================================
def phase_preprocess(cfg, force=False):
    """
    raw NIfTI -> 2D axial slices.
      * cbct / ct : clip to [HU_MIN, HU_MAX], scale to [0, 1], float32
      * mask      : official body mask, uint8 (0/1)
    File name: <pid>_<zzz>.npy   (Z index zero-padded, ascending along Z)
    """
    banner("PHASE 1 | SLICING + PREPROCESSING")
    if not os.path.isdir(cfg.RAW_DIR):
        raise FileNotFoundError(f"RAW_DIR not found: {cfg.RAW_DIR}")

    out = {k: os.path.join(cfg.PROCESSED_DIR, k) for k in ("cbct", "ct", "mask")}
    for d in out.values():
        os.makedirs(d, exist_ok=True)

    patient_dirs = sorted(d for d in glob.glob(os.path.join(cfg.RAW_DIR, "*")) if os.path.isdir(d))
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders inside {cfg.RAW_DIR}")
    print(f"Raw dir       : {cfg.RAW_DIR}  ({len(patient_dirs)} patient folders)")
    print(f"Output dir    : {cfg.PROCESSED_DIR}")
    print(f"HU window     : [{cfg.HU_MIN:.0f}, {cfg.HU_MAX:.0f}] -> [0, 1]")

    total, skipped_done, skipped_bad = 0, 0, 0
    for pdir in tqdm(patient_dirs, desc="Slicing volumes"):
        pid = os.path.basename(pdir)
        paths = {k: os.path.join(pdir, f"{k}.nii.gz") for k in ("cbct", "ct", "mask")}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"  [skip] {pid}: cbct/ct/mask missing")
            skipped_bad += 1
            continue

        # cheap resume: skip patients whose slices already exist
        if not force:
            rd = sitk.ImageFileReader()
            rd.SetFileName(paths["cbct"])
            rd.ReadImageInformation()
            depth = rd.GetSize()[2]
            if len(glob.glob(os.path.join(out["cbct"], f"{pid}_*.npy"))) == depth:
                skipped_done += 1
                continue

        cbct = sitk.GetArrayFromImage(sitk.ReadImage(paths["cbct"])).astype(np.float32)  # (Z, Y, X)
        ct = sitk.GetArrayFromImage(sitk.ReadImage(paths["ct"])).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(paths["mask"])).astype(np.uint8)

        if not (cbct.shape == ct.shape == mask.shape):
            print(f"  [skip] {pid}: shape mismatch cbct{cbct.shape} ct{ct.shape} mask{mask.shape}")
            skipped_bad += 1
            continue

        # volume-level guard against the "CT == CBCT" file-matching bug
        if float(np.mean(np.abs(cbct - ct))) < 1e-6:
            raise ValueError(f"{pid}: CBCT and CT volumes are identical - check the input files!")

        cbct_n = hu_to_norm(cbct, cfg).astype(np.float32)
        ct_n = hu_to_norm(ct, cfg).astype(np.float32)

        for z in range(cbct_n.shape[0]):
            name = f"{pid}_{z:03d}.npy"
            np.save(os.path.join(out["cbct"], name), cbct_n[z])
            np.save(os.path.join(out["ct"], name), ct_n[z])
            np.save(os.path.join(out["mask"], name), mask[z])
            total += 1

    with open(cfg.meta_path(), "w") as f:
        json.dump({"HU_MIN": cfg.HU_MIN, "HU_MAX": cfg.HU_MAX, "axis": "Z first (SimpleITK array)"}, f, indent=2)

    print(f"\nSlices written this run : {total}")
    print(f"Patients already done   : {skipped_done}   | skipped/bad: {skipped_bad}")
    _preprocess_audit(cfg)


def _preprocess_audit(cfg, n_samples=10):
    """Random-sample check that CBCT and CT slices really differ."""
    files = glob.glob(os.path.join(cfg.PROCESSED_DIR, "cbct", "*.npy"))
    if not files:
        raise RuntimeError("Preprocessing finished but no .npy slices were found!")
    diffs = []
    for p in random.sample(files, min(n_samples, len(files))):
        name = os.path.basename(p)
        a = np.load(p)
        b = np.load(os.path.join(cfg.PROCESSED_DIR, "ct", name))
        diffs.append(float(np.mean(np.abs(a - b))))
    print(f"Audit: mean |CBCT-CT| over {len(diffs)} random slices = {np.mean(diffs):.4f}")
    if sum(diffs) == 0.0:
        raise RuntimeError("AUDIT FAILED: sampled CBCT and CT slices are identical.")
    print("Audit passed: CBCT and CT slices differ.")


def check_preprocess_meta(cfg):
    """Refuse to run if slices were normalised with another HU window."""
    if not os.path.exists(cfg.meta_path()):
        print("[warn] preprocess_meta.json not found - cannot verify the HU window "
              "used to create the slices. Re-run --phase preprocess --force if unsure.")
        return
    with open(cfg.meta_path()) as f:
        meta = json.load(f)
    if float(meta["HU_MIN"]) != cfg.HU_MIN or float(meta["HU_MAX"]) != cfg.HU_MAX:
        raise RuntimeError(
            f"HU window mismatch: slices were made with [{meta['HU_MIN']}, {meta['HU_MAX']}] "
            f"but Config has [{cfg.HU_MIN}, {cfg.HU_MAX}]. Re-run preprocess with --force.")


# ==============================================================================
# 3. PHASE 2 - PATIENT-LEVEL SPLIT (deterministic, saved to JSON)
# ==============================================================================
def _assert_disjoint(train_ids, val_ids, test_ids):
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    assert not (tr & va), f"Train/Val leakage: {tr & va}"
    assert not (tr & te), f"Train/Test leakage: {tr & te}"
    assert not (va & te), f"Val/Test leakage: {va & te}"


def get_split(cfg, force_new=False):
    """
    Returns dict(train=[...], val=[...], test=[...]).
    The split is created ONCE (sorted ids + fixed seed) and stored in SPLIT_FILE;
    every later run (train / test / another machine) reads the same file, which
    removes the risk of train patients ending up in the test set.
    """
    if os.path.exists(cfg.SPLIT_FILE) and not force_new:
        with open(cfg.SPLIT_FILE) as f:
            sp = json.load(f)
        _assert_disjoint(sp["train"], sp["val"], sp["test"])
        present = set(list_patient_ids(cfg))
        missing = [p for k in ("train", "val", "test") for p in sp[k] if p not in present]
        if missing:
            print(f"[warn] {len(missing)} ids from {cfg.SPLIT_FILE} have no slices in "
                  f"{cfg.PROCESSED_DIR}: {missing[:5]}...")
        return sp

    ids = [p for p in list_patient_ids(cfg) if p.startswith(cfg.BRAIN_PREFIX)]
    if not ids:
        raise RuntimeError(f"No patients starting with '{cfg.BRAIN_PREFIX}' in {cfg.PROCESSED_DIR}")

    if cfg.TEST_IDS_OVERRIDE:
        test_ids = list(cfg.TEST_IDS_OVERRIDE)
        unknown = [p for p in test_ids if p not in ids]
        if unknown:
            raise ValueError(f"TEST_IDS_OVERRIDE contains unknown patients: {unknown}")
        rest = [p for p in ids if p not in set(test_ids)]
        train_ids, val_ids = train_test_split(rest, test_size=len(test_ids), random_state=cfg.SEED)
    else:
        hold = cfg.VAL_FRAC + cfg.TEST_FRAC
        train_ids, temp_ids = train_test_split(ids, test_size=hold, random_state=cfg.SEED)
        val_ids, test_ids = train_test_split(temp_ids, test_size=cfg.TEST_FRAC / hold,
                                             random_state=cfg.SEED)

    _assert_disjoint(train_ids, val_ids, test_ids)
    sp = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids),
          "seed": cfg.SEED, "prefix": cfg.BRAIN_PREFIX}
    with open(cfg.SPLIT_FILE, "w") as f:
        json.dump(sp, f, indent=2)
    return sp


def phase_split(cfg, force_new=False):
    banner("PHASE 2 | PATIENT-LEVEL SPLIT + LEAKAGE AUDIT")
    sp = get_split(cfg, force_new=force_new)
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    print(f"Split file          : {cfg.SPLIT_FILE}")
    print(f"Training patients   : {len(tr)}")
    print(f"Validation patients : {len(va)}")
    print(f"Testing patients    : {len(te)}")
    print(f"Overlap train/val   : {len(tr & va)}")
    print(f"Overlap train/test  : {len(tr & te)}")
    print(f"Overlap val/test    : {len(va & te)}")
    print("AUDIT PASSED: sets are disjoint.")
    return sp


# ==============================================================================
# 4. DATASET
# ==============================================================================
class UFormerDataset(Dataset):
    """
    Returns (cbct, ct, mask), each (1, IMG_SIZE, IMG_SIZE) float32.
    Images: bilinear resize. Mask: nearest resize (stays strictly 0/1).
    """

    def __init__(self, data_dir, valid_ids=None, target_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.target_size = (target_size, target_size)
        files = sorted(glob.glob(os.path.join(data_dir, "cbct", "*.npy")))
        if valid_ids is not None:
            keep = set(valid_ids)
            files = [f for f in files if os.path.basename(f).split("_")[0] in keep]
        self.cbct_files = files

    def __len__(self):
        return len(self.cbct_files)

    def __getitem__(self, idx):
        name = os.path.basename(self.cbct_files[idx])
        cbct = np.load(self.cbct_files[idx])
        ct = np.load(os.path.join(self.data_dir, "ct", name))
        mask = np.load(os.path.join(self.data_dir, "mask", name))

        def to_t(a):
            return torch.tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        cbct = F.interpolate(to_t(cbct), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        ct = F.interpolate(to_t(ct), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(to_t(mask), size=self.target_size, mode="nearest").squeeze(0)
        return cbct, ct, mask


def make_loader(cfg, ids, shuffle, batch_size):
    ds = UFormerDataset(cfg.PROCESSED_DIR, valid_ids=ids, target_size=cfg.IMG_SIZE)
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=cfg.NUM_WORKERS,
              pin_memory=torch.cuda.is_available())
    if cfg.NUM_WORKERS > 0:
        kw.update(prefetch_factor=2, persistent_workers=True)
    return ds, DataLoader(ds, **kw)


# ==============================================================================
# 5. MODEL (U-Former: LeWin blocks = window attention + locally-enhanced FFN)
# ==============================================================================
class DepthWiseConv(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class LeFF(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DepthWiseConv(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.linear1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = x.flatten(2).transpose(1, 2)
        return self.linear2(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, win_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        wh = ww = self.win_size
        nwh, nww = H // wh, W // ww
        x = x.reshape(B, nwh, wh, nww, ww, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, wh * ww, C)
        tw = windows.shape[0]
        qkv = self.qkv(windows).reshape(tw, wh * ww, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(tw, wh * ww, C)
        out = self.proj(out)
        out = out.reshape(B, nwh, nww, wh, ww, C)
        return out.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * W, C)


class LeWinBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, win_size)
        self.norm2 = nn.LayerNorm(dim)
        self.leff = LeFF(dim, mlp_ratio=mlp_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.leff(self.norm2(x), H, W)
        return x


class UFormer(nn.Module):
    """2-level U-shaped transformer, sigmoid output in [0, 1]."""

    def __init__(self, in_chans=1, out_chans=1, embed_dim=32, win_size=8):
        super().__init__()
        self.win_size = win_size
        self.input_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.enc_block1 = LeWinBlock(dim=embed_dim, num_heads=1, win_size=win_size)
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=4, stride=2, padding=1)
        self.enc_block2 = LeWinBlock(dim=embed_dim * 2, num_heads=2, win_size=win_size)
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = LeWinBlock(dim=embed_dim * 4, num_heads=4, win_size=win_size)
        self.up2 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.dec_block2 = LeWinBlock(dim=embed_dim * 2, num_heads=2, win_size=win_size)
        self.up1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.dec_block1 = LeWinBlock(dim=embed_dim, num_heads=1, win_size=win_size)
        self.output_proj = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _run(block, x, H, W):
        """(B,C,H,W) -> tokens -> block -> (B,C,H,W)"""
        B, C = x.shape[0], x.shape[1]
        tokens = x.flatten(2).transpose(1, 2)
        return block(tokens, H, W).transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        x_proj = self.input_proj(x)
        B, C, H, W = x_proj.shape
        x_enc1 = self._run(self.enc_block1, x_proj, H, W)

        x_down1 = self.down1(x_enc1)
        H2, W2 = x_down1.shape[2:]
        x_enc2 = self._run(self.enc_block2, x_down1, H2, W2)

        x_down2 = self.down2(x_enc2)
        H3, W3 = x_down2.shape[2:]
        x_bot = self._run(self.bottleneck, x_down2, H3, W3)

        x_up2 = self.up2(x_bot) + x_enc2
        x_dec2 = self._run(self.dec_block2, x_up2, H2, W2)

        x_up1 = self.up1(x_dec2) + x_enc1
        x_dec1 = self._run(self.dec_block1, x_up1, H, W)

        return self.sigmoid(self.output_proj(x_dec1))


def build_model(cfg, device):
    return UFormer(embed_dim=cfg.EMBED_DIM, win_size=cfg.WIN_SIZE).to(device)


# ==============================================================================
# 6. LOSS  (swap here to experiment)
# ==============================================================================
class MaskedL1Loss(nn.Module):
    """Mean absolute error over pixels where mask == 1 only."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.l1(pred.float(), target.float())
        return (loss * mask).sum() / (mask.sum() + 1e-8)


def build_criterion(cfg):
    return MaskedL1Loss()


# ==============================================================================
# 7. METRICS  (all in HU, all inside the body mask)
# ==============================================================================
def slice_metrics(pred_hu, tgt_hu, mask, data_range):
    """Masked MAE / PSNR / SSIM for one 2D slice. Returns None for empty masks."""
    m = mask.astype(bool)
    n = m.sum()
    if n == 0:
        return None
    diff = np.abs(pred_hu - tgt_hu)
    mae = float((diff * m).sum() / n)
    mse = float(((diff ** 2) * m).sum() / n)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else 100.0
    ssim = float(compare_ssim(tgt_hu * m, pred_hu * m, data_range=data_range))
    return mae, float(psnr), ssim


def volume_metrics(pred_hu, gt_hu, mask, data_range):
    """
    Masked 3D metrics for one patient.
      SSIM         : zero outside the mask, whole volume (comparable with earlier logs;
                     slightly optimistic because the identical background counts too)
      SSIM_in_mask : SSIM map of the un-masked volumes, averaged over mask voxels only
    """
    m = mask.astype(bool)
    n = m.sum()
    if n == 0:
        return None
    diff = np.abs(pred_hu - gt_hu)
    mae = float((diff * m).sum() / n)
    mse = float(((diff ** 2) * m).sum() / n)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else 100.0
    ssim = float(compare_ssim(gt_hu * m, pred_hu * m, data_range=data_range))
    _, smap = compare_ssim(gt_hu, pred_hu, data_range=data_range, full=True)
    ssim_in = float(smap[m].mean())
    return dict(MAE=mae, PSNR=float(psnr), SSIM=ssim, SSIM_in_mask=ssim_in)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, cfg, desc="Eval", sample_cb=None):
    """Full-precision evaluation over a loader; averages over non-empty slices."""
    model.eval()
    loss_sum, n_batches = 0.0, 0
    s_mae = s_psnr = s_ssim = 0.0
    n_valid = 0
    for cbct, ct, mask in tqdm(loader, desc=desc, leave=False):
        cbct_d, ct_d, mask_d = cbct.to(device), ct.to(device), mask.to(device)
        out = model(cbct_d)
        if criterion is not None:
            loss_sum += criterion(out, ct_d, mask_d).item()
            n_batches += 1
        pred = out.cpu().squeeze(1).numpy()
        tgt = ct.squeeze(1).numpy()
        msk = mask.squeeze(1).numpy()
        inp = cbct.squeeze(1).numpy()
        for b in range(pred.shape[0]):
            p_hu, t_hu = norm_to_hu(pred[b], cfg), norm_to_hu(tgt[b], cfg)
            r = slice_metrics(p_hu, t_hu, msk[b], cfg.HU_RANGE)
            if r is None:
                continue
            s_mae += r[0]; s_psnr += r[1]; s_ssim += r[2]
            n_valid += 1
            if sample_cb is not None:
                sample_cb(norm_to_hu(inp[b], cfg), p_hu, t_hu, msk[b], r[0])
    n = max(n_valid, 1)
    return dict(loss=loss_sum / max(n_batches, 1), mae=s_mae / n, psnr=s_psnr / n,
                ssim=s_ssim / n, n_valid=n_valid)


# ==============================================================================
# 8. PLOTTING (Figure API -> works headless and in notebooks)
# ==============================================================================
def save_sample_figure(path, inp_hu, pred_hu, tgt_hu, mae):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(20, 5))
    axes = fig.subplots(1, 4)
    for ax, img, title in zip(axes[:3], (inp_hu, pred_hu, tgt_hu),
                              ("Input CBCT", f"Predicted sCT (MAE {mae:.1f} HU)", "Ground-truth CT")):
        ax.imshow(img, cmap="gray", vmin=-1000, vmax=1000)
        ax.set_title(title)
        ax.axis("off")
    im = axes[3].imshow(np.abs(pred_hu - tgt_hu), cmap="hot", vmin=0, vmax=300)
    axes[3].set_title("|sCT - CT|  (0-300 HU)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_training_curves(log_path, out_path):
    try:
        from matplotlib.figure import Figure
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        ep = [int(r["epoch"]) for r in rows]
        fig = Figure(figsize=(15, 4))
        a1, a2, a3 = fig.subplots(1, 3)
        a1.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
        a1.plot(ep, [float(r["val_loss"]) for r in rows], label="val")
        a1.set_title("Loss"); a1.legend()
        a2.plot(ep, [float(r["val_mae"]) for r in rows]); a2.set_title("Val MAE (HU)")
        a3.plot(ep, [float(r["val_ssim"]) for r in rows]); a3.set_title("Val SSIM")
        for a in (a1, a2, a3):
            a.set_xlabel("epoch"); a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    except Exception as e:  # plotting must never break training
        print(f"[warn] could not plot training curves: {e}")


# ==============================================================================
# 9. PHASE 3 - TRAINING
# ==============================================================================
def phase_train(cfg):
    banner("PHASE 3 | TRAINING")
    check_preprocess_meta(cfg)
    set_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    device = get_device()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    sp = get_split(cfg)
    train_ds, train_loader = make_loader(cfg, sp["train"], True, cfg.BATCH_SIZE)
    val_ds, val_loader = make_loader(cfg, sp["val"], False, cfg.EVAL_BATCH_SIZE)
    print(f"Device        : {device}")
    print(f"Train         : {len(sp['train'])} patients / {len(train_ds)} slices")
    print(f"Validation    : {len(sp['val'])} patients / {len(val_ds)} slices")

    model = build_model(cfg, device)
    print(f"Parameters    : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # ---- OPTIMISER (change optimiser / add LR scheduler here) --------------
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = build_criterion(cfg).to(device)
    amp_ctx, scaler, use_amp = make_amp(device, cfg.AMP_TRAIN)
    print(f"Mixed precision (train): {use_amp}")

    # ---- resume -------------------------------------------------------------
    start_epoch, best_val_mae = 0, float("inf")
    resume = cfg.RESUME
    if resume == "auto":
        resume = cfg.last_path() if os.path.exists(cfg.last_path()) else ""
    if resume:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"RESUME checkpoint not found: {resume}")
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scaler_state_dict" in ck and use_amp:
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = int(ck["epoch"])
        best_val_mae = float(ck.get("best_val_mae", ck.get("val_mae", float("inf"))))
        if best_val_mae == float("inf") and os.path.exists(cfg.best_path()):
            best_val_mae = float(torch.load(cfg.best_path(), map_location="cpu").get("val_mae", float("inf")))
        print(f"Resumed from {resume} at epoch {start_epoch}; best val MAE to beat: {best_val_mae:.2f} HU")
    else:
        with open(cfg.log_path(), "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mae",
                                    "val_psnr", "val_ssim", "lr", "seconds"])
    with open(os.path.join(cfg.CHECKPOINT_DIR, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if start_epoch >= cfg.NUM_EPOCHS:
        print(f"Nothing to do: checkpoint epoch {start_epoch} >= NUM_EPOCHS {cfg.NUM_EPOCHS}")
        return

    # ---- epochs -------------------------------------------------------------
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        t0 = time.time()
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [train]", leave=False)
        for cbct, ct, mask in bar:
            cbct = cbct.to(device, non_blocking=True)
            ct = ct.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(cbct)
                loss = criterion(out, ct, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / len(train_loader)

        val = evaluate_loader(model, val_loader, criterion, device, cfg, desc=f"Epoch {epoch + 1} [val]")
        lr = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"Epoch {epoch + 1:>3}/{cfg.NUM_EPOCHS} | train {train_loss:.4f} | val {val['loss']:.4f} | "
              f"MAE {val['mae']:.2f} HU | PSNR {val['psnr']:.2f} dB | SSIM {val['ssim']:.4f} | {secs:.0f}s")

        with open(cfg.log_path(), "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val["loss"], val["mae"], val["psnr"],
                                    val["ssim"], lr, round(secs, 1)])

        is_best = val["mae"] < best_val_mae
        if is_best:
            best_val_mae = val["mae"]
        state = {"epoch": epoch + 1,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scaler_state_dict": scaler.state_dict() if use_amp else None,
                 "val_mae": val["mae"], "best_val_mae": best_val_mae}
        torch.save(state, cfg.last_path())
        if is_best:
            torch.save(state, cfg.best_path())
            print(f"  -> new best model saved (val MAE {best_val_mae:.2f} HU)")
        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            torch.save(state, os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch + 1:03d}.pth"))

    plot_training_curves(cfg.log_path(), os.path.join(cfg.CHECKPOINT_DIR, "training_curves.png"))
    print(f"\nTraining complete. Best val MAE {best_val_mae:.2f} HU -> {cfg.best_path()}")


# ==============================================================================
# 10. INFERENCE HELPERS (shared by test + infer)
# ==============================================================================
def load_model(cfg, weights=None, device=None):
    device = device or get_device()
    path = weights or cfg.WEIGHTS or cfg.best_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    ck = torch.load(path, map_location=device)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model = build_model(cfg, device)
    model.load_state_dict(sd)
    model.eval()
    info = f"epoch {ck.get('epoch', '?')}" if isinstance(ck, dict) else "raw state_dict"
    print(f"Loaded weights: {path}  ({info})")
    return model


@torch.no_grad()
def predict_volume01(model, vol01, cfg, device):
    """
    vol01: (Z, H, W) float32 in [0, 1]  ->  prediction (Z, H, W) in [0, 1]
    Each slice is resized to IMG_SIZE for the network and back to native H x W.
    """
    model.eval()
    Z, H, W = vol01.shape
    out = np.zeros((Z, H, W), dtype=np.float32)
    size = (cfg.IMG_SIZE, cfg.IMG_SIZE)
    for s in range(0, Z, cfg.INFER_BATCH):
        chunk = torch.from_numpy(vol01[s:s + cfg.INFER_BATCH]).float().unsqueeze(1).to(device)
        x = F.interpolate(chunk, size=size, mode="bilinear", align_corners=False)
        y = model(x)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        out[s:s + cfg.INFER_BATCH] = y.squeeze(1).cpu().numpy()
    return out


def read_geometry(cfg, pid, shape_zyx):
    """Spacing / origin / direction from the raw CT header (header only, no pixel read)."""
    p = os.path.join(cfg.RAW_DIR, pid, "ct.nii.gz")
    if not os.path.exists(p):
        print(f"  [warn] {p} not found -> sCT saved with default geometry (1mm, origin 0)")
        return None
    rd = sitk.ImageFileReader()
    rd.SetFileName(p)
    rd.ReadImageInformation()
    if tuple(rd.GetSize()[::-1]) != tuple(shape_zyx):
        print(f"  [warn] {pid}: raw CT size {rd.GetSize()[::-1]} != slices {tuple(shape_zyx)} "
              f"-> default geometry")
        return None
    return rd.GetSpacing(), rd.GetOrigin(), rd.GetDirection()


def to_sitk(arr, geom=None):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32))
    if geom is not None:
        img.SetSpacing(geom[0]); img.SetOrigin(geom[1]); img.SetDirection(geom[2])
    return img


# ==============================================================================
# 11. PHASE 4 - TESTING (2D slice metrics + 3D patient metrics + sCT export)
# ==============================================================================
def test_2d(cfg, model, test_ids, device):
    banner("PHASE 4a | 2D SLICE-LEVEL TEST (unseen patients, native fp32)")
    ds, loader = make_loader(cfg, test_ids, False, cfg.EVAL_BATCH_SIZE)
    print(f"Test patients: {len(test_ids)} | slices: {len(ds)}")
    saved = {"done": False}
    fig_path = os.path.join(cfg.RESULTS_DIR, "test_sample.png")

    def sample_cb(inp, pred, tgt, msk, mae):
        if not saved["done"] and msk.sum() > 5000:
            save_sample_figure(fig_path, inp, pred, tgt, mae)
            saved["done"] = True

    res = evaluate_loader(model, loader, None, device, cfg, desc="2D test", sample_cb=sample_cb)
    print(f"Evaluated slices : {res['n_valid']}")
    print(f"MAE  : {res['mae']:.2f} HU")
    print(f"PSNR : {res['psnr']:.2f} dB")
    print(f"SSIM : {res['ssim']:.4f}")
    with open(os.path.join(cfg.RESULTS_DIR, "test_2d_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def test_3d(cfg, model, test_ids, device):
    banner("PHASE 4b | 3D PATIENT-LEVEL TEST + sCT NIfTI EXPORT")
    nifti_dir = os.path.join(cfg.RESULTS_DIR, "NIfTI")
    os.makedirs(nifti_dir, exist_ok=True)
    rows = []
    for pid in tqdm(test_ids, desc="3D reconstruction"):
        files = slice_files(cfg, pid)
        if not files:
            print(f"  [skip] {pid}: no slices")
            continue
        z = [z_index(f) for f in files]
        if z != list(range(z[0], z[-1] + 1)):
            print(f"  [skip] {pid}: missing slices in Z sequence")
            continue

        cbct = np.stack([np.load(f) for f in files])
        ct = np.stack([np.load(f.replace(os.sep + "cbct" + os.sep, os.sep + "ct" + os.sep)) for f in files])
        mask = np.stack([np.load(f.replace(os.sep + "cbct" + os.sep, os.sep + "mask" + os.sep)) for f in files])

        pred_hu = norm_to_hu(predict_volume01(model, cbct, cfg, device), cfg)
        gt_hu = norm_to_hu(ct, cfg)

        m = volume_metrics(pred_hu, gt_hu, mask, cfg.HU_RANGE)
        if m is None:
            print(f"  [skip] {pid}: empty mask")
            continue
        rows.append({"patient": pid, "slices": len(files), **m})

        geom = read_geometry(cfg, pid, pred_hu.shape)
        sitk.WriteImage(to_sitk(pred_hu, geom), os.path.join(nifti_dir, f"{pid}_sCT.nii.gz"))
        sitk.WriteImage(to_sitk(gt_hu, geom), os.path.join(nifti_dir, f"{pid}_CT.nii.gz"))
        if cfg.SAVE_CBCT_NIFTI:
            sitk.WriteImage(to_sitk(norm_to_hu(cbct, cfg), geom), os.path.join(nifti_dir, f"{pid}_CBCT.nii.gz"))

    if not rows:
        print("No patients evaluated.")
        return None

    csv_path = os.path.join(cfg.RESULTS_DIR, "test_3d_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return v.mean(), v.std()

    best = min(rows, key=lambda r: r["MAE"])["patient"]
    worst = max(rows, key=lambda r: r["MAE"])["patient"]
    lines = ["3D PATIENT-LEVEL TEST SUMMARY (metrics inside body mask)",
             "=" * 56,
             f"Patients evaluated : {len(rows)}",
             f"MAE  (HU)          : {ms('MAE')[0]:.2f} +/- {ms('MAE')[1]:.2f}",
             f"PSNR (dB)          : {ms('PSNR')[0]:.2f} +/- {ms('PSNR')[1]:.2f}",
             f"SSIM (masked vol.) : {ms('SSIM')[0]:.4f} +/- {ms('SSIM')[1]:.4f}",
             f"SSIM (in-mask map) : {ms('SSIM_in_mask')[0]:.4f} +/- {ms('SSIM_in_mask')[1]:.4f}",
             f"Best patient       : {best}",
             f"Worst patient      : {worst}"]
    text = "\n".join(lines)
    with open(os.path.join(cfg.RESULTS_DIR, "test_summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nPer-patient CSV : {csv_path}")
    print(f"sCT NIfTI files : {nifti_dir}")
    return rows


def phase_test(cfg, weights=None):
    banner("PHASE 4 | TESTING")
    check_preprocess_meta(cfg)
    device = get_device()
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    sp = get_split(cfg)
    model = load_model(cfg, weights, device)
    test_2d(cfg, model, sp["test"], device)
    test_3d(cfg, model, sp["test"], device)


# ==============================================================================
# 12. PHASE 5 - INFERENCE ON NEW RAW CBCT VOLUME(S)
# ==============================================================================
def phase_infer(cfg, input_path, output_path, weights=None):
    """
    Raw CBCT NIfTI -> sCT NIfTI (same size / spacing / origin / direction).
    NOTE: the CBCT must be in the same format as the training data (already
    resampled/cropped like Task2, HU-calibrated) - the network only handles the
    intensity mapping, not registration or resampling.
    """
    banner("PHASE 5 | INFERENCE (raw CBCT -> sCT)")
    if not input_path or not output_path:
        raise ValueError("--phase infer needs --input <file|folder> and --output <file|folder>")
    device = get_device()
    model = load_model(cfg, weights, device)

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        jobs = [(p, os.path.join(output_path, os.path.basename(p).replace(".nii.gz", "").replace(".nii", "") + "_sCT.nii.gz"))
                for p in sorted(glob.glob(os.path.join(input_path, "*.nii*")))]
    else:
        jobs = [(input_path, output_path)]
    if not jobs:
        raise FileNotFoundError(f"No .nii/.nii.gz files in {input_path}")

    for src, dst in jobs:
        img = sitk.ReadImage(src)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)          # (Z, Y, X) in HU
        if arr.ndim != 3:
            raise ValueError(f"{src}: expected a 3D volume, got shape {arr.shape}")
        vol01 = hu_to_norm(arr, cfg).astype(np.float32)
        pred_hu = norm_to_hu(predict_volume01(model, vol01, cfg, device), cfg)
        out = sitk.GetImageFromArray(pred_hu.astype(np.float32))
        out.CopyInformation(img)                                       # keep the geometry
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        sitk.WriteImage(out, dst)
        print(f"  {src}  ->  {dst}   (HU range {pred_hu.min():.0f} .. {pred_hu.max():.0f})")


# ==============================================================================
# 13. ENTRY POINT
# ==============================================================================
ALL_PHASES = ["preprocess", "split", "train", "test"]


def run(phases=("all",), cfg=None, force=False, input_path=None, output_path=None,
        weights=None, **overrides):
    """Programmatic entry (notebooks): run(['train'], NUM_EPOCHS=30)."""
    cfg = cfg or Config()
    apply_overrides(cfg, [f"{k}={v if not isinstance(v, (list, tuple)) else ','.join(v)}"
                          for k, v in overrides.items()])
    phases = list(phases)
    if "all" in phases:
        phases = ALL_PHASES
    for ph in phases:
        if ph == "preprocess":
            phase_preprocess(cfg, force=force)
        elif ph == "split":
            phase_split(cfg, force_new=force)
        elif ph == "train":
            phase_train(cfg)
        elif ph == "test":
            phase_test(cfg, weights)
        elif ph == "infer":
            phase_infer(cfg, input_path, output_path, weights)
        else:
            raise ValueError(f"Unknown phase '{ph}'. Choose from {ALL_PHASES + ['infer', 'all']}")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="U-Former CBCT->sCT pipeline (brain)")
    ap.add_argument("--phase", nargs="+", default=["all"],
                    help="preprocess | split | train | test | infer | all "
                         "(all = preprocess split train test)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override Config values, e.g. --set NUM_EPOCHS=30 BATCH_SIZE=16")
    ap.add_argument("--force", action="store_true",
                    help="preprocess: redo all patients | split: create a new split")
    ap.add_argument("--weights", default=None, help="checkpoint for test/infer (default best_model.pth)")
    ap.add_argument("--input", default=None, help="infer: CBCT .nii.gz file or folder")
    ap.add_argument("--output", default=None, help="infer: output .nii.gz file or folder")
    args, _ = ap.parse_known_args(argv)   # tolerant of Jupyter's extra args

    cfg = apply_overrides(Config(), args.set)
    run(args.phase, cfg=cfg, force=args.force, input_path=args.input,
        output_path=args.output, weights=args.weights)


if __name__ == "__main__":
    main()
