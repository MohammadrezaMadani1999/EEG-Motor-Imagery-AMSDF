# -*- coding: utf-8 -*-
"""
================================================================================
OFNER 2017 (BNCI 001-2017) - FB-CAT-Net PIPELINE (7-CLASS MOTOR IMAGERY)
--------------------------------------------------------------------------------
Filter-Bank Cross-Attention Temporal Network with Euclidean Alignment
--------------------------------------------------------------------------------
Novel contributions (for the paper):
  1. Physiology-informed filter bank: MRCP (0.1-3 Hz, movement-type information
     per Ofner et al. 2017), Mu (8-13 Hz) and Beta (13-30 Hz) ERD/ERS streams,
     each with a BAND-SPECIFIC temporal kernel length matched to its dominant
     time scale (long kernels for slow MRCPs, short kernels for beta bursts).
  2. Per-run Euclidean Alignment (EA): unsupervised whitening of each run's
     covariance to suppress inter-run non-stationarity (the dominant noise
     source in this low-SNR dataset).
  3. Cross-band token Transformer: tokens from all bands (+ learnable band
     embeddings + CLS token) interact via multi-head self-attention, allowing
     the model to discover cross-frequency couplings; an auxiliary softmax
     gate provides interpretable per-class band importance.
  4. Sliding-crop training augmentation + multi-crop test-time ensembling
     (ATCNet-style), plus channel dropout and mild Gaussian noise.
  5. Leakage-free protocol: 10-fold Leave-One-Run-Out (LORO); early stopping
     on a HELD-OUT RUN (run-wise validation), never on shuffled trials.
  6. Built-in EEGNet baseline, single-band ablations, Wilcoxon signed-rank
     statistics, and publication-ready figures/tables.

Critical preprocessing facts (all empirically verified on the GDF files):
  * Class markers 1536-1542 sit AT THE CUE (exactly 2.000 s after the 768
    trial-start marker in every trial) -> the MI window is [0, 3] s from the
    class marker. Do not add any offset.
  * The GDF files contain whole-timepoint NaN recording gaps. They are
    linearly interpolated for filtering; any epoch overlapping a gap
    (+/- 0.5 s filter-transient margin) is dropped and reported.
  * All band-pass filtering is applied to the CONTINUOUS recording BEFORE
    epoching (filtering 3-s epochs at 0.1-3 Hz destroys the MRCP with edge
    transients).
  * Signals are decimated 512 -> 128 Hz AFTER low-pass filtering (alias-free).
  * MNE reads these GDFs with a unit mismatch (values ~1e6 too large);
    harmless here because every trial is z-scored per channel.
  * Euclidean Alignment: plain whitening of the 0.1-3 Hz band can amplify
    low-eigenvalue noise directions. The implementation therefore supports
    shrinkage regularization (EA_SHRINKAGE) and EA is OFF by default until
    validated per-dataset with sanity_lda.py; treat EA on/off (and its
    shrinkage level) as a reported ablation in the paper.

Expected performance context (literature, subject-specific, this dataset):
  Ofner et al. 2017 (sLDA, MI, 6-class): ~27% | EEGNet: ~32% | TSPNet: ~50%
  7-class chance level: 14.3%. A LORO result in the 40-50% range with proper
  statistics is a solid, publishable contribution.
================================================================================
"""

import os
import gc
import copy
import json
import math
import random
import warnings

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, cohen_kappa_score)

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"D:\OFNER"
SAVE_DIR = r".\Paper_Results_AMSDF_Final"
os.makedirs(SAVE_DIR, exist_ok=True)

N_SUBJECTS = 15
N_RUNS = 10
FS_RAW = 512
DECIM = 4                      # 512 Hz -> 128 Hz
FS = FS_RAW // DECIM           # 128 Hz
N_CLASSES = 7

CLASS_NAMES = ["Elbow Flex", "Elbow Ext", "Supination", "Pronation",
               "Hand Close", "Hand Open", "Rest"]

# Anatomical task grouping used by the joint-aware auxiliary loss.
JOINT_LABELS = np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64)
JOINT_NAMES = ["Elbow", "Forearm", "Hand", "Rest"]
ANTAGONIST_PAIRS = [(0, 1), (2, 3), (4, 5)]

# Class cue markers (marker 768 = trial start, intentionally excluded)
EVENT_TO_CLASS = {"1536": 0, "1537": 1, "1538": 2, "1539": 3,
                  "1540": 4, "1541": 5, "1542": 6}

EEG_CHANNELS = [
    "F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h",
    "FFC2h", "FFC4h", "FFC6h", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FTT7h", "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h",
    "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "TTP7h", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
    "TTP8h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", "P1",
    "Pz", "P2", "P4", "PPO1h", "PPO2h"
]
N_CHANNELS = len(EEG_CHANNELS)

# Physiology-informed filter bank: (low_Hz, high_Hz, temporal_kernel_seconds)
# Long kernels for slow MRCPs, shorter for faster rhythms.
FILTER_BANK = [
    ("MRCP", 0.1, 3.0, 1.00),
    ("Mu",   8.0, 13.0, 0.50),
    ("Beta", 13.0, 30.0, 0.25),
]
N_BANDS = len(FILTER_BANK)
BAND_NAMES = [b[0] for b in FILTER_BANK]
BROADBAND = (0.1, 40.0)        # used only by the EEGNet baseline

# CRITICAL TIMING (empirically verified on the GDF files: the 768 trial-start
# marker precedes each class marker by exactly 2.000 s, and end-of-event codes
# class+0x8000 follow 3 s later): class markers 1536-1542 are placed AT THE
# CUE (trial second 2). Epoching [0, 3] s from the class marker therefore
# captures the full MI period (trial seconds 2-5). Do NOT add an offset.
TASK_ONSET_SEC = 0.0
EPOCH_SEC = 3.0                # MI window: [0, 3] s relative to the cue marker
EPOCH_SAMPLES = int(FS * EPOCH_SEC)          # 384
CROP_SEC = 2.5                 # training crop length (sliding-crop aug.)
CROP_SAMPLES = int(FS * CROP_SEC)            # 320
N_TEST_CROPS = 5               # multi-crop test-time ensembling

BATCH_SIZE = 64
EPOCHS = 120
WARMUP_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-3
LABEL_SMOOTHING = 0.10
PATIENCE = 20
SEED = 42

RUN_ABLATIONS = False           # ON for the final run: single-band ablations
                               # (MRCP/Mu/Beta) feed the ablation table.
RUN_BASELINE = False            # standard EEGNet on broadband signal (required)

# Euclidean Alignment: OFF — LOCKED by two paired experiments:
# (1) plain EA collapsed the deep model to chance (13.8% vs 21.2% without);
# (2) even shrinkage-EA (a=0.30) cost 2-3% on every deep model in the paired
#     S01 comparison (FB-CAT 21.7->19.8, Mu 19.5->17.1, EEGNet 19.5->16.7),
#     while sLDA was unharmed (21.0->22.4). Reportable finding: EA whitening
#     harms deep-net training in within-subject cross-run decoding here.
APPLY_EA = False
EA_SHRINKAGE = 0.30

# Diagnostic switch: with False the model skips the Transformer and uses only
# per-band conv encoders + gated fusion (a strong compact baseline).
# LOCKED False for the conference run: paired A/B tests on S01-S03 showed the
# Transformer overfits ~340 training trials (S01: 17.6% vs 21.7% without).
USE_TRANSFORMER = False

# AMSDF hyperparameters ------------------------------------------------------
D_LATENT = 48
JOINT_LOSS_WEIGHT = 0.20
SUBJECT_ADV_WEIGHT = 0.10
SUBJECT_LATENT_WEIGHT = 0.20
ORTHOGONAL_WEIGHT = 0.01
GRL_LAMBDA = 0.10
SAVE_REPRESENTATIONS = True

# Optional analyses in this integrated script. Keep cross-subject off for the
# first run; enable it only for the full 15-subject study.
RUN_CROSS_SUBJECT = False
RUN_RSA = True

# Seed ensembling: number of models trained per fold (predictions fused by
# averaging log-probabilities). Set to 3 for the FINAL run (steadier accuracy,
# lower fold variance); 1 for quick tuning runs.
N_SEEDS = 3

# Smoke-test mode: set True for the first run. It limits the subjects/runs/seeds
# without changing the model itself. For the paper run keep it False.
QUICK_MODE = True
QUICK_SUBJECTS = [1]
QUICK_RUNS = 2
QUICK_SEEDS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_SUBJECTS = QUICK_SUBJECTS if QUICK_MODE else list(range(1, N_SUBJECTS + 1))
RUN_N_RUNS = QUICK_RUNS if QUICK_MODE else N_RUNS
RUN_N_SEEDS = QUICK_SEEDS if QUICK_MODE else N_SEEDS


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# PREPROCESSING
# =============================================================================
def check_finite(X, where=""):
    """Fail loudly on non-finite values appearing AFTER NaN handling."""
    if not np.isfinite(X).all():
        n_bad = np.size(X) - np.isfinite(X).sum()
        raise ValueError(f"Non-finite values ({n_bad}) detected in {where}. "
                         f"Inspect the raw file instead of masking the issue.")
    return X


# Filter transient safety margin around NaN gaps (samples at 512 Hz).
NAN_MARGIN = int(0.5 * FS_RAW)


def clean_nonfinite(data, tag=""):
    """Ofner2017 GDF files contain whole-timepoint NaN gaps (recording
    breaks/edges). Strategy: (1) linearly interpolate gaps per channel so
    the continuous filter runs cleanly, (2) return a boolean bad-sample
    mask (dilated by NAN_MARGIN for filter transients) so that any epoch
    overlapping a gap is DROPPED and reported - never silently used."""
    bad = ~np.isfinite(data).all(axis=0)                 # (T,) any-channel bad
    n_bad = int(bad.sum())
    if n_bad:
        t = np.arange(data.shape[1])
        for c in range(data.shape[0]):
            ch = data[c]
            good = np.isfinite(ch)
            if not good.all():
                if not good.any():
                    raise ValueError(f"{tag}: channel {c} entirely non-finite.")
                data[c] = np.interp(t, t[good], ch[good])
        # dilate the mask to cover filter edge transients around each gap
        kernel = np.ones(2 * NAN_MARGIN + 1, dtype=bool)
        bad = np.convolve(bad, kernel, mode="same") > 0
        print(f"  [info] {tag}: {n_bad} NaN timepoints "
              f"({n_bad / FS_RAW:.2f}s) interpolated; epochs overlapping "
              f"these gaps (+/-{NAN_MARGIN / FS_RAW:.1f}s margin) will be "
              f"dropped.")
    return data, bad


def load_run_continuous(subject, run):
    """Load one GDF run; return continuous data (C, T) and cue events."""
    filename = f"motorimagination_subject{subject}_run{run}.gdf"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, None

    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    if len(available) != N_CHANNELS:
        print(f"  [warn] S{subject:02d}R{run:02d}: "
              f"{len(available)}/{N_CHANNELS} channels found - run skipped.")
        return None, None
    raw.pick(available)
    raw.reorder_channels(EEG_CHANNELS)

    data = raw.get_data().astype(np.float64)          # (C, T) at 512 Hz
    data, bad_mask = clean_nonfinite(data, tag=f"S{subject:02d}R{run:02d}")
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    inv = {v: k for k, v in event_id.items()}

    cues = []                                          # (sample_512, class)
    for ev in events:
        ann = inv.get(ev[2], str(ev[2]))
        if ann in EVENT_TO_CLASS:
            cues.append((int(ev[0]), EVENT_TO_CLASS[ann]))
    del raw
    gc.collect()
    return data, cues, bad_mask


def bandpass_continuous(data, low, high):
    """Zero-phase IIR band-pass on the CONTINUOUS signal (order-4 Butterworth,
    applied forward-backward by MNE). Filtering before epoching is essential
    for the 0.1-3 Hz MRCP band."""
    Xf = mne.filter.filter_data(
        data, sfreq=FS_RAW, l_freq=low, h_freq=high,
        method="iir", iir_params={"order": 4, "ftype": "butter"},
        verbose=False)
    return check_finite(Xf, f"band {low}-{high} Hz")


def epoch_and_decimate(data_filt, cues, bad_mask):
    """Cut the MI period [TASK_ONSET, TASK_ONSET + EPOCH_SEC] s after each
    trial-start marker (markers sit at t=0; the cue appears at t=2 s, so
    this yields [2, 5] s = the actual task window).
    Epochs overlapping a NaN gap (dilated mask) are dropped and counted.
    Decimation by simple slicing is alias-free because every band is
    low-pass limited to <= 40 Hz << 64 Hz (new Nyquist)."""
    offset = int(FS_RAW * TASK_ONSET_SEC)
    win = int(FS_RAW * EPOCH_SEC)
    X, y, n_dropped = [], [], 0
    for sample, cls in cues:
        start = sample + offset
        seg = data_filt[:, start:start + win]
        if seg.shape[1] != win:
            n_dropped += 1
            continue
        if bad_mask[start:start + win].any():
            n_dropped += 1
            continue
        X.append(seg[:, ::DECIM].astype(np.float32))
        y.append(cls)
    if not X:
        return (np.empty((0, N_CHANNELS, EPOCH_SAMPLES), np.float32),
                np.empty(0, np.int64), n_dropped)
    return np.stack(X), np.asarray(y, np.int64), n_dropped


def euclidean_alignment(X, ref=None, shrinkage=None):
    """Per-run Euclidean Alignment (He & Wu, 2020) with optional shrinkage
    regularization: whiten trials with the inverse square root of the run's
    mean spatial covariance blended toward identity. Unsupervised
    (label-free), computed within each run -> no label leakage."""
    if shrinkage is None:
        shrinkage = EA_SHRINKAGE
    if ref is None:
        R = np.zeros((X.shape[1], X.shape[1]), np.float64)
        for x in X:
            R += (x @ x.T) / x.shape[1]
        R /= len(X)
    else:
        R = ref
    if shrinkage > 0:
        mu = np.trace(R) / R.shape[0]
        R = (1.0 - shrinkage) * R + shrinkage * mu * np.eye(R.shape[0])
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, 1e-10)
    R_isqrt = (V * (w ** -0.5)) @ V.T
    return np.einsum("ij,njk->nik", R_isqrt, X).astype(np.float32)


def zscore_trials(X):
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True)
    s[s < 1e-8] = 1.0
    return ((X - m) / s).astype(np.float32)


def load_subject(subject, n_runs=RUN_N_RUNS):
    """Return dict: band_name -> (n_trials, C, T) plus labels and run ids.
    Pipeline per run: CAR -> continuous band-pass -> epoch -> decimate ->
    per-run EA -> per-trial z-score."""
    band_names = [b[0] for b in FILTER_BANK] + ["Broad"]
    store = {b: [] for b in band_names}
    y_all, run_all = [], []

    total_dropped = 0
    for run in range(1, n_runs + 1):
        data, cues, bad_mask = load_run_continuous(subject, run)
        if data is None or not cues:
            continue
        data = data - data.mean(axis=0, keepdims=True)          # CAR
        check_finite(data, f"S{subject:02d}R{run:02d} after interpolation")

        y_run, n_drop = None, 0
        for name, lo, hi, _ in FILTER_BANK:
            Xf = bandpass_continuous(data, lo, hi)
            Xe, y_run, n_drop = epoch_and_decimate(Xf, cues, bad_mask)
            if APPLY_EA and len(Xe) > 1:
                Xe = euclidean_alignment(Xe)
            store[name].append(zscore_trials(Xe))
        # broadband stream for the EEGNet baseline
        Xf = bandpass_continuous(data, *BROADBAND)
        Xe, y_run, n_drop = epoch_and_decimate(Xf, cues, bad_mask)
        if APPLY_EA and len(Xe) > 1:
            Xe = euclidean_alignment(Xe)
        store["Broad"].append(zscore_trials(Xe))

        if n_drop:
            print(f"  [info] S{subject:02d}R{run:02d}: {n_drop} trial(s) "
                  f"dropped (NaN-gap overlap or truncated window).")
        total_dropped += n_drop
        y_all.append(y_run)
        run_all.append(np.full(len(y_run), run, np.int64))
        del data
        gc.collect()

    bands = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    y = np.concatenate(y_all)
    runs = np.concatenate(run_all)

    counts = np.bincount(y, minlength=N_CLASSES)
    print(f"  trials/class: {counts.tolist()}  (total {len(y)}, "
          f"dropped {total_dropped})")
    if len(y) + total_dropped != 420:
        print(f"  [warn] expected 420 cue events, found "
              f"{len(y) + total_dropped}.")
    return bands, y, runs


# =============================================================================
# DATASET (sliding-crop augmentation, channel dropout, mild noise)
# =============================================================================
class BandDataset(Dataset):
    """X: (n_trials, n_bands, C, T_epoch). Training returns a random
    CROP_SAMPLES crop (sliding-crop augmentation); evaluation returns the
    full epoch (multi-crop ensembling handled in the eval loop)."""

    def __init__(self, X, y, train=False, subject_ids=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.train = train
        self.subject_ids = None if subject_ids is None else torch.tensor(subject_ids, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, label = self.X[idx], self.y[idx]
        if self.train:
            start = torch.randint(0, x.shape[-1] - CROP_SAMPLES + 1, (1,)).item()
            x = x[..., start:start + CROP_SAMPLES].clone()
            if torch.rand(1).item() < 0.30:              # channel dropout
                n_drop = torch.randint(1, 7, (1,)).item()
                chans = torch.randperm(x.shape[1])[:n_drop]
                x[:, chans, :] = 0.0
            if torch.rand(1).item() < 0.50:              # mild Gaussian noise
                x = x + torch.randn_like(x) * 0.05
        if self.subject_ids is None:
            return x, label
        return x, label, self.subject_ids[idx]


# =============================================================================
# MODEL: Adaptive Multi-Scale Disentangled Fusion (AMSDF)
# =============================================================================
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambd)


class SEBlock(nn.Module):
    """Squeeze-and-excitation over feature maps."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class BandEncoder(nn.Module):
    """Band-specific EEG encoder -> temporal token sequence."""
    def __init__(self, n_channels, kernel_sec, d_model=48, F1=16, D=2,
                 pool1=8, dropout=0.30):
        super().__init__()
        k = int(FS * kernel_sec) // 2 * 2 + 1
        F2 = F1 * D
        self.temporal = nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.se = SEBlock(F2)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.sep = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 15), padding=(0, 7), groups=F2, bias=False),
            nn.Conv2d(F2, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.ELU())
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.bn1(self.temporal(x)))
        x = F.elu(self.bn2(self.spatial(x)))
        x = self.se(x)
        x = self.drop1(self.pool1(x))
        x = self.drop2(self.sep(x))
        return x.squeeze(2).transpose(1, 2)  # (B, T', D)


class AdaptiveTemporalFusion(nn.Module):
    """Learn w_b(t) across MRCP/Mu/Beta instead of fixed concatenation."""
    def __init__(self, n_bands, d_model=48):
        super().__init__()
        hidden = max(16, d_model // 2)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features):
        # stack: (B, bands, time, D)
        stack = torch.stack(features, dim=1)
        logits = self.score(stack).squeeze(-1)           # (B, bands, time)
        temporal_w = torch.softmax(logits, dim=1)
        fused = (temporal_w.unsqueeze(-1) * stack).sum(dim=1)
        fused = self.norm(fused)
        global_w = temporal_w.mean(dim=-1)
        return fused, temporal_w, global_w


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        a = torch.softmax(self.score(x).squeeze(-1), dim=1)
        z = torch.sum(a.unsqueeze(-1) * x, dim=1)
        return z, a


class AMSDFNet(nn.Module):
    """Adaptive Multi-Scale Disentangled Fusion.

    Movement representation z_movement is optimized for the 7-class task and
    the anatomical joint task. During cross-subject training, a GRL subject
    classifier discourages subject identity from leaking into z_movement while
    a parallel z_subject branch explicitly models subject identity.
    """
    def __init__(self, n_channels, band_subset=None, d_model=48,
                 n_classes=None, n_subjects=None, use_subject_branch=False):
        super().__init__()
        self.band_subset = (list(range(N_BANDS)) if band_subset is None
                            else list(band_subset))
        self.n_classes = N_CLASSES if n_classes is None else int(n_classes)
        self.use_subject_branch = bool(use_subject_branch)

        self.encoders = nn.ModuleList([
            BandEncoder(n_channels, FILTER_BANK[b][3], d_model=d_model)
            for b in self.band_subset
        ])
        self.fusion = AdaptiveTemporalFusion(len(self.band_subset), d_model)
        self.pool = AttentionPool(d_model)

        self.movement_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, D_LATENT),
            nn.GELU(), nn.Dropout(0.30))
        self.subject_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, D_LATENT),
            nn.GELU(), nn.Dropout(0.30))

        self.class_head = nn.Sequential(
            nn.LayerNorm(D_LATENT), nn.Dropout(0.30),
            nn.Linear(D_LATENT, self.n_classes))
        self.joint_head = nn.Sequential(
            nn.LayerNorm(D_LATENT), nn.Linear(D_LATENT, 4))

        if self.use_subject_branch:
            if n_subjects is None or int(n_subjects) < 2:
                raise ValueError('n_subjects>=2 required for subject branch.')
            self.grl = GradientReversal(GRL_LAMBDA)
            self.subject_adv_head = nn.Sequential(
                nn.LayerNorm(D_LATENT), nn.Linear(D_LATENT, int(n_subjects)))
            self.subject_head = nn.Sequential(
                nn.LayerNorm(D_LATENT), nn.Linear(D_LATENT, int(n_subjects)))
        else:
            self.grl = None
            self.subject_adv_head = None
            self.subject_head = None

    def forward(self, x):
        features = []
        for i, enc in enumerate(self.encoders):
            features.append(enc(x[:, self.band_subset[i]]))

        fused, temporal_w, global_w = self.fusion(features)
        pooled, temporal_attn = self.pool(fused)

        z_m = self.movement_proj(pooled)
        z_s = self.subject_proj(pooled)
        class_logits = self.class_head(z_m)
        joint_logits = self.joint_head(z_m)

        if self.use_subject_branch:
            subject_adv = self.subject_adv_head(self.grl(z_m))
            subject_latent = self.subject_head(z_s)
        else:
            subject_adv = None
            subject_latent = None

        return {
            'class_logits': class_logits,
            'joint_logits': joint_logits,
            'subject_adv': subject_adv,
            'subject_latent': subject_latent,
            'z_movement': z_m,
            'z_subject': z_s,
            'temporal_band_weights': temporal_w,
            'global_band_weights': global_w,
            'temporal_attention': temporal_attn,
        }


# Backward-compatible name for any downstream code that expects FBCATNet.
FBCATNet = AMSDFNet


class EEGNetBaseline(nn.Module):
    """Standard EEGNet-8,2 on the broadband stream."""
    def __init__(self, n_channels, samples=CROP_SAMPLES, F1=8, D=2, F2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, 65), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            nn.Conv2d(F1 * D, F1 * D, (1, 17), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.5), nn.Flatten())
        with torch.no_grad():
            n_feat = self.net(torch.zeros(1, 1, n_channels, samples)).shape[1]
        self.fc = nn.Linear(n_feat, N_CLASSES)

    def forward(self, x):
        logits = self.fc(self.net(x[:, 0].unsqueeze(1)))
        return {
            'class_logits': logits,
            'joint_logits': None,
            'subject_adv': None,
            'subject_latent': None,
            'z_movement': None,
            'z_subject': None,
            'temporal_band_weights': None,
            'global_band_weights': None,
            'temporal_attention': None,
        }


# =============================================================================
# TRAINING / EVALUATION
# =============================================================================
def make_scheduler(optimizer, epochs=EPOCHS, warmup=WARMUP_EPOCHS):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(out, y, subject_ids=None):
    cls = F.cross_entropy(out['class_logits'], y, label_smoothing=LABEL_SMOOTHING)
    total = cls

    joint = torch.tensor(0.0, device=y.device)
    if out.get('joint_logits') is not None:
        joint_y = torch.as_tensor(JOINT_LABELS[y.detach().cpu().numpy()],
                                  dtype=torch.long, device=y.device)
        joint = F.cross_entropy(out['joint_logits'], joint_y)
        total = total + JOINT_LOSS_WEIGHT * joint

    subj_adv = torch.tensor(0.0, device=y.device)
    subj_latent = torch.tensor(0.0, device=y.device)
    ortho = torch.tensor(0.0, device=y.device)
    if subject_ids is not None and out.get('subject_adv') is not None:
        subj_adv = F.cross_entropy(out['subject_adv'], subject_ids)
        subj_latent = F.cross_entropy(out['subject_latent'], subject_ids)
        zm = F.normalize(out['z_movement'], dim=1)
        zs = F.normalize(out['z_subject'], dim=1)
        ortho = torch.mean(torch.sum(zm * zs, dim=1) ** 2)
        total = (total + SUBJECT_ADV_WEIGHT * subj_adv
                       + SUBJECT_LATENT_WEIGHT * subj_latent
                       + ORTHOGONAL_WEIGHT * ortho)

    return total, {'cls': cls.detach(), 'joint': joint.detach(),
                   'subject_adv': subj_adv.detach(),
                   'subject_latent': subj_latent.detach(),
                   'ortho': ortho.detach()}


@torch.no_grad()
def evaluate_multicrop(model, loader, crop_samples=None, n_test_crops=None,
                       return_details=True):
    model.eval()
    cs = CROP_SAMPLES if crop_samples is None else int(crop_samples)
    nc = N_TEST_CROPS if n_test_crops is None else int(n_test_crops)
    max_start = max(0, EPOCH_SAMPLES - cs)
    starts = np.linspace(0, max_start, max(1, nc)).astype(int)

    y_true, y_pred, logprobs = [], [], []
    z_m, z_s, gw, tw = [], [], [], []

    for batch in loader:
        x, y = batch[:2]
        x = x.to(DEVICE)
        lp_sum = None
        zm_sum = None
        zs_sum = None
        gw_sum = None
        tw_sum = None

        for s in starts:
            out = model(x[..., s:s + cs])
            lp = F.log_softmax(out['class_logits'], dim=1)
            if lp_sum is None:
                lp_sum = lp
                zm_sum = out['z_movement']
                zs_sum = out['z_subject']
                gw_sum = out['global_band_weights']
                tw_sum = out['temporal_band_weights']
            else:
                lp_sum = lp_sum + lp
                zm_sum = zm_sum + out['z_movement']
                zs_sum = zs_sum + out['z_subject']
                gw_sum = gw_sum + out['global_band_weights']
                tw_sum = tw_sum + out['temporal_band_weights']

        n = float(len(starts))
        lp_avg = lp_sum / n
        zm_avg = zm_sum / n
        zs_avg = zs_sum / n
        gw_avg = gw_sum / n
        tw_avg = tw_sum / n

        y_true.extend(y.numpy().tolist())
        y_pred.extend(lp_avg.argmax(dim=1).cpu().numpy().tolist())
        logprobs.append(lp_avg.cpu().numpy())
        z_m.append(zm_avg.cpu().numpy())
        z_s.append(zs_avg.cpu().numpy())
        gw.append(gw_avg.cpu().numpy())
        tw.append(tw_avg.cpu().numpy())

    return {
        'true': np.asarray(y_true, dtype=np.int64),
        'pred': np.asarray(y_pred, dtype=np.int64),
        'logprobs': np.concatenate(logprobs, axis=0),
        'z_movement': np.concatenate(z_m, axis=0),
        'z_subject': np.concatenate(z_s, axis=0),
        'global_band_weights': np.concatenate(gw, axis=0),
        'temporal_band_weights': np.concatenate(tw, axis=0),
    }


def train_one_fold(model_fn, X, y, runs, test_run, seed_offset=0,
                   n_runs=None):
    torch.manual_seed(SEED + 1000 * seed_offset + test_run)
    np.random.seed(SEED + 1000 * seed_offset + test_run)
    random.seed(SEED + 1000 * seed_offset + test_run)

    available_runs = sorted(np.unique(runs).tolist())
    if test_run not in available_runs:
        raise ValueError(f'test_run {test_run} not present')

    train_runs = [r for r in available_runs if r != test_run]
    if len(train_runs) < 2:
        raise ValueError('Need at least two training runs for validation.')
    val_run = train_runs[(test_run - 1) % len(train_runs)]
    fit_runs = [r for r in train_runs if r != val_run]

    tr_m = np.isin(runs, fit_runs)
    va_m = runs == val_run
    te_m = runs == test_run

    tr_loader = DataLoader(BandDataset(X[tr_m], y[tr_m], train=True),
                           batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0)
    va_loader = DataLoader(BandDataset(X[va_m], y[va_m]),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0)
    te_loader = DataLoader(BandDataset(X[te_m], y[te_m]),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0)

    model = model_fn().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer)

    best_score, best_state, patience = -np.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss, _ = compute_loss(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val = evaluate_multicrop(model, va_loader,
                                 crop_samples=CROP_SAMPLES,
                                 n_test_crops=N_TEST_CROPS)
        score = balanced_accuracy_score(val['true'], val['pred'])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate_multicrop(model, te_loader,
                              crop_samples=CROP_SAMPLES,
                              n_test_crops=N_TEST_CROPS)

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return test


def loro_cross_validation(model_fn, X, y, runs, tag='', n_seeds=None):
    all_results = []
    if n_seeds is None:
        n_seeds = RUN_N_SEEDS

    for test_run in sorted(np.unique(runs).tolist()):
        seed_results = []
        for seed_idx in range(n_seeds):
            r = train_one_fold(model_fn, X, y, runs, test_run,
                               seed_offset=seed_idx)
            seed_results.append(r)

        lp = np.mean(np.stack([r['logprobs'] for r in seed_results]), axis=0)
        pred = lp.argmax(axis=1)
        base = seed_results[0]

        result = {
            'true': base['true'],
            'pred': pred,
            'runs': np.full(len(pred), test_run, dtype=np.int64),
            'z_movement': np.mean(np.stack([r['z_movement'] for r in seed_results]), axis=0),
            'z_subject': np.mean(np.stack([r['z_subject'] for r in seed_results]), axis=0),
            'global_band_weights': np.mean(np.stack([r['global_band_weights'] for r in seed_results]), axis=0),
            'temporal_band_weights': np.mean(np.stack([r['temporal_band_weights'] for r in seed_results]), axis=0),
        }
        all_results.append(result)
        print(f'    [{tag}] fold run{test_run:02d}: '
              f'acc {accuracy_score(result["true"], result["pred"])*100:.1f}%')

    return {
        'true': np.concatenate([r['true'] for r in all_results]),
        'pred': np.concatenate([r['pred'] for r in all_results]),
        'runs': np.concatenate([r['runs'] for r in all_results]),
        'z_movement': np.concatenate([r['z_movement'] for r in all_results]),
        'z_subject': np.concatenate([r['z_subject'] for r in all_results]),
        'global_band_weights': np.concatenate([r['global_band_weights'] for r in all_results]),
        'temporal_band_weights': np.concatenate([r['temporal_band_weights'] for r in all_results]),
    }



# =============================================================================
# OPTIONAL CROSS-SUBJECT DISENTANGLEMENT / RSA
# =============================================================================

def _stack_subject(subject_cache, subjects, exclude_last_run=True):
    Xs, ys, sids = [], [], []
    for sid in subjects:
        item = subject_cache[sid]
        X = np.stack([item['bands'][name] for name in BAND_NAMES], axis=1)
        y = item['y']
        runs = item['runs']
        if exclude_last_run:
            hold_run = sorted(np.unique(runs).tolist())[-1]
            mask = runs != hold_run
        else:
            mask = np.ones(len(y), dtype=bool)
        Xs.append(X[mask]); ys.append(y[mask])
        sids.append(np.full(mask.sum(), sid-1, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids)


def _stack_validation_runs(subject_cache, subjects):
    Xs, ys, sids = [], [], []
    for sid in subjects:
        item = subject_cache[sid]
        X = np.stack([item['bands'][name] for name in BAND_NAMES], axis=1)
        y = item['y']; runs = item['runs']
        hold_run = sorted(np.unique(runs).tolist())[-1]
        mask = runs == hold_run
        Xs.append(X[mask]); ys.append(y[mask])
        sids.append(np.full(mask.sum(), sid-1, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids)


def train_cross_subject_fold(subject_cache, test_subject, seed_offset=0):
    train_subjects = [s for s in subject_cache if s != test_subject]
    Xtr, ytr, sidtr = _stack_subject(subject_cache, train_subjects, True)
    Xva, yva, sidva = _stack_validation_runs(subject_cache, train_subjects)
    item = subject_cache[test_subject]
    Xte = np.stack([item['bands'][name] for name in BAND_NAMES], axis=1)
    yte = item['y']

    tr_ds = BandDataset(Xtr, ytr, train=True, subject_ids=sidtr)
    va_ds = BandDataset(Xva, yva, train=False, subject_ids=sidva)
    te_ds = BandDataset(Xte, yte, train=False)
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=0)

    set_seed(SEED + 5000*seed_offset + test_subject)
    model = AMSDFNet(N_CHANNELS, band_subset=[0,1,2], n_classes=N_CLASSES,
                     n_subjects=N_SUBJECTS, use_subject_branch=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = 60
    scheduler = make_scheduler(optimizer, epochs)
    best_score, best_state, patience = -np.inf, None, 0

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb, sb in tr_loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss, _ = compute_loss(out, yb, sb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        val = evaluate_multicrop(model, va_loader)
        score = balanced_accuracy_score(val['true'], val['pred'])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break

    model.load_state_dict(best_state)
    test = evaluate_multicrop(model, te_loader)
    out = metrics_dict(test['true'], test['pred'])

    del model, Xtr, ytr, sidtr, Xva, yva, sidva
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return out


def run_cross_subject_analysis(subject_cache):
    rows=[]
    for sid in sorted(subject_cache):
        ms=[]
        for seed_idx in range(RUN_N_SEEDS):
            ms.append(train_cross_subject_fold(subject_cache, sid, seed_idx))
        row={'Test_Subject':sid}
        for key in ['Accuracy','Balanced_Acc','Macro_F1','Kappa']:
            row[key]=float(np.mean([m[key] for m in ms]))
        rows.append(row)
        print(f'  LOSO S{sid:02d}: BalAcc={row["Balanced_Acc"]*100:.2f}%')
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR,'Analysis_cross_subject_LOSO.csv'), index=False)
    return df


def run_rsa_from_results(subject_results):
    rows=[]
    for sid,res in subject_results.items():
        z=res['z_movement']; y=res['true']
        present=[c for c in range(N_CLASSES) if np.sum(y==c)>=2]
        centers=np.stack([z[y==c].mean(axis=0) for c in present])
        n=len(present); rdm=np.zeros((n,n),float)
        for i in range(n):
            for j in range(n):
                a,b=centers[i],centers[j]
                if a.std()<1e-8 or b.std()<1e-8:
                    rdm[i,j]=1.0
                else:
                    rdm[i,j]=1.0-np.corrcoef(a,b)[0,1]
        tri=np.triu_indices(n,k=1); neural=rdm[tri]
        refs={name:[] for name in ['Joint','RestMovement','Antagonist']}
        for i in range(n):
            for j in range(n):
                ci,cj=present[i],present[j]
                refs['Joint'].append(0.0 if JOINT_LABELS[ci]==JOINT_LABELS[cj] else 1.0)
                refs['RestMovement'].append(0.0 if ((ci==6)==(cj==6)) else 1.0)
                ant=((ci,cj) in ANTAGONIST_PAIRS or (cj,ci) in ANTAGONIST_PAIRS)
                refs['Antagonist'].append(0.0 if ant or ci==cj else 1.0)
        for name,v in refs.items():
            ref=np.asarray(v).reshape(n,n)[tri]
            rho,p=spearmanr(neural,ref)
            rows.append({'Subject':sid,'Reference':name,'Spearman_Rho':rho,'P':p})
        np.save(os.path.join(SAVE_DIR,f'RSA_S{sid:02d}_RDM.npy'),rdm)
    df=pd.DataFrame(rows)
    p=df['P'].to_numpy(float); q=np.full_like(p,np.nan)
    ok=np.isfinite(p)
    if ok.any():
        order=np.argsort(p[ok]); sp=p[ok][order]; m=len(sp)
        adj=np.minimum.accumulate((sp*m/np.arange(1,m+1))[::-1])[::-1]
        tmp=np.empty_like(adj); tmp[order]=np.minimum(adj,1.0); q[ok]=tmp
    df['FDR_q']=q
    df.to_csv(os.path.join(SAVE_DIR,'Analysis_RSA_subjectwise.csv'),index=False)
    return df

# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    print('=' * 90)
    print('  OFNER 2017 | AMSDF | Adaptive Multi-Scale Disentangled Fusion')
    print(f'  Device: {DEVICE} | fs={FS} Hz | epoch={EPOCH_SEC}s | crop={CROP_SEC}s')
    print(f'  Bands: {BAND_NAMES} | EA={"ON" if APPLY_EA else "OFF"} | seeds={RUN_N_SEEDS}')
    print('=' * 90)

    rows = []
    subject_results = {}
    subject_cache = {}
    variant_acc = {'AMSDF': []}
    if RUN_ABLATIONS:
        for b, name in enumerate(BAND_NAMES):
            variant_acc[f'Only_{name}'] = []
    if RUN_BASELINE:
        variant_acc['EEGNet'] = []

    class_gates = {c: [] for c in range(N_CLASSES)}
    all_true, all_pred = [], []

    for subject in RUN_SUBJECTS:
        print(f'\nSubject {subject:02d} ' + '-' * 58)
        try:
            bands, y, runs = load_subject(subject, n_runs=RUN_N_RUNS)
        except Exception as e:
            print(f'  [error] loading failed: {e}')
            continue

        # Keep a copy only when cross-subject LOSO is requested.
        if RUN_CROSS_SUBJECT:
            subject_cache[subject] = {'bands': {k: v.copy() for k, v in bands.items()},
                                      'y': y.copy(), 'runs': runs.copy()}

        X_fb = np.stack([bands[name] for name in BAND_NAMES], axis=1)
        X_broad = bands['Broad'][:, None]
        del bands
        gc.collect()

        # ---------------- Main AMSDF ----------------
        result = loro_cross_validation(
            lambda: AMSDFNet(N_CHANNELS, band_subset=[0, 1, 2],
                             n_classes=N_CLASSES),
            X_fb, y, runs, tag='AMSDF', n_seeds=RUN_N_SEEDS)

        m = metrics_dict(result['true'], result['pred'])
        print(f'  >> S{subject:02d} AMSDF | '
              f'Acc {m["Accuracy"]*100:.2f}% | '
              f'BalAcc {m["Balanced_Acc"]*100:.2f}% | '
              f'F1 {m["Macro_F1"]*100:.2f}% | '
              f'kappa {m["Kappa"]:.3f}')

        rows.append({'Subject': f'S{subject:02d}',
                     'Accuracy': m['Accuracy']*100,
                     'Balanced_Acc': m['Balanced_Acc']*100,
                     'Macro_F1': m['Macro_F1']*100,
                     'Kappa': m['Kappa']})
        variant_acc['AMSDF'].append(m['Accuracy']*100)
        subject_results[subject] = result
        all_true.extend(result['true']); all_pred.extend(result['pred'])

        for cls, gw in zip(result['true'], result['global_band_weights']):
            class_gates[int(cls)].append(gw)

        if SAVE_REPRESENTATIONS:
            np.savez_compressed(
                os.path.join(SAVE_DIR, f'S{subject:02d}_AMSDF_heldout_representation.npz'),
                true=result['true'], pred=result['pred'], runs=result['runs'],
                z_movement=result['z_movement'], z_subject=result['z_subject'],
                global_band_weights=result['global_band_weights'],
                temporal_band_weights=result['temporal_band_weights'])

        # ---------------- Optional single-band ablations ----------------
        if RUN_ABLATIONS:
            for b, name in enumerate(BAND_NAMES):
                res_b = loro_cross_validation(
                    lambda b=b: AMSDFNet(N_CHANNELS, band_subset=[b],
                                         n_classes=N_CLASSES),
                    X_fb, y, runs, tag=f'Only_{name}', n_seeds=RUN_N_SEEDS)
                ab = accuracy_score(res_b['true'], res_b['pred']) * 100
                variant_acc[f'Only_{name}'].append(ab)
                print(f'  >> S{subject:02d} Only_{name}: {ab:.2f}%')

        # ---------------- Optional EEGNet baseline -----------------------
        if RUN_BASELINE:
            res_e = loro_cross_validation(
                lambda: EEGNetBaseline(N_CHANNELS),
                X_broad, y, runs, tag='EEGNet', n_seeds=RUN_N_SEEDS)
            ae = accuracy_score(res_e['true'], res_e['pred']) * 100
            variant_acc['EEGNet'].append(ae)
            print(f'  >> S{subject:02d} EEGNet: {ae:.2f}%')

        del X_fb, X_broad
        gc.collect()

    # ---------------- Tables ---------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, 'Table1_AMSDF_subjectwise.csv'), index=False)

    summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Kappa'],
        'Mean': [df['Accuracy'].mean(), df['Balanced_Acc'].mean(),
                 df['Macro_F1'].mean(), df['Kappa'].mean()],
        'SD': [df['Accuracy'].std(ddof=1), df['Balanced_Acc'].std(ddof=1),
               df['Macro_F1'].std(ddof=1), df['Kappa'].std(ddof=1)]})
    summary.to_csv(os.path.join(SAVE_DIR, 'Table1_AMSDF_group_summary.csv'), index=False)

    comp = []
    main = np.asarray(variant_acc['AMSDF'], dtype=float)
    for name, vals in variant_acc.items():
        vals = np.asarray(vals, dtype=float)
        row = {'Model': name, 'Mean_Acc': vals.mean(), 'SD': vals.std(ddof=1) if len(vals)>1 else np.nan}
        if name != 'AMSDF' and len(vals) == len(main) and len(vals) > 1:
            try:
                _, p = wilcoxon(main, vals)
                row['Wilcoxon_p_vs_AMSDF'] = p
            except Exception:
                row['Wilcoxon_p_vs_AMSDF'] = np.nan
        comp.append(row)
    pd.DataFrame(comp).to_csv(os.path.join(SAVE_DIR, 'Table2_AMSDF_comparison.csv'), index=False)

    # ---------------- Figures --------------------------------------------
    if all_true:
        cm = confusion_matrix(all_true, all_pred,
                              labels=np.arange(N_CLASSES), normalize='true') * 100
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('AMSDF — normalized confusion matrix (%)')
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'Fig1_AMSDF_confusion_matrix.png'), dpi=300)
        plt.close()

    # Mean adaptive band usage per class.
    W = np.zeros((N_CLASSES, N_BANDS), dtype=float)
    for c in range(N_CLASSES):
        if class_gates[c]:
            W[c] = np.mean(np.stack(class_gates[c]), axis=0)

    plt.figure(figsize=(9.5, 5.5))
    x = np.arange(N_CLASSES); width = 0.25
    for b, name in enumerate(BAND_NAMES):
        plt.bar(x + (b-1)*width, W[:, b], width, label=name)
    plt.xticks(x, CLASS_NAMES, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel('Mean adaptive band weight')
    plt.title('AMSDF — per-class adaptive multi-scale fusion')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Fig2_AMSDF_class_band_weights.png'), dpi=300)
    plt.close()

    # Subject-wise performance.
    if len(rows):
        plt.figure(figsize=(10, 5))
        xs = np.arange(len(rows))
        plt.bar(xs, [r['Balanced_Acc'] for r in rows])
        plt.axhline(100/N_CLASSES, color='k', ls='--', lw=1, label='Chance 14.3%')
        plt.xticks(xs, [r['Subject'] for r in rows])
        plt.ylabel('Balanced accuracy (%)')
        plt.title('AMSDF — subject-wise LORO balanced accuracy')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'Fig3_AMSDF_subjectwise_balacc.png'), dpi=300)
        plt.close()

    if RUN_CROSS_SUBJECT:
        print('\nRunning cross-subject LOSO / disentanglement...')
        run_cross_subject_analysis({k:v for k,v in subject_cache.items() if k in RUN_SUBJECTS})

    if RUN_RSA and subject_results:
        print('\nRunning RSA on held-out movement representations...')
        run_rsa_from_results(subject_results)

    # Configuration record.
    cfg = {
        'model': 'AMSDF',
        'D_LATENT': D_LATENT,
        'JOINT_LOSS_WEIGHT': JOINT_LOSS_WEIGHT,
        'SUBJECT_ADV_WEIGHT': SUBJECT_ADV_WEIGHT,
        'SUBJECT_LATENT_WEIGHT': SUBJECT_LATENT_WEIGHT,
        'ORTHOGONAL_WEIGHT': ORTHOGONAL_WEIGHT,
        'GRL_LAMBDA': GRL_LAMBDA,
        'FILTER_BANK': FILTER_BANK,
        'APPLY_EA': APPLY_EA,
        'USE_TRANSFORMER': USE_TRANSFORMER,
        'N_SEEDS': N_SEEDS,
        'DEVICE': DEVICE,
        'RUN_CROSS_SUBJECT': RUN_CROSS_SUBJECT,
        'RUN_RSA': RUN_RSA,
        'RUN_SUBJECTS': RUN_SUBJECTS,
        'RUN_N_RUNS': RUN_N_RUNS,
        'RUN_N_SEEDS': RUN_N_SEEDS,
    }
    with open(os.path.join(SAVE_DIR, 'config_AMSDF.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    print('\n' + '=' * 90)
    if len(main) > 0:
        print(f'AMSDF mean Acc = {main.mean():.2f}% +/- {main.std(ddof=1) if len(main)>1 else 0:.2f}%')
    print(f'Chance = {100/N_CLASSES:.1f}%')
    print(f'All AMSDF results saved to: {SAVE_DIR}')
    print('=' * 90)


if __name__ == '__main__':
    main()
