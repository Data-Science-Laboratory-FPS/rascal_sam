import os
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, fbeta_score,
    matthews_corrcoef, roc_auc_score
)
import random
import copy
 
# ===============================
# SEED
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
# ===============================
# DATASET
# Images are loaded in grayscale (1 channel) to match the expected
# input of the torchxrayvision backbone trained on single-channel CXRs.
# ===============================
class ImageDataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row["filename"])
        label = row["label"]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
 
 
# ===============================
# BACKBONE — DenseNet121 pretrained on MIMIC-CXR (torchxrayvision)
#
# The Madrid reference paper uses a DenseNet121 pretrained on MIMIC-CXR
# for the 14 CheXpert labels (mimic_gen_aug_epoch5.h5).
# We replicate this using torchxrayvision, the closest publicly available
# equivalent: DenseNet121 trained on MIMIC-CXR.
#
# layers_not_trainable: number of backbone parameters to freeze.
# The reference notebook treats this as a tunable hyperparameter.
# We use 0 (full fine-tuning) as the starting point.
# ===============================
try:
    import torchxrayvision as txrv
    USE_TXRV = True
except ImportError:
    USE_TXRV = False
    print("[WARNING] torchxrayvision is not installed.")
    print("          Install with: pip install torchxrayvision")
    print("          Falling back to ImageNet DenseNet121 (not recommended).")
 
 
class DenseNetCXR(nn.Module):
    """
    DenseNet121 pretrained on MIMIC-CXR (torchxrayvision).
    Equivalent to the backbone used in the Madrid reference paper.
 
    layers_not_trainable : number of backbone parameters to freeze
                           (0 = full fine-tuning).
    dropout              : dropout probability before the final classifier.
    """
    def __init__(self, layers_not_trainable=0, dropout=0.1):
        super().__init__()
 
        if USE_TXRV:
            base = txrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
            for i, (_, param) in enumerate(base.named_parameters()):
                if i < layers_not_trainable:
                    param.requires_grad = False
            self.features = base.features
        else:
            base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.features = base.features
 
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, 1)
 
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
 
 
# ===============================
# MODEL SELECTION CRITERION
#
# Both model selection and threshold selection use balanced accuracy
# (mean of sensitivity and specificity) for consistency.
# ===============================
def _balanced_accuracy(y_true, y_probs, threshold=0.5):
    """Balanced accuracy: mean of sensitivity and specificity."""
    y_pred = (y_probs >= threshold).astype(int)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
    return (sens + spec) / 2
 
 
# ===============================
# RESOLUTION
#
# The reference paper uses 320×320 px instead of 224×224.
# Higher resolution preserves fine radiological signal
# (consolidations, infiltrates) that may be lost at lower resolution.
# ===============================
IMG_SIZE = 320
 
 
# ===============================
# ENSEMBLE CONFIGURATION
#
# The reference paper selects the top-10 checkpoints per run,
# repeats training 3 times, and averages predictions across
# 30 checkpoints in total.
# ===============================
TOP_K_CHECKPOINTS = 10   # best checkpoints saved per seed
N_SEEDS           = 3    # training repetitions per fold
ENSEMBLE_SEEDS    = [42, 123, 7]  # fixed seeds for reproducibility
 
 
def train_model_with_checkpoints(model, train_loader, val_loader, device,
                                  epochs=10, lr=1e-4, top_k=TOP_K_CHECKPOINTS):
    """
    Trains the model and saves the top_k best checkpoints ranked by
    balanced accuracy on the validation set.
    Returns a list of state_dicts sorted from best to worst.
    """
    model.to(device)
 
    # Compute pos_weight and initial bias for class imbalance
    labels = []
    for _, y in train_loader:
        labels.extend(y.numpy())
    labels     = np.array(labels)
    n_pos      = labels.sum()
    n_neg      = len(labels) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos).to(device)
 
    p        = n_pos / (n_pos + n_neg)
    bias_val = np.log(p / (1 - p))
    model.classifier.bias.data = torch.tensor([bias_val]).float().to(device)
 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    checkpoint_pool = []  # list of (score, state_dict)
 
    for epoch in range(epochs):
        model.train()
        running_loss = 0
 
        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
 
        val_probs, y_val = predict_from_model(model, val_loader, device)
        score = _balanced_accuracy(y_val, val_probs, threshold=0.5)
 
        checkpoint_pool.append((score, copy.deepcopy(model.state_dict())))
 
        print(f"  Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} "
              f"| Val BalAcc: {score:.3f}")
 
    # Keep only the top_k checkpoints by balanced accuracy
    checkpoint_pool.sort(key=lambda x: x[0], reverse=True)
    top_checkpoints = [sd for _, sd in checkpoint_pool[:top_k]]
 
    print(f"  → {len(top_checkpoints)} checkpoints selected "
          f"(best BalAcc: {checkpoint_pool[0][0]:.3f})")
 
    return top_checkpoints
 
 
def predict_from_model(model, dataloader, device):
    """Generate predictions from a single model."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x      = x.to(device)
            logits = model(x).squeeze(1)
            p      = torch.sigmoid(logits)
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(probs), np.array(labels)
 
 
def predict_ensemble(checkpoints_list, val_loader, device,
                     layers_not_trainable=0, dropout=0.1):
    """
    Averages predictions across all checkpoints in the ensemble.
    checkpoints_list: list of state_dicts from all seeds and top_k combined.
    """
    all_probs  = []
    labels_ref = None
 
    for sd in checkpoints_list:
        model = DenseNetCXR(
            layers_not_trainable=layers_not_trainable,
            dropout=dropout
        )
        model.load_state_dict(sd)
        model.to(device)
        probs, labels = predict_from_model(model, val_loader, device)
        all_probs.append(probs)
        if labels_ref is None:
            labels_ref = labels
 
    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs, labels_ref
 
 
# ===============================
# PREDICT (alias for Grad-CAM compatibility)
# ===============================
def predict(model, dataloader, device):
    return predict_from_model(model, dataloader, device)
 
 
# ===============================
# CLINICAL THRESHOLD SELECTION
#
# Searches for the threshold that maximises (sensitivity + specificity) / 2.
# Always applied to validation set probabilities, never to training set.
# ===============================
def find_threshold_clinical(y_true, y_probs):
    thresholds = np.linspace(0.01, 0.9, 200)
    best_t     = 0.5
    best_score = -1
 
    for t in thresholds:
        score = _balanced_accuracy(y_true, y_probs, threshold=t)
        if score > best_score:
            best_score = score
            best_t     = t
 
    return best_t
 
 
# ===============================
# BOOTSTRAP 95% CI
#
# Computes metrics with 95% confidence intervals via bootstrap resampling
# on concatenated out-of-fold predictions, following the Madrid reference paper.
# ===============================
def bootstrap_ci(y_true, y_probs, threshold, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Computes bootstrap 95% CI for all classification metrics.
    Returns a dict with mean, ci_low, and ci_high per metric.
    """
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    metrics_boot = {
        "Recall": [], "Specificity": [], "BalAcc": [],
        "Precision": [], "F1": [], "F2": [], "MCC": [], "AUC": []
    }
 
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt  = y_true[idx]
        yp  = y_probs[idx]
 
        if len(np.unique(yt)) < 2:
            continue
 
        yp_bin = (yp >= threshold).astype(int)
        sens   = recall_score(yt, yp_bin, zero_division=0)
        spec   = recall_score(1 - yt, 1 - yp_bin, zero_division=0)
 
        metrics_boot["Recall"].append(sens)
        metrics_boot["Specificity"].append(spec)
        metrics_boot["BalAcc"].append((sens + spec) / 2)
        metrics_boot["Precision"].append(precision_score(yt, yp_bin, zero_division=0))
        metrics_boot["F1"].append(f1_score(yt, yp_bin, zero_division=0))
        metrics_boot["F2"].append(fbeta_score(yt, yp_bin, beta=2, zero_division=0))
        metrics_boot["MCC"].append(matthews_corrcoef(yt, yp_bin))
        metrics_boot["AUC"].append(roc_auc_score(yt, yp))
 
    alpha   = (1 - ci) / 2
    results = {}
    for metric, values in metrics_boot.items():
        values = np.array(values)
        results[metric] = {
            "mean":    np.mean(values),
            "ci_low":  np.percentile(values, alpha * 100),
            "ci_high": np.percentile(values, (1 - alpha) * 100)
        }
    return results
 
 
def print_bootstrap_results(ci_results, experiment_name=""):
    print(f"\n===== BOOTSTRAP 95% CI — {experiment_name} =====")
    print(f"{'Metric':<14} {'Mean':>8} {'CI low':>12} {'CI high':>12}")
    print("-" * 50)
    for metric, vals in ci_results.items():
        print(f"{metric:<14} {vals['mean']:>8.3f} "
              f"{vals['ci_low']:>12.3f} {vals['ci_high']:>12.3f}")
 
 
# ===============================
# EVALUATION (per-fold point estimates)
# ===============================
def evaluate(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    sens   = recall_score(y_true, y_pred, zero_division=0)
    spec   = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
    return {
        "Recall":      sens,
        "Specificity": spec,
        "BalAcc":      (sens + spec) / 2,
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "F2":          fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "AUC":         roc_auc_score(y_true, y_probs),
        "ConfMatrix":  confusion_matrix(y_true, y_pred),
        "Threshold":   threshold
    }
 
 
# ===============================
# MAIN PIPELINE — 4-FOLD WITH ENSEMBLE
#
# Changes vs v2:
#   1. Resolution 320×320 instead of 224×224
#   2. Each fold is trained N_SEEDS times (3 seeds)
#   3. Top-10 checkpoints saved per seed (by BalAcc on validation)
#   4. Final probabilities are the average of 30 checkpoints
#      (3 seeds × 10 checkpoints)
# ===============================
def run_pipeline_4fold(df, base_dir, experiment_name, batch_size=8,
                       layers_not_trainable=0, dropout=0.1, lr=1e-4, epochs=10):
    """
    Stratified 4-fold pipeline with checkpoint ensemble.
 
    Per fold:
      - Training is repeated N_SEEDS times with different random seeds
      - Top TOP_K_CHECKPOINTS checkpoints are saved per seed (by BalAcc)
      - Final probabilities are averaged across N_SEEDS × TOP_K checkpoints
 
    Parameters
    ----------
    layers_not_trainable : backbone layers to freeze (0 = full fine-tuning)
    dropout              : dropout before the classifier
    lr                   : learning rate
    epochs               : training epochs per seed
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {'MIMIC-CXR (torchxrayvision)' if USE_TXRV else 'ImageNet (fallback)'}")
    print(f"Resolution: {IMG_SIZE}×{IMG_SIZE} | N_SEEDS={N_SEEDS} | TOP_K={TOP_K_CHECKPOINTS}")
    print(f"layers_not_trainable={layers_not_trainable}, dropout={dropout}, lr={lr}")
    print(f"Total checkpoints per fold: {N_SEEDS * TOP_K_CHECKPOINTS}")
 
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
 
    skf         = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    all_metrics = []
    all_y_true  = []
    all_y_probs = []
 
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
 
        df_train = df.iloc[train_idx]
        df_val   = df.iloc[val_idx]
 
        train_ds = ImageDataset(df_train, base_dir, train_tf)
        val_ds   = ImageDataset(df_val,   base_dir, val_tf)
 
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
 
        out_dir = os.path.join("experiments", experiment_name, f"fold_{fold}")
        os.makedirs(out_dir, exist_ok=True)
 
        fold_checkpoints = []
 
        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            print(f"\n  --- Seed {seed_idx+1}/{N_SEEDS} (seed={seed}) ---")
            set_seed(seed)
 
            model = DenseNetCXR(
                layers_not_trainable=layers_not_trainable,
                dropout=dropout
            )
 
            top_checkpoints = train_model_with_checkpoints(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, top_k=TOP_K_CHECKPOINTS
            )
            fold_checkpoints.extend(top_checkpoints)
 
            seed_dir = os.path.join(out_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            for ck_idx, sd in enumerate(top_checkpoints):
                torch.save(sd, os.path.join(seed_dir, f"checkpoint_{ck_idx:02d}.pt"))
 
        print(f"\n  Ensemble: {len(fold_checkpoints)} total checkpoints for fold {fold}")
 
        val_probs, y_val = predict_ensemble(
            fold_checkpoints, val_loader, device,
            layers_not_trainable=layers_not_trainable,
            dropout=dropout
        )
 
        best_threshold = find_threshold_clinical(y_val, val_probs)
 
        metrics = evaluate(y_val, val_probs, best_threshold)
        metrics["fold"]          = fold
        metrics["n_checkpoints"] = len(fold_checkpoints)
        all_metrics.append(metrics)
 
        all_y_true.append(y_val)
        all_y_probs.append(val_probs)
 
        np.save(os.path.join(out_dir, "y_true.npy"),    y_val)
        np.save(os.path.join(out_dir, "y_probs.npy"),   val_probs)
        np.save(os.path.join(out_dir, "threshold.npy"), np.array(best_threshold))
        np.save(os.path.join(out_dir, "val_idx.npy"),   val_idx)
        np.save(os.path.join(out_dir, "filenames.npy"), df_val["filename"].values)
 
    # ===============================
    # FINAL RESULTS
    # ===============================
    df_results = pd.DataFrame(all_metrics)
 
    print("\n===== RESULTS PER FOLD =====")
    print(df_results[["fold", "Recall", "Specificity", "BalAcc",
                       "F1", "F2", "AUC", "MCC", "Threshold"]].to_string(index=False))
 
    print("\n===== MEAN ± STD =====")
    numeric_cols = ["Recall", "Specificity", "BalAcc", "F1", "F2",
                    "AUC", "MCC", "Threshold"]
    for col in numeric_cols:
        print(f"  {col:<14}: {df_results[col].mean():.3f} ± {df_results[col].std():.3f}")
 
    all_y_true_cat   = np.concatenate(all_y_true)
    all_y_probs_cat  = np.concatenate(all_y_probs)
    global_threshold = df_results["Threshold"].mean()
 
    ci_results = bootstrap_ci(all_y_true_cat, all_y_probs_cat,
                               threshold=global_threshold,
                               n_bootstrap=1000)
    print_bootstrap_results(ci_results, experiment_name=experiment_name)
 
    out_exp = os.path.join("experiments", experiment_name)
    df_results.to_csv(os.path.join(out_exp, "fold_results.csv"), index=False)
    np.save(os.path.join(out_exp, "bootstrap_ci.npy"), ci_results)
 
    return df_results, ci_results