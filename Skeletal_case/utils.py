
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, brier_score_loss
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms


image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def compute_metrics(pred):
    logits, labels = pred
    preds = (logits >= 0.5).astype(int) 
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, preds)
    }


def compute_additional_metrics(y_true, y_pred, threshold=0.5):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Sensitivity (Recall), Specificity, Balanced Accuracy
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "f1": f1_score(y_true, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": cm
    }   


def plot_confusion_matrix(cm, class_names=["No Fracture", "Fracture"]):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()



def fracture_collate_fn(batch):
    """
    batch: list of dicts from FractureDataset.__getitem__
    Returns:
        - batched and padded tensors for images, input_ids, attention_mask
        - labels
        - optional: case_id for debugging
    """

    # === 1. Handle images (variable number per case) ===
    all_images = []
    image_counts = []
    for item in batch:
        imgs = item["pixel_values"]  # list of tensors
        image_counts.append(len(imgs))
        all_images.extend(imgs)

    # Stack all images, shape: [total_images, 3, H, W]
    all_images = torch.stack(all_images)

    # Pad image sets so we can batch them as [B, N, C, H, W]
    max_images = max(image_counts)
    padded_images = []
    idx = 0
    for count in image_counts:
        imgs = all_images[idx:idx + count]
        idx += count
        if count < max_images:
            pad = imgs[0].unsqueeze(0).repeat(max_images - count, 1, 1, 1)  # replicate first image
            imgs = torch.cat([imgs, pad], dim=0)
        padded_images.append(imgs)

    images_tensor = torch.stack(padded_images)  # [B, max_N, 3, H, W]

    # === 2. Tokenized text ===
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # === 3. Labels ===
    labels = torch.stack([item["label"] for item in batch])

    # === 4. Case IDs (for debugging/traceability) ===
    case_ids = [item["case_id"] for item in batch]

    return {
        "pixel_values": images_tensor,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "case_ids": case_ids
    }

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, brier_score_loss, balanced_accuracy_score,
    confusion_matrix
)

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# def compute_all_metrics_with_ci(y_true, y_pred, y_prob=None, n_bootstraps=1000, ci=90, seed=42, multilabel=False):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    # Choose average mode for multilabel vs binary
    avg_mode = 'micro' if multilabel else 'binary'
    
    # Metric functions to apply:
    metrics = {
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'f1': lambda yt, yp: f1_score(yt, yp, average=avg_mode),
        'auroc': lambda yt, yp_prob: roc_auc_score(yt, yp_prob, average=avg_mode) if y_prob is not None else None,
        'mcc': lambda yt, yp: matthews_corrcoef(yt, yp),
        'brier': lambda yt, yp: brier_score_loss(yt, yp) if y_prob is not None else None,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'balanced_accuracy': lambda yt, yp: balanced_accuracy_score(yt, yp)
    }
    
    results = {}
    
    # Prepare arrays for bootstrapping
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
    
    for metric_name, metric_fn in metrics.items():
        if metric_fn is None:
            results[metric_name] = (None, (None, None))
            continue

        scores = []
        for _ in range(n_bootstraps):
            indices = rng.choice(n, n, replace=True)
            yt_sample = y_true[indices]
            yp_sample = y_pred[indices]
            yp_prob_sample = y_prob[indices] if y_prob is not None else None

            try:
                if metric_name == 'auroc':
                    score = metric_fn(yt_sample, yp_sample, yp_prob_sample)
                else:
                    score = metric_fn(yt_sample, yp_sample)
                if score is not None and not np.isnan(score):
                    scores.append(score)
            except Exception:
                continue
        
        if len(scores) == 0:
            # Unable to compute metric for bootstrap samples
            results[metric_name] = (None, (None, None))
            continue
        
        alpha = (100 - ci) / 2
        lower = np.percentile(scores, alpha)
        upper = np.percentile(scores, 100 - alpha)
        mean_score = np.mean(scores)
        results[metric_name] = (mean_score, (lower, upper))
    
    return results

def compute_all_metrics_with_ci(y_true, y_pred, y_prob=None, n_bootstraps=1000, ci=90, seed=42, multilabel=False):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    avg_mode = 'micro' if multilabel else 'binary'
    metrics = {
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'f1': lambda yt, yp: f1_score(yt, yp, average=avg_mode),
        'auroc': lambda yt, yp, yp_prob: roc_auc_score(yt, yp_prob) if yp_prob is not None else None,
        'mcc': lambda yt, yp: matthews_corrcoef(yt, yp),
        'brier': lambda yt, yp: brier_score_loss(yt, yp) if y_prob is not None else None,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'balanced_accuracy': lambda yt, yp: balanced_accuracy_score(yt, yp)
    }
    results = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
    for metric_name, metric_fn in metrics.items():
        if metric_fn is None:
            results[metric_name] = (None, (None, None))
            continue
        scores = []
        for _ in range(n_bootstraps):
            indices = rng.choice(n, n, replace=True)
            yt_sample = y_true[indices]
            yp_sample = y_pred[indices]
            yp_prob_sample = y_prob[indices] if y_prob is not None else None
            try:
                if metric_name == 'auroc':
                    score = metric_fn(yt_sample, yp_sample, yp_prob_sample)
                else:
                    score = metric_fn(yt_sample, yp_sample)
                if score is not None and not np.isnan(score):
                    scores.append(score)
            except Exception as e:
                print(f"Error computing {metric_name} for bootstrap sample")
                print(e)
                continue
        if len(scores) == 0:
            results[metric_name] = (None, (None, None))
            continue
        alpha = (100 - ci) / 2
        lower = np.percentile(scores, alpha)
        upper = np.percentile(scores, 100 - alpha)
        mean_score = np.mean(scores)
        results[metric_name] = (mean_score, (lower, upper))
    return results


