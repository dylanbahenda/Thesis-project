import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, brier_score_loss, balanced_accuracy_score,
    multilabel_confusion_matrix
)

def multilabel_sensitivity(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    return np.mean(sensitivity)

def multilabel_specificity(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    return np.mean(specificity)

def multilabel_balanced_accuracy(y_true, y_pred):
    sens = multilabel_sensitivity(y_true, y_pred)
    spec = multilabel_specificity(y_true, y_pred)
    return (sens + spec) / 2

def compute_all_metrics_with_ci(y_true, y_pred, y_prob=None, n_bootstraps=1000, ci=90, seed=42, multilabel=False):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
    
    n = len(y_true)
    avg_mode = 'micro' if multilabel else 'binary'

    metrics = {
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'f1': lambda yt, yp: f1_score(yt, yp, average=avg_mode),
        'auroc': lambda yt, yp: roc_auc_score(yt, yp, average=avg_mode) if y_prob is not None else None,
        'mcc': lambda yt, yp: matthews_corrcoef(yt.flatten(), yp.flatten()),
        'brier': lambda yt, yp: np.mean([
            brier_score_loss(yt[:, i], yp[:, i]) for i in range(yt.shape[1])
        ]) if y_prob is not None else None,
        'sensitivity': multilabel_sensitivity if multilabel else None,
        'specificity': multilabel_specificity if multilabel else None,
        'balanced_accuracy': multilabel_balanced_accuracy if multilabel else lambda yt, yp: balanced_accuracy_score(yt, yp)
    }

    results = {}

    for metric_name, metric_fn in metrics.items():
        if metric_fn is None:
            results[metric_name] = (None, (None, None))
            continue

        scores = []
        for _ in range(n_bootstraps):
            indices = rng.choice(n, n, replace=True)
            yt_sample = y_true[indices]
            if metric_name in ['auroc', 'brier']:
                yp_sample = y_prob[indices]
            else:
                yp_sample = y_pred[indices]

            try:
                score = metric_fn(yt_sample, yp_sample)
                if score is not None and not np.isnan(score):
                    scores.append(score)
            except Exception:
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
