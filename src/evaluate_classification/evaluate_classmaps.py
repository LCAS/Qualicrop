import numpy as np
from data.load_data import example_data_load
from typing import List, Optional, Dict, Any


def pixelwise_metrics(gt_maps: List[np.ndarray],
                      pred_maps: List[np.ndarray],
                      ignore_class: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate multiple classification maps (pixel-wise).

    Parameters
    ----------
    gt_maps : list of HxW integer arrays
        Ground-truth label maps.
    pred_maps : list of HxW integer arrays
        Predicted label maps.
    ignore_class : int or None, optional
        Class label to ignore. Pixels are dropped if *either* y_true or y_pred
        equals this class.

    Returns
    -------
    results : dict
        {
          'classes': np.ndarray[K],
          'conf_mat': np.ndarray[K,K],
          'precision': np.ndarray[K] (per-class),
          'recall': np.ndarray[K] (per-class),
          'f1': np.ndarray[K] (per-class),
          'macroPrecision': float,
          'macroRecall': float,
          'macroF1': float,
          'microPrecision': float,
          'microRecall': float,
          'microF1': float,
          'mAP': float,                    # == macroPrecision
          'accuracy': float,
          'kappa': float,                  # Cohen's Kappa
          'mcc': float                     # Multiclass MCC
        }
    """
    if len(gt_maps) != len(pred_maps):
        raise ValueError("gt_maps and pred_maps must have the same length.")

    # Collect and mask pixels from all maps
    y_true_all = []
    y_pred_all = []

    for k, (yt, yp) in enumerate(zip(gt_maps, pred_maps), start=1):
        if yt.shape != yp.shape:
            raise ValueError(f"Map {k} has mismatched sizes between true and pred: {yt.shape} vs {yp.shape}.")

        if ignore_class is None:
            mask = np.ones(yt.shape, dtype=bool)
        else:
            mask = (yt != ignore_class) & (yp != ignore_class)

        y_true_all.append(yt[mask].ravel())
        y_pred_all.append(yp[mask].ravel())

    if not y_true_all:  # no maps
        raise ValueError("No maps provided.")

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    if y_true.size == 0:
        # After masking nothing remains
        # Return NaNs with empty structures to avoid crashes.
        return {
            'classes': np.array([], dtype=int),
            'conf_mat': np.zeros((0, 0), dtype=np.int64),
            'precision': np.array([]),
            'recall': np.array([]),
            'f1': np.array([]),
            'macroPrecision': np.nan,
            'macroRecall': np.nan,
            'macroF1': np.nan,
            'microPrecision': np.nan,
            'microRecall': np.nan,
            'microF1': np.nan,
            'mAP': np.nan,
            'accuracy': np.nan,
            'kappa': np.nan,
            'mcc': np.nan
        }

    # Identify classes present across true and pred
    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    K = classes.size
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Confusion matrix (K x K) with rows=true, cols=pred
    conf_mat = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_mat[class_to_idx[t], class_to_idx[p]] += 1

    # Per-class metrics
    TP = np.diag(conf_mat).astype(float)
    FP = conf_mat.sum(axis=0).astype(float) - TP
    FN = conf_mat.sum(axis=1).astype(float) - TP

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where((TP + FP) > 0, TP / (TP + FP), np.nan)
        recall    = np.where((TP + FN) > 0, TP / (TP + FN), np.nan)
        f1        = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)

    # Macro averages (ignore NaNs)
    def nanmean(x):
        return np.nan if np.all(np.isnan(x)) else np.nanmean(x)

    macroPrecision = nanmean(precision)
    macroRecall    = nanmean(recall)
    macroF1        = nanmean(f1)

    # Micro averages
    totalTP = float(TP.sum())
    totalFP = float(FP.sum())
    totalFN = float(FN.sum())

    microPrecision = totalTP / (totalTP + totalFP) if (totalTP + totalFP) > 0 else np.nan
    microRecall    = totalTP / (totalTP + totalFN) if (totalTP + totalFN) > 0 else np.nan
    microF1        = (2 * microPrecision * microRecall / (microPrecision + microRecall)
                      if (np.isfinite(microPrecision) and np.isfinite(microRecall) and (microPrecision + microRecall) > 0)
                      else np.nan)

    mAP = macroPrecision

    # Overall accuracy
    accuracy = float((y_true == y_pred).sum()) / float(y_true.size)

    # Cohen's Kappa (from confusion matrix)
    N = conf_mat.sum()
    po = TP.sum() / N if N > 0 else np.nan
    row_marginals = conf_mat.sum(axis=1).astype(float)
    col_marginals = conf_mat.sum(axis=0).astype(float)
    pe = (row_marginals @ col_marginals) / (N * N) if N > 0 else np.nan
    kappa = ((po - pe) / (1 - pe)) if (np.isfinite(po) and np.isfinite(pe) and (1 - pe) > 0) else np.nan


    c = TP.sum()
    s = N
    p_k = row_marginals           # actual per class (rows)
    t_k = col_marginals           # predicted per class (cols)
    sum_pk_tk = float((p_k * t_k).sum())
    denom_left  = (s**2 - (p_k**2).sum())
    denom_right = (s**2 - (t_k**2).sum())
    denom = np.sqrt(denom_left * denom_right)
    mcc = ((c * s - sum_pk_tk) / denom) if denom > 0 else np.nan

    return {
        'classes': classes,
        'conf_mat': conf_mat,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macroPrecision': macroPrecision,
        'macroRecall': macroRecall,
        'macroF1': macroF1,
        'microPrecision': microPrecision,
        'microRecall': microRecall,
        'microF1': microF1,
        'mAP': mAP,
        'accuracy': accuracy,
        'kappa': kappa,
        'mcc': mcc
    }

if __name__ == "__main__":
    gt_maps,pred_maps=example_data_load()

    res = pixelwise_metrics(gt_maps=gt_maps, pred_maps=pred_maps, ignore_class=0)
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(k, ":\n", v)
        else:
            print(k, ":", v)
