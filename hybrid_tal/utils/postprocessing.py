"""
Post-processing utilities for Temporal Action Localization.
Includes Soft-NMS and mAP evaluation following the paper's methodology.
"""

import numpy as np
from collections import defaultdict


def soft_nms(segments, scores, labels, sigma=0.5, threshold=0.001, max_seg_num=200):
    """
    Soft-NMS for temporal action proposals.
    
    Instead of hard suppression, Soft-NMS decays the scores of overlapping
    proposals using a Gaussian penalty (Bodla et al., 2017).
    
    Args:
        segments: (N, 2) array of [start, end] timestamps
        scores: (N,) array of confidence scores
        labels: (N,) array of class labels
        sigma: Gaussian decay parameter
        threshold: Score threshold for keeping proposals
        max_seg_num: Maximum number of proposals to keep
    
    Returns:
        kept_segments, kept_scores, kept_labels
    """
    if len(segments) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int)
    
    segments = np.array(segments, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=int)
    
    N = len(segments)
    indices = np.arange(N)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    segments = segments[order]
    scores = scores[order]
    labels = labels[order]
    
    kept_segments = []
    kept_scores = []
    kept_labels = []
    
    while len(segments) > 0 and len(kept_segments) < max_seg_num:
        # Take the highest scoring proposal
        idx = 0
        kept_segments.append(segments[idx])
        kept_scores.append(scores[idx])
        kept_labels.append(labels[idx])
        
        if len(segments) == 1:
            break
        
        # Compute temporal IoU with remaining proposals
        remaining_segs = segments[1:]
        remaining_scores = scores[1:]
        remaining_labels = labels[1:]
        
        inter_start = np.maximum(segments[idx, 0], remaining_segs[:, 0])
        inter_end = np.minimum(segments[idx, 1], remaining_segs[:, 1])
        inter = np.maximum(0, inter_end - inter_start)
        
        dur_a = segments[idx, 1] - segments[idx, 0]
        dur_b = remaining_segs[:, 1] - remaining_segs[:, 0]
        union = dur_a + dur_b - inter
        
        iou = inter / np.maximum(union, 1e-6)
        
        # Gaussian decay
        decay = np.exp(-(iou ** 2) / sigma)
        remaining_scores = remaining_scores * decay
        
        # Filter by threshold
        keep = remaining_scores > threshold
        segments = remaining_segs[keep]
        scores = remaining_scores[keep]
        labels = remaining_labels[keep]
        
        # Re-sort
        order = scores.argsort()[::-1]
        segments = segments[order]
        scores = scores[order]
        labels = labels[order]
    
    if len(kept_segments) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int)
    
    return (
        np.array(kept_segments),
        np.array(kept_scores),
        np.array(kept_labels),
    )


def compute_temporal_iou(pred_seg, gt_seg):
    """Compute temporal IoU between two segments."""
    inter_start = max(pred_seg[0], gt_seg[0])
    inter_end = min(pred_seg[1], gt_seg[1])
    inter = max(0, inter_end - inter_start)
    
    union = (pred_seg[1] - pred_seg[0]) + (gt_seg[1] - gt_seg[0]) - inter
    return inter / max(union, 1e-6)


def compute_average_precision(precision, recall):
    """Compute AP using the 11-point interpolation method."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Integrate area under precision-recall curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_mAP(predictions, ground_truth, tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Evaluate mean Average Precision for temporal action localization.
    
    Args:
        predictions: dict of {video_id: list of (start, end, label, score)}
        ground_truth: dict of {video_id: list of (start, end, label)}
        tiou_thresholds: List of tIoU thresholds
    
    Returns:
        mAP_dict: dict with mAP at each threshold and average
    """
    # Collect all unique classes
    all_classes = set()
    for video_id, gts in ground_truth.items():
        for gt in gts:
            all_classes.add(gt[2])
    
    results = {}
    
    for tiou in tiou_thresholds:
        ap_per_class = []
        
        for cls in sorted(all_classes):
            # Collect predictions and GT for this class
            cls_predictions = []
            cls_gt = defaultdict(list)
            n_gt = 0
            
            for video_id, gts in ground_truth.items():
                for gt in gts:
                    if gt[2] == cls:
                        cls_gt[video_id].append(gt)
                        n_gt += 1
            
            for video_id, preds in predictions.items():
                for pred in preds:
                    if pred[2] == cls:
                        cls_predictions.append((video_id, pred[0], pred[1], pred[3]))
            
            if n_gt == 0:
                continue
            
            # Sort predictions by score (descending)
            cls_predictions.sort(key=lambda x: -x[3])
            
            tp = np.zeros(len(cls_predictions))
            fp = np.zeros(len(cls_predictions))
            
            # Track which GTs have been matched
            matched = defaultdict(lambda: np.zeros(100, dtype=bool))
            
            for idx, (vid, start, end, score) in enumerate(cls_predictions):
                if vid not in cls_gt:
                    fp[idx] = 1
                    continue
                
                gts = cls_gt[vid]
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    iou = compute_temporal_iou([start, end], [gt[0], gt[1]])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= tiou and not matched[vid][best_gt_idx]:
                    tp[idx] = 1
                    matched[vid][best_gt_idx] = True
                else:
                    fp[idx] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / n_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            ap = compute_average_precision(precision, recall)
            ap_per_class.append(ap)
        
        mAP = np.mean(ap_per_class) if ap_per_class else 0.0
        results[f'mAP@{tiou}'] = mAP * 100  # Percentage
    
    # Average mAP
    avg_mAP = np.mean([results[k] for k in results])
    results['avg_mAP'] = avg_mAP
    
    return results


def decode_predictions(
    cls_logits,
    reg_preds,
    masks,
    feat_stride,
    num_frames,
    scale_factor,
    num_classes,
    pre_nms_thresh=0.001,
    pre_nms_topk=2000,
    nms_sigma=0.5,
    nms_threshold=0.001,
    max_seg_num=200,
    duration=None,
    fps=30,
):
    """
    Decode model outputs into temporal action proposals.
    
    Args:
        cls_logits: List of (1, num_classes, T_l) for each level
        reg_preds: List of (1, 2, T_l) for each level
        masks: List of (1, 1, T_l) for each level
        ...
    
    Returns:
        segments: (N, 2) array of [start, end] in seconds
        scores: (N,) array of confidence scores
        labels: (N,) array of class labels
    """
    all_segments = []
    all_scores = []
    all_labels = []
    
    for level in range(len(cls_logits)):
        cls = cls_logits[level][0].sigmoid()  # (num_classes, T_l)
        reg = reg_preds[level][0]  # (2, T_l)
        mask = masks[level][0, 0]  # (T_l,)
        
        stride = feat_stride * num_frames * (scale_factor ** level)
        T_l = cls.shape[1]
        
        # Point coordinates
        points = torch.arange(T_l, device=cls.device).float() * stride + stride / 2.0
        
        for t in range(T_l):
            if mask[t] < 0.5:
                continue
            
            for c in range(num_classes):
                score = cls[c, t].item()
                if score < pre_nms_thresh:
                    continue
                
                left_dist = reg[0, t].item() * stride
                right_dist = reg[1, t].item() * stride
                
                start = (points[t].item() - left_dist) / fps
                end = (points[t].item() + right_dist) / fps
                
                # Clip to valid range
                start = max(0, start)
                if duration is not None:
                    end = min(duration, end)
                
                if end > start:
                    all_segments.append([start, end])
                    all_scores.append(score)
                    all_labels.append(c)
    
    if len(all_segments) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int)
    
    # Convert to arrays
    all_segments = np.array(all_segments)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Pre-NMS top-K
    if len(all_scores) > pre_nms_topk:
        topk_idx = all_scores.argsort()[::-1][:pre_nms_topk]
        all_segments = all_segments[topk_idx]
        all_scores = all_scores[topk_idx]
        all_labels = all_labels[topk_idx]
    
    # Per-class Soft-NMS
    final_segments = []
    final_scores = []
    final_labels = []
    
    for c in range(num_classes):
        cls_mask = all_labels == c
        if cls_mask.sum() == 0:
            continue
        
        cls_segs = all_segments[cls_mask]
        cls_scores = all_scores[cls_mask]
        cls_labels = all_labels[cls_mask]
        
        kept_segs, kept_scores, kept_labels = soft_nms(
            cls_segs, cls_scores, cls_labels,
            sigma=nms_sigma,
            threshold=nms_threshold,
            max_seg_num=max_seg_num,
        )
        
        final_segments.append(kept_segs)
        final_scores.append(kept_scores)
        final_labels.append(kept_labels)
    
    if len(final_segments) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int)
    
    return (
        np.concatenate(final_segments),
        np.concatenate(final_scores),
        np.concatenate(final_labels),
    )
