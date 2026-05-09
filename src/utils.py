import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_norm(channels: int) -> torch.nn.GroupNorm:
    if channels % 8 == 0:
        return torch.nn.GroupNorm(8, channels)
    if channels % 4 == 0:
        return torch.nn.GroupNorm(4, channels)
    if channels % 2 == 0:
        return torch.nn.GroupNorm(2, channels)
    return torch.nn.GroupNorm(1, channels)


def binary_bce_loss_with_ignore(logits, target, ignore_index=-1):
    if target.ndim == 3:
        target = target.unsqueeze(1)

    target = target.to(device=logits.device, dtype=logits.dtype)

    valid_mask = target != ignore_index

    clean_target = torch.where(
        valid_mask,
        target,
        torch.zeros_like(target),
    )

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        clean_target,
        reduction="none",
    )

    loss = loss * valid_mask.float()
    return loss.sum() / valid_mask.float().sum().clamp_min(1.0)


@torch.no_grad()
def binary_seg_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    ignore_index: int = -1,
    eps: float = 1e-6,
):
    if target.ndim == 3:
        target = target.unsqueeze(1)

    target = target.to(logits.device)
    valid = target != ignore_index

    prob = torch.sigmoid(logits)
    pred = prob > threshold

    target_bool = target > 0.5

    pred = pred & valid
    target_bool = target_bool & valid

    tp = (pred & target_bool).sum().float()
    fp = (pred & ~target_bool & valid).sum().float()
    fn = (~pred & target_bool & valid).sum().float()
    tn = (~pred & ~target_bool & valid).sum().float()

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    valid_count = valid.sum().float().clamp_min(1.0)
    target_pos_ratio = target_bool.sum().float() / valid_count
    pred_pos_ratio = pred.sum().float() / valid_count

    if valid.any():
        valid_prob = prob[valid]
        prob_sum = valid_prob.sum()
        prob_sq_sum = (valid_prob * valid_prob).sum()
        prob_mean = prob_sum / valid_count
        prob_var = (prob_sq_sum / valid_count - prob_mean * prob_mean).clamp_min(0.0)
        prob_std = prob_var.sqrt()
    else:
        prob_sum = logits.new_tensor(0.0)
        prob_sq_sum = logits.new_tensor(0.0)
        prob_mean = logits.new_tensor(0.0)
        prob_std = logits.new_tensor(0.0)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "target_pos_ratio": target_pos_ratio.item(),
        "pred_pos_ratio": pred_pos_ratio.item(),
        "prob_mean": prob_mean.item(),
        "prob_std": prob_std.item(),
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
        "valid_count": valid_count.item(),
        "target_pos_count": target_bool.sum().float().item(),
        "pred_pos_count": pred.sum().float().item(),
        "prob_sum": prob_sum.item(),
        "prob_sq_sum": prob_sq_sum.item(),
    }
