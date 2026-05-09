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
    target = target.float()

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
 
