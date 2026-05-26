import torch
import torch.nn as nn
import torch.nn.functional as F


"""
PyTorch losses and metrics for semantic segmentation.

This file includes:
- DiceLoss
- FocalLoss
- CrossEntropyLoss wrapper
- CustomLoss = DiceLoss + FocalLoss
- Dice metric
- IoU metric

Expected tensor shapes
----------------------
y_pred:
    [B, C, H, W]
    Raw logits before softmax.

y_true:
    [B, H, W]
    Integer class labels.
"""


# =============================================================================
# Loss functions
# =============================================================================

class DiceLoss(nn.Module):
    """
    Multi-class Dice loss.

    Dice = 2 * intersection / (prediction + target)
    Loss = 1 - Dice
    """

    def __init__(self, smooth=1.0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):
        num_classes = y_pred.shape[1]

        # Convert logits to class probabilities.
        y_pred = F.softmax(y_pred, dim=1)

        # Convert integer labels to one-hot masks.
        y_true_one_hot = F.one_hot(
            y_true.long(),
            num_classes=num_classes,
        )
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()

        # Sum over batch and spatial dimensions.
        dims = (0, 2, 3)

        intersection = torch.sum(y_pred * y_true_one_hot, dim=dims)
        cardinality = torch.sum(y_pred + y_true_one_hot, dim=dims)

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            cardinality + self.smooth + self.eps
        )

        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Focal loss down-weights easy pixels and focuses training on hard pixels.

    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Log-probability for each class.
        log_prob = F.log_softmax(y_pred, dim=1)

        # Select the log-probability of the true class for each pixel.
        y_true_expanded = y_true.unsqueeze(1).long()
        log_prob_true = log_prob.gather(
            dim=1,
            index=y_true_expanded,
        ).squeeze(1)

        prob_true = torch.exp(log_prob_true)

        focal_weight = (1.0 - prob_true) ** self.gamma
        loss = -focal_weight * log_prob_true

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(
                    self.alpha,
                    device=y_pred.device,
                    dtype=y_pred.dtype,
                )
            else:
                alpha = self.alpha

            alpha_true = alpha.gather(
                dim=0,
                index=y_true.reshape(-1).long(),
            ).reshape_as(y_true)

            loss = alpha_true * loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class CrossEntropyLoss(nn.Module):
    """
    Wrapper for PyTorch cross-entropy loss.
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
        )

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true.long())


class CustomLoss(nn.Module):
    """
    Combined segmentation loss.

    Total loss = dice_weight * DiceLoss + focal_weight * FocalLoss
    """

    def __init__(self, dice_weight=1.0, focal_weight=1.0, gamma=3.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, y_pred, y_true):
        dice_value = self.dice_loss(y_pred, y_true)
        focal_value = self.focal_loss(y_pred, y_true)

        return (
            self.dice_weight * dice_value
            + self.focal_weight * focal_value
        )


# =============================================================================
# Evaluation metrics
# =============================================================================

def dice(pred, target, eps=1.0):
    """
    Compute mean Dice score over foreground classes.

    Parameters
    ----------
    pred : torch.Tensor
        Model logits with shape [B, C, H, W].

    target : torch.Tensor
        Ground-truth labels with shape [B, H, W].

    eps : float
        Small constant for numerical stability.

    Notes
    -----
    Class 0 is treated as background and ignored.
    """
    num_classes = pred.shape[1]
    pred_label = torch.argmax(pred, dim=1)

    scores = []

    for class_id in range(1, num_classes):
        pred_mask = pred_label == class_id
        target_mask = target == class_id

        intersection = torch.sum(pred_mask & target_mask).float()
        denominator = torch.sum(pred_mask).float() + torch.sum(target_mask).float()

        score = (2.0 * intersection + eps) / (denominator + eps)
        scores.append(score)

    if not scores:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(scores).mean()


def iou(pred, target, eps=1.0):
    """
    Compute mean IoU over foreground classes.

    Parameters
    ----------
    pred : torch.Tensor
        Model logits with shape [B, C, H, W].

    target : torch.Tensor
        Ground-truth labels with shape [B, H, W].

    eps : float
        Small constant for numerical stability.

    Notes
    -----
    Class 0 is treated as background and ignored.
    """
    num_classes = pred.shape[1]
    pred_label = torch.argmax(pred, dim=1)

    scores = []

    for class_id in range(1, num_classes):
        pred_mask = pred_label == class_id
        target_mask = target == class_id

        intersection = torch.sum(pred_mask & target_mask).float()
        union = torch.sum(pred_mask | target_mask).float()

        if union > 0:
            score = (intersection + eps) / (union + eps)
            scores.append(score)

    if not scores:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(scores).mean()