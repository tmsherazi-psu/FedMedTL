import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, num_classes=8):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes

    def forward(self, input, target, smooth=1):
        if self.use_sigmoid:
            input = torch.sigmoid(input)
        # Assume input is [N, C, H, W] and target is [N, H, W]
        input = F.softmax(input, dim=1)  # Softmax over channels
        input = input.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).view(-1)
        intersection = (input * target).sum()
        dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
        return 1 - dice


# --- Binary Cross Entropy Loss (Modified for Multi-Class) ---
class BCELoss(nn.Module):
    def __init__(self, use_sigmoid=True, num_classes=8):
        super(BCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes

    def forward(self, input, target):
        if self.use_sigmoid:
            input = torch.sigmoid(input)
        else:
            input = F.softmax(input, dim=1)

        # Convert target to one-hot
        target = F.one_hot(target, num_classes=self.num_classes).float()
        return F.binary_cross_entropy(input.view(-1), target.view(-1))


# --- Dice + BCE Loss (Multi-Class Support) ---
class DiceBCELoss(nn.Module):
    def __init__(self, use_sigmoid=True, num_classes=8, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, input, target):
        if self.use_sigmoid:
            input = torch.sigmoid(input)
        else:
            input = F.softmax(input, dim=1)

        input = input.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).float().view(-1)

        # Dice Loss
        intersection = (input * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                input.sum() + target.sum() + self.smooth
        )

        # BCE Loss
        bce = F.binary_cross_entropy(input, target)

        return bce + dice_loss


# --- Focal Loss (Multi-Class) ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7, num_classes=8):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, input, target):
        input = F.log_softmax(input, dim=1)
        input = input.clamp(self.eps, 1 - self.eps)
        target = F.one_hot(target, num_classes=self.num_classes).float()

        ce = -target * input
        weight = (1 - torch.exp(-input)) ** self.gamma
        focal = weight * ce
        loss = focal.sum(dim=1).mean()

        return loss


# --- Dice + Focal Loss (Multi-Class) ---
class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=2, smooth=1, eps=1e-7, num_classes=8):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, input, target):
        input = F.log_softmax(input, dim=1)
        input = input.clamp(self.eps, 1 - self.eps)
        target_oh = F.one_hot(target, num_classes=self.num_classes).float()

        # Focal Loss
        ce = -target_oh * input
        weight = (1 - torch.exp(-input)) ** self.gamma
        focal = weight * ce
        focal_loss = focal.sum(dim=1).mean()

        # Dice Loss
        input_sig = torch.sigmoid(input).view(-1)
        target_flat = target_oh.view(-1)
        intersection = (input_sig * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                input_sig.sum() + target_flat.sum() + self.smooth
        )

        return focal_loss + dice_loss


# --- Label Smoothing Cross-Entropy Loss (Updated for Multi-Class) ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements label smoothing regularization as described in Eq.11 and Eq.13 of your paper.
    Converts hard labels to soft ones to improve generalization.
    """
    def __init__(self, alpha=0.1, num_classes=8):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        targets = targets.view(-1)

        # Create smoothed labels
        device = inputs.device
        true_dist = torch.zeros(batch_size, self.num_classes, device=device)
        true_dist.fill_(self.alpha / (self.num_classes - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), 1 - self.alpha)

        # Compute cross entropy with smoothed labels
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(log_probs * true_dist).sum(dim=1).mean()

        return loss


# --- Knowledge Distillation Loss (KL Divergence) ---
class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss: Transfers knowledge from teacher to student via KL divergence.
    Used in FedMedTL for personalized FL.
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight between hard loss and soft loss

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss