# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
) -> torch.Tensor:
    """
    Function that computes the focal loss for binary classification.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape representing 
            the predictions (logits) for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. 
            Stores the binary classification label for each element.
        alpha (float, optional): Weighting factor in range (0,1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float, optional): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.0.
        reduction (str, optional): 'none' | 'mean' | 'sum'. 
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed. Default: 'mean'.

    Returns:
        torch.Tensor: Loss tensor with the specified reduction applied.
    """
    # Calculate probabilities from logits
    p = torch.sigmoid(inputs)

    # Calculate binary cross entropy with logits for numerical stability
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Calculate p_t
    # p_t = p if target == 1 else 1 - p
    p_t = p * targets + (1 - p) * (1 - targets)

    # Calculate the modulating factor (1 - p_t)^gamma
    loss = bce_loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class SigmoidFocalLoss(nn.Module):
    """SigmoidFocalLoss, addresses class imbalance by down-weighting well-classified examples.

    Args:
        - alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
        - gamma (float): Exponent of the modulating factor to focus on hard examples.
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Shape:
        - Inputs: (N, *) where * means, any number of additional dimensions
        - Targets: (N, *), same shape as the inputs
        - Output: scalar if `reduction` is 'mean' or 'sum', otherwise (N, *).

    Examples::

        >>> loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(3, requires_grad=True)
        >>> targets = torch.empty(3).random_(2)
        >>> output = loss(logits, targets)
        >>> output.backward()
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )