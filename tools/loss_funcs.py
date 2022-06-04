import torch
import torch.nn.functional as F


def Cross_entropy_loss(prediction, label):
    r"""
    RCF使用的类别不平均损失函数
    """
    mask = label.clone()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = F.binary_cross_entropy(prediction, label, weight=mask, reduce=False)
    return torch.sum(cost)