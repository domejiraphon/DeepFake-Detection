import torch
import torch.nn as nn
import torch.nn.functional as F

softmax = nn.Softmax(dim=-1)


def cross_entropy(logits, target, num_classes):
    one_hot = F.one_hot(target, num_classes)
    prob = softmax(logits)

    summ = -torch.sum(one_hot * torch.log(prob + 1e-12), dim=-1)
    loss = torch.mean(summ)
    return loss


def l2_loss(x, y):
    return torch.mean((x - y) ** 2)
