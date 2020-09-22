import torch

from detectron2.layers import nonzero_tuple

__all__ = ["atss_subsample"]


def atss_subsample(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):

    # positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    positive = nonzero_tuple(labels != bg_label)[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx