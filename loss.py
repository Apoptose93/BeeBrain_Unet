from typing import Sequence, Optional

import torch


class CombinedLoss(torch.nn.Module):
    """Defines a loss function as a weighted sum of combinable loss criteria.
    Args:
        criteria: List of loss criterion modules that should be combined.
        weight: Weight assigned to the individual loss criteria (in the same
            order as ``criteria``).
        device: The device on which the loss should be computed. This needs
            to be set to the device that the loss arguments are allocated on.
    """

    def __init__(
            self,
            criteria: Sequence[torch.nn.Module],
            weight: Optional[Sequence[float]] = None,
            device: torch.device = None
    ):
        super().__init__()
        self.criteria = torch.nn.ModuleList(criteria)
        self.device = device
        if weight is None:
            weight = torch.ones(len(criteria))
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            assert weight.shape == (len(criteria),)
        self.register_buffer('weight', weight.to(self.device))

    def forward(self, *args):
        loss = torch.tensor(0., device=self.device)
        for crit, weight in zip(self.criteria, self.weight):
            crit = crit.to(self.device)
            loss += weight * crit(*args)
        return loss


def _channelwise_sum(x: torch.Tensor):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])  # = (0, 2, 3, ...)
    return x.sum(dim=reduce_dims)


def dice_loss(probs,
              target,
              weight=1.,
              eps=0.0001,
              ignore_index=False,
              no_reduction=False,
              ):
    # Probs need to be softmax probabilities, not raw network outputs
    tsh, psh = target.shape, probs.shape

    if tsh == psh:  # Already one-hot
        onehot_target = target.to(probs.dtype)
    elif tsh[0] == psh[0] and tsh[1:] == psh[2:]:  # Assume dense target storage, convert to one-hot
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(
            f'Target shape {target.shape} is not compatible with output shape {probs.shape}.'
        )

    intersection = probs * onehot_target  # (N, C, ...)
    numerator = 2 * _channelwise_sum(intersection)  # (C,)
    denominator = probs + onehot_target  # (N, C, ...)
    denominator = _channelwise_sum(denominator) + eps  # (C,)
    loss_per_channel = 1 - (numerator / denominator)  # (C,)
    weighted_loss_per_channel = weight * loss_per_channel  # (C,)
    if ignore_index:
        weighted_loss_per_channel = weighted_loss_per_channel[1:]
        if no_reduction:
            return weighted_loss_per_channel
        else:
            return weighted_loss_per_channel.mean()
    else:
        if no_reduction:
            return weighted_loss_per_channel
        else:
            return weighted_loss_per_channel.mean()


class DiceLoss(torch.nn.Module):
    """Generalized Dice Loss, as described in https://arxiv.org/abs/1707.03237.
    Works for n-dimensional data. Assuming that the ``output`` tensor to be
    compared to the ``target`` has the shape (N, C, D, H, W), the ``target``
    can either have the same shape (N, C, D, H, W) (one-hot encoded) or
    (N, D, H, W) (with dense class indices, as in
    ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, the
    ``target`` is automatically internally converted to a one-hot tensor
    for loss calculation.
    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            Use this if your model outputs 2 channels or more.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
        apply_sigmoid: If ``True``, a sigmoid operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply sigmoid as the last layer.
            Use this if your model ouputs only 1 channel.
            If ``False``, ``output`` is assumed to already contain sigmoid
            probabilities.
        weight: Weight tensor for class-wise loss rescaling.
            Has to be of shape (C,). If ``None``, classes are weighted equally.
    """

    def __init__(self,
                 apply_activation_function='softmax',
                 weight=torch.tensor(1.),
                 ignore_index=False,
                 no_reduction=False,
                 ):
        super().__init__()
        if apply_activation_function == 'softmax':
            self.layer = torch.nn.Softmax(dim=1)
        elif apply_activation_function == 'sigmoid':
            self.layer = torch.nn.Sigmoid()

        self.dice = dice_loss
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.no_reduction = no_reduction

    def forward(self, output, target):
        probs = self.layer(output)
        return self.dice(probs,
                         target,
                         weight=self.weight,
                         ignore_index=self.ignore_index,
                         no_reduction=self.no_reduction
                         )


