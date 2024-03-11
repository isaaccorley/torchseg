import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._functional import soft_dice_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .reductions import LossReduction


class DiceLoss(nn.Module):
    def __init__(
            self,
            classes: Optional[list[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            mask_to_one_hot: bool = False,
            power: float = 1.0,
            reduction: str = 'mean',
            smooth: float = 0.0,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7,
    ):
        """
        DA RISCRIVERE
        Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            - mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            - classes:  List of classes that contribute in loss computation.
                By default, all channels are included.
            - log_loss: If True, loss computed as `- log(dice_coeff)`,
                otherwise `1 - dice_coeff`
            - from_logits: If True, assumes input is raw logits
            - ignore_index: Label that indicates ignored pixels  not contributing to the loss
            - smooth: Smoothness constant for dice coefficient added to the numerator to avoid zero
            - eps: A small epsilon added to the denominator  for numerical stability to avoid nan
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (B, C, H, W),
             - **y_true** - torch.Tensor of shape (B, C, H, W) or (B, 1, H, W),
        where C is the number of classes.

        Reference
            https://docs.monai.io/en/stable/_modules/monai/losses/dice.html#DiceLoss
        """
        if reduction not in LossReduction.available_reductions():
            raise ValueError(f'Unsupported reduction: {reduction}, '
                             f'available options are {LossReduction.available_reductions()}.')
        super().__init__()

        self.classes = classes
        self.from_logits = from_logits
        self.mask_to_one_hot = mask_to_one_hot
        self.power = power
        self.reduction = reduction
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]
        spatial_dims: list[int] = torch.arange(2, len(y_pred.shape)).tolist()

        if self.classes is not None:
            if num_classes == 1:
                warnings.warn("Single channel prediction, masking classes is not supported for Binary Segmentation")
            else:
                self.classes = to_tensor(self.classes, dtype = torch.long)

        if self.from_logits:
            if num_classes == 1:
                y_pred = F.logsigmoid(y_pred).exp()
            else:
                y_pred = F.log_softmax(y_pred, dim = 1).exp()

        if self.mask_to_one_hot:
            if num_classes == 1:
                warnings.warn("Single channel prediction, 'mask_to_one_hot = True' ignored.")
            else:
                # maybe there is a better way to handle this?
                permute_dims = tuple(dim - 1 for dim in spatial_dims)
                y_true = F.one_hot(y_true, num_classes).squeeze(dim = 1) # N, 1, H, W, ... ---> N, H, W, ..., C
                y_true = y_true.permute(0, -1, *permute_dims) # N, 1, H, W, ..., C ---> N, C, H, W, ...

        if y_true.shape != y_pred.shape:
            raise AssertionError(f"Ground truth has different shape ({y_true.shape})"
                                 f" from predicted mask ({y_pred.shape})")

        # Only reduce spatial dimensions
        scores = soft_dice_score(y_pred,
                                 y_true.type_as(y_pred),
                                 power = self.power,
                                 smooth = self.smooth,
                                 eps = self.eps,
                                 dims = spatial_dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        """
        Personal Opinion: maybe this can be skipped. By summing over batch
        and spatial dimensions I think the idea is: I constantly have channels
        with pixels = 0 which should not be included in the loss computation,
        but if this happens there is something wrong with the masks.
        While by just summing over spatial dimensions it means that an empty channel
        should not be used to compute the DiceLoss, but it may simply mean
        that the specific class is not present in this specific mask.
        """
        # to delete?
        #dims = tuple(d for d in range(len(y_true.shape)) if d != 1)
        #mask = y_true.sum(dims) > 0
        #loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[:, self.classes, :]

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        elif self.reduction == LossReduction.NONE:
            broadcast_shape = list(loss.shape[0:2]) + [1] * (len(y_true.shape) - 2)
            loss = loss.view(broadcast_shape)

        return loss




class DiceLossOld(nn.Module):
    def __init__(
        self,
        mode: str,
        classes: Optional[list[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        power: float= 1.0,
        reduction: str = 'mean',
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """
        Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            - mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            - classes:  List of classes that contribute in loss computation.
                By default, all channels are included.
            - log_loss: If True, loss computed as `- log(dice_coeff)`,
                otherwise `1 - dice_coeff`
            - from_logits: If True, assumes input is raw logits
            - smooth: Smoothness constant for dice coefficient
                added to the numerator to avoid zero
            - ignore_index: Label that indicates ignored pixels
                (does not contribute to loss)
            - eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        if mode not in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}:
            raise ValueError(f'Unsupported mode: {mode}, '
                             f'available options are ["binary", "multiclass", "multilabel"].')
        if reduction not in {MEAN_REDUCTION, SUM_REDUCTION, NONE_REDUCTION}:
            raise ValueError(f'Unsupported reduction: {reduction}, '
                             f'available options are ["mean", "sum", "none"].')
        super().__init__()
        self.mode = mode
        if classes is not None:
            assert (
                mode != BINARY_MODE
            ), "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype = torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.power = power
        self.reduction = reduction
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        height, width = y_pred.size(2), y_pred.size(3)
        dims = (0, 2)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable
            # result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred,
                                 y_true.type_as(y_pred),
                                 power = self.power,
                                 smooth = self.smooth,
                                 eps = self.eps,
                                 dims = dims)

        print("shape of scores: ", scores.shape)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        # ---> to check
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        if self.reduction == MEAN_REDUCTION:
            loss = torch.mean(loss)
        elif self.reduction == SUM_REDUCTION:
            loss = torch.sum(loss)
        elif self.reduction == NONE_REDUCTION:
            loss = loss.view(bs, num_classes, height, width)

        return loss