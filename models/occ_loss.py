from custom_types import *


def occupancy_bce(predict: T, winding_gt: T, ignore: Optional[T] = None, *args) -> T:
    if winding_gt.dtype is not torch.bool:
        winding_gt = winding_gt.gt(0)
    labels = winding_gt.flatten().float()
    predict = predict.flatten()
    if ignore is not None:
        ignore = (~ignore).flatten().float()
    loss = nnf.binary_cross_entropy_with_logits(predict, labels, weight=ignore)
    return loss


def reg_z_loss(z: T) -> T:
    norms = z.norm(2, 1)
    loss = norms.mean()
    return loss
