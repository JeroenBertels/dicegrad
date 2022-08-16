import tensorflow as tf
from functools import partial
from metrics_missing_labels import get_metric
from deepvoxnet2.backwards_compatibility.losses import *

def missing_labels_dice_loss(y_true, y_pred, threshold=None, reduce_along_batch = False, calibrate = False, feature_weights = None, **kwargs):
    return 1 - get_metric("missing_labels_dice_metric", threshold=threshold, reduce_along_batch=reduce_along_batch, calibrate = calibrate,feature_weights = feature_weights, **kwargs)(y_true, y_pred)


def leaf_dice_loss(y_true, y_pred, threshold=None, reduce_along_batch = False,lossx5= False, **kwargs):
    return 1 - get_metric("leaf_dice_metric", threshold=threshold, reduce_along_batch=reduce_along_batch, lossx5 = lossx5,**kwargs)(y_true, y_pred)


def marginal_dice_loss(y_true, y_pred, threshold=None, reduce_along_batch = False, **kwargs):
    return 1 - get_metric("marginal_dice_metric", threshold=threshold, reduce_along_batch=reduce_along_batch,**kwargs)(y_true, y_pred)


def get_loss(loss_name, reduction_mode="mean", custom_loss_name=None, **kwargs):
    if loss_name == "leaf_dice_loss":
        loss = leaf_dice_loss

    elif loss_name == "marginal_dice_loss":
        loss = marginal_dice_loss

    else:
        raise NotImplementedError("The requested loss is not implemented.")

    loss = partial(loss, reduction_mode=reduction_mode, **kwargs)
    loss.__name__ = loss_name if custom_loss_name is None else custom_loss_name
    return loss