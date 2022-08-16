import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from collections import Iterable
from pymirc.metrics.tf_metrics import generalized_dice_coeff
from deepvoxnet2.backwards_compatibility.metrics import *
from deepvoxnet2.keras.metrics import _metric

def leaf_dice_metric(y_true, y_pred, eps=tf.keras.backend.epsilon(), reduce_along_batch=False, **kwargs):
    missing_labels = tf.cast(tf.reduce_sum(y_true[:,:,:,:,1:],axis=(1,2,3),keepdims = True)==0,tf.float32)

    y_pred_adapted = (1-missing_labels) * y_pred[:,:,:,:,1:]
    y_true_adapted = (1 - missing_labels) * y_true[:, :, :, :, 1:]

    intersection = tf.math.reduce_sum(y_true_adapted * y_pred_adapted, axis=(1,2,3), keepdims=True)
    denom = tf.math.reduce_sum(y_true_adapted, axis=(1,2,3), keepdims=True) + tf.math.reduce_sum(y_pred_adapted, axis=(1,2,3),keepdims=True)

    if reduce_along_batch:
        intersection = tf.math.reduce_sum(intersection,axis=0,keepdims=True)
        denom = tf.math.reduce_sum(denom,axis=0,keepdims=True)

    DSC = (1/tf.reduce_sum(1-missing_labels, axis=4, keepdims=True)) *tf.math.reduce_sum((2 * intersection) / (denom + eps),axis = 4,keepdims = True)
    DSC = (1-missing_labels)*DSC

    return DSC


def marginal_dice_metric(y_true, y_pred, **kwargs):
    missing_labels = tf.cast(tf.reduce_sum(y_true[:,:,:,:,1:],axis=(1,2,3),keepdims = True)==0,tf.float32)
    y_pred_BG = (y_pred[:,:,:,:,0:1] + tf.reduce_sum(missing_labels * y_pred[:,:,:,:,1:],axis = 4,keepdims = True)) # add probabilities missing labels to background
    y_pred_label = (1-missing_labels) * y_pred[:,:,:,:,1:] # set missing labels to zero
    y_pred_adapted= tf.concat([y_pred_BG, y_pred_label], axis=4)

    DSC =  generalized_dice_coeff(y_true, y_pred_adapted, keepdims=True, reduce_along_features=False)
    DSC = tf.concat([DSC[:,:,:,:,0:1],(1-missing_labels)*DSC[:,:,:,:,1:]],axis=4)
    return DSC

def get_metric(
        metric_name,
        batch_dim_as_spatial_dim=False,
        feature_dim_as_spatial_dim=False,
        threshold=None,
        argmax=False,
        map_batch=False,
        map_features=False,
        reduction_mode=None,
        percentile=None,
        reduction_axes=(0, 1, 2, 3, 4),
        custom_metric_name=None,
        threshold_mode="greater",
        weights = None,
        **kwargs):

    if metric_name == "leaf_dice_metric":
        metric = leaf_dice_metric

    elif metric_name == "marginal_dice_metric":
        metric = marginal_dice_metric

    else:
        raise NotImplementedError("The requested metric is not implemented.")

    metric = partial(
        _metric,
        metric_name=metric_name,
        metric=metric,
        batch_dim_as_spatial_dim=batch_dim_as_spatial_dim,
        feature_dim_as_spatial_dim=feature_dim_as_spatial_dim,
        threshold=threshold,
        argmax=argmax,
        map_batch=map_batch,
        map_features=map_features,
        reduction_mode=reduction_mode,
        percentile=percentile,
        reduction_axes=reduction_axes,
        threshold_mode=threshold_mode,
        weights = weights,
        **kwargs
    )
    if custom_metric_name is not None:
        metric.__name__ = custom_metric_name

    elif reduction_mode is None:
        metric.__name__ = metric_name

    elif reduction_mode == "mean":
        metric.__name__ = "mean_" + metric_name

    elif reduction_mode == "sum":
        metric.__name__ = "sum_" + metric_name

    elif reduction_mode == "median" or (reduction_mode == "percentile" and (percentile is None or percentile == 50)):
        metric.__name__ = "median_" + metric_name

    else:
        assert reduction_mode == "percentile" and percentile is not None, "Unknown reduction_mode/percentile combination requested."
        metric.__name__ = f"p{percentile}_" + metric_name

    return metric

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    y_true = np.random.rand(1, 100, 100, 100, 1)
    # y_pred = np.random.rand(1, 100, 100, 100, 1)
    y_pred = np.clip(y_true + np.random.rand(1, 100, 100, 100, 1) / 2, 0, 1)
    y_true = y_true > 0.5

    # ECE checking
    ece_0 = get_metric("ece", quantiles_as_bins=False, ece_from_bin_stats=False)(y_true, y_pred)
    ece_1 = get_metric("ece", quantiles_as_bins=True, ece_from_bin_stats=False)(y_true, y_pred)
    ece_2 = get_metric("ece", quantiles_as_bins=False, ece_from_bin_stats=True)(y_true, y_pred)
    ece_3 = get_metric("ece", quantiles_as_bins=True, ece_from_bin_stats=True)(y_true, y_pred)
    print(ece_0, ece_1, ece_2, ece_3)
    bin_stats_0 = get_metric("ece", quantiles_as_bins=False, ece_from_bin_stats=False, return_bin_stats=True)(y_true, y_pred)
    bin_stats_1 = get_metric("ece", quantiles_as_bins=True, ece_from_bin_stats=False, return_bin_stats=True)(y_true, y_pred)
    confidence_0, accuracy_0 = bin_stats_0[0, :, 0, 0, 0], bin_stats_0[0, :, 1, 0, 0]
    confidence_1, accuracy_1 = bin_stats_1[0, :, 0, 0, 0], bin_stats_1[0, :, 1, 0, 0]
    plt.figure()
    plt.plot(confidence_0, accuracy_0, "b.")
    plt.plot(confidence_1, accuracy_1, "r.")
    plt.title("ECE")
    plt.show()

    # AUC checking
    auc_0 = get_metric("auc", y_true_thresholds=None)(y_true, y_pred)
    auc_1 = get_metric("auc", y_true_thresholds=0.5)(y_true, y_pred)
    print(auc_0, auc_1)
    auc_stats_0 = get_metric("auc", return_auc_stats=True, y_true_thresholds=None)(y_true, y_pred)
    auc_stats_1 = get_metric("auc", return_auc_stats=True, y_true_thresholds=0.5)(y_true, y_pred)
    x_0, y_0 = auc_stats_0[0, :auc_stats_0.shape[1] // 2, 0, 0, 0], auc_stats_0[0, auc_stats_0.shape[1] // 2:, 0, 0, 0]
    x_1, y_1 = auc_stats_1[0, :auc_stats_1.shape[1] // 2, 0, 0, 0], auc_stats_1[0, auc_stats_1.shape[1] // 2:, 0, 0, 0]
    plt.figure()
    plt.plot(x_0, y_0, "b.")
    plt.plot(x_1, y_1, "r.")
    plt.title("AUC")
    plt.show()
