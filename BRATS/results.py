import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.analysis.plotting import Lineplot, Figure, Series
from deepvoxnet2.utilities.transformations import crop
from deepvoxnet2.utilities.drawing import draw_figure, overlay_mask
from BRATS.dataset import create_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dice_score = get_metric("dice_coefficient", threshold=0.5)
vol_diff = get_metric("volume_error", voxel_volume=0.001)
pred_vol = get_metric("pred_volume", voxel_volume=0.001)


def make_table(mask_subset, fraction, phi, epsilon, batch_size, fold_is=(0, 1, 2, 3, 4)):
    indices = pd.MultiIndex.from_product([(fraction,), (mask_subset,), (batch_size,)], names=["fraction", "mask_subset", "batch_size"])
    columns = pd.MultiIndex.from_product([("hgg", "lgg"), (phi,), (epsilon,)], names=["subset", "phi", "epsilon"])
    table = pd.DataFrame(index=indices, columns=columns)
    experiment_name = f"{mask_subset}_{fraction}_{phi}_{epsilon}_{batch_size}"
    for subset in ("hgg", "lgg"):
        results = []
        for fold_i in fold_is:
            val_data = create_dataset(data="val", fold_i=fold_i, mask_subset=mask_subset, fraction=fraction, subset=subset)
            for case_id in val_data:
                y_true = val_data[case_id]["record_0"]["output"].load()
                y_pred = nib.load(os.path.join(BASE_DIR, f"experiments_of_paper/{case_id}/record_0/Validation/{experiment_name}_Round_0_Fold_{fold_i}/full_val__s0__b0__x.nii.gz")).get_fdata()[None]
                result = (
                    dice_score(y_true, y_pred).numpy().item(),
                    vol_diff(y_true, y_pred).numpy().item(),
                    pred_vol(y_true, y_pred).numpy().item()
                )
                results.append(result)

        table.at[(fraction, mask_subset, batch_size), (subset, phi, epsilon)] = np.array(results)

    return table


def make_plot(table):
    fraction, mask_subset, batch_size = table.index[0]
    _, phi, epsilon = table.columns[0]
    pred_vols, labels = [], []
    for subset in ("hgg", "lgg"):
        pred_vols_ = list(table.at[(fraction, mask_subset, batch_size), (subset, phi, epsilon)][:, 2])
        pred_vols += pred_vols_
        labels += [subset == "hgg"] * len(pred_vols_)

    if phi == "I" and epsilon == 1e-7:
        legend = r"$\mathbb{I}$"

    elif phi == "BI" and epsilon == 1e-7:
        legend = r"$\mathbb{BI}$"

    elif phi == "I" and epsilon == 8789:
        legend = r"$\mathbb{I}_\epsilon$"

    else:
        raise ValueError("This experiment config was not performed in the paper.")

    styles_dict = {1: ":", 2: "-.", 4: "--", 8: "-"}
    colors_dict = {r"$\mathbb{I}$": "grey", r"$\mathbb{BI}$": "b", r"$\mathbb{I}_\epsilon$": "r"}
    roc = Figure(
        xalim=[0, 1], yalim=[0, 1],
        awidthininches=5, aheightininches=5,
        xlabel=r"1-Specificity", ylabel=r"Sensitivity",
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        mask_inner_region=False, mask_outer_region=False,
        lmwidthininches=0.75, bmheightininches=0.75, tmheightininches=0.5,
        title=r"HGG classification in BRATS"
    )
    pred_vols, labels = np.asarray(pred_vols), np.asarray(labels)
    x, y = [], []
    for threshold in np.linspace(-1, 1000, 1002):
        labels_ = pred_vols >= threshold
        x.append(1 - np.sum(np.logical_not(labels_) * np.logical_not(labels)) / (np.sum(np.logical_not(labels)) + 1e-7))
        y.append(np.sum(labels_ * labels) / (np.sum(labels) + 1e-7))

    auc = np.abs(np.round(np.trapz(y, x), 2))
    roc.lineplot(x, y, color=colors_dict[legend], marker=None, linestyle=styles_dict[batch_size], alpha=1)
    roc.legend([r"{}, B={}: {:.2f}".format(legend, batch_size, auc)], fontsize=15)
    roc.show()
    return roc


def make_example(table, case_id, slice_id=68, crop_size=(200, 200), line_thickness=2, line_type="-", draw_y_true=False):
    fraction, mask_subset, batch_size = table.index[0]
    _, phi, epsilon = table.columns[0]
    experiment_name = f"{mask_subset}_{fraction}_{phi}_{epsilon}_{batch_size}"
    data = create_dataset()
    flair = data[case_id]["record_0"]["input"].load()[0, ..., slice_id, 0].T
    y_true = data[case_id]["record_0"]["output"].load()[0, ..., slice_id, 0].T
    if draw_y_true:
        y_pred = y_true

    else:
        y_pred = nib.load(glob.glob(os.path.join(BASE_DIR, f"experiments_of_paper/{case_id}/record_0/Validation/{experiment_name}_Round_0_Fold_*/full_val__s0__b0__x.nii.gz"))[0]).get_fdata()[..., slice_id, 0].T

    fig = plt.figure()
    overlay = draw_figure(flair, mask_array=y_pred, shift=0, scale=255/1000, line_color=(0, 255, 255), line_thickness=line_thickness, line_type=line_type)
    overlay = crop(overlay, crop_size).astype("uint8")
    plt.imshow(overlay[..., ::-1])
    plt.axis('off')
    return fig


if __name__ == "__main__":
    results = make_table(
        fold_is=(0,),
        mask_subset="lgg",
        fraction=1,
        phi="BI",
        epsilon=1e-7,
        batch_size=4)
    print(results.applymap(lambda x: np.mean(x, axis=0).round(2)))
    roc = make_plot(results)
    roc.show()
    example_b = make_example(results, "HGG/Brats18_CBICA_BFB_1", slice_id=60)
    example_b.show()
    # example_a = make_example(results, "LGG/Brats18_TCIA13_630_1", slice_id=68)
    # example_a.show()
