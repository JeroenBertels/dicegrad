import os
import glob
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scripts.jberte3.dicegrad.brats_datasets import create_dataset
from scripts.jberte3.KAROLINSKA2021.chapter_4.qualitative_figures import draw_figure, overlay_mask
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.analysis.plotting import Lineplot, Figure, Series
from deepvoxnet2.utilities.transformations import crop


def print_table():
    # primary metrics
    tp_fn = get_metric("true_positive", reduction_mode="sum")
    tn_fn = get_metric("true_negative", reduction_mode="sum")
    fp_fn = get_metric("false_positive", reduction_mode="sum")
    fn_fn = get_metric("false_negative", reduction_mode="sum")
    # calculate primary metrics and put into table
    runs = ["brats_benchmark_x", "brats_hgg_x"]
    subsets = ["hgg+", "lgg+", "lgg-"]
    experiments = ["perimage", "perbatch", "perimage-eps"]
    batch_sizes = [1, 2, 4, 8]
    indices = pd.MultiIndex.from_product([runs, batch_sizes], names=["run", "batch_size"])
    columns = pd.MultiIndex.from_product([subsets, experiments], names=["subset", "experiment"])
    table = pd.DataFrame(index=indices, columns=columns)
    # for run in runs:
    #     for subset in subsets:
    #         for experiment in experiments:
    #             for batch_size in batch_sizes:
    #                 print(run, subset, experiment, batch_size)
    #                 results = []
    #                 for fold_i in [0, 1, 2, 3, 4]:
    #                     val_data = Mirc(create_dataset("val", fold_i=fold_i, subset=subset[:-1], mask_subset=subset[:-1] if subset[-1] == "-" else None))
    #                     for dataset_id in val_data:
    #                         for case_id in val_data[dataset_id]:
    #                             y_true = val_data[dataset_id][case_id]["record_0"]["wt"].load() > 0.5
    #                             y_pred = nib.load(f"/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/Runs/{run}/{case_id}/record_0/Validation/{experiment}_auto-1e-3-30_bs-{batch_size}_f-{74 if 'benchmark' in run else 100}_Round_0_Fold_{fold_i}/full_val__s0__b0__x.nii.gz").get_fdata()[None] > 0.5
    #                             result = [
    #                                 tp_fn(y_true, y_pred).numpy().item(),
    #                                 tn_fn(y_true, y_pred).numpy().item(),
    #                                 fp_fn(y_true, y_pred).numpy().item(),
    #                                 fn_fn(y_true, y_pred).numpy().item()
    #                             ]
    #                             results.append(result)
    #
    #                 table.at[(run, batch_size), (subset, experiment)] = np.array(results)

    # save/load table with primary metrics
    results_table_path = "/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/git/micpy/scripts/jberte3/dicegrad/results.pkl"
    # table.to_pickle(results_table_path)
    table = pd.read_pickle(results_table_path)
    # calculate secondary metrics
    voxel_volume = 0.008
    volume_table = table.copy()
    dice_table = table.copy()
    for run in runs:
        for subset in subsets:
            for experiment in experiments:
                for batch_size in batch_sizes:
                    tps = table.at[(run, batch_size), (subset, experiment)][:, 0]
                    tns = table.at[(run, batch_size), (subset, experiment)][:, 1]
                    fps = table.at[(run, batch_size), (subset, experiment)][:, 2]
                    fns = table.at[(run, batch_size), (subset, experiment)][:, 3]
                    vol_diff = (fps - fns) * voxel_volume
                    dice_score = (2 * tps + 1e-7) / (2 * tps + fps + fns + 1e-7)
                    volume_table.at[(run, batch_size), (subset, experiment)] = "{:.0f}".format(np.mean(vol_diff))
                    dice_table.at[(run, batch_size), (subset, experiment)] = "{:.2f}".format(np.mean(dice_score))

                    if run == "brats_hgg_x":
                        tps_ = table.at[("brats_benchmark_x", batch_size), (subset, experiment)][:, 0]
                        tns_ = table.at[("brats_benchmark_x", batch_size), (subset, experiment)][:, 1]
                        fps_ = table.at[("brats_benchmark_x", batch_size), (subset, experiment)][:, 2]
                        fns_ = table.at[("brats_benchmark_x", batch_size), (subset, experiment)][:, 3]
                        vol_diff_ = (fps_ - fns_) * voxel_volume
                        dice_score_ = (2 * tps_ + 1e-7) / (2 * tps_ + fps_ + fns_ + 1e-7)
                        p = Series.basic_test(vol_diff_, vol_diff)
                        # p = Series.basic_test(dice_score_, dice_score)
                        if p > 0.95 or p < 0.05:
                            print(run, subset, experiment, batch_size)
                            volume_table.at[(run, batch_size), (subset, experiment)] = "{:.0f}".format(-np.mean(vol_diff))
                            # dice_table.at[(run, batch_size), (subset, experiment)] = "{:.2f}".format(-np.mean(dice_score))

    volume_table_ = volume_table.to_latex(multirow=True)\
        .replace("perimage", "I")\
        .replace("perbatch", "IB")\
        .replace("perimage", "Ieps")\
        .replace("brats\_benchmark\_x", "benchmark")\
        .replace("brats\_hgg\_x", "masked")\
        .replace("run", "run$\downarrow$")\
        .replace("batch\_size", "bs$\downarrow$")\
        .replace("experiment", r"exp$\rightarrow$")\
        .replace("subset", r"sub$\rightarrow$")
    print(volume_table_)
    dice_table_ = dice_table.to_latex(multirow=True)\
        .replace("perimage", "I")\
        .replace("perbatch", "IB")\
        .replace("perimage", "Ieps")\
        .replace("brats\_benchmark\_x", "benchmark")\
        .replace("brats\_hgg\_x", "masked")\
        .replace("run", "run$\downarrow$")\
        .replace("batch\_size", "bs$\downarrow$")\
        .replace("experiment", r"exp$\rightarrow$")\
        .replace("subset", r"sub$\rightarrow$")
    print(dice_table_)

def draw_plots():
    # calculate ROC curve for LGG/HGG classification based on volume
    legend_dict = {
        "perimage": r"$\mathbb{I}$",
        "perbatch": r"$\mathbb{BI}$",
        "perimage-eps": r"$\mathbb{I}_\epsilon$"
    }
    legend = []
    roc = Figure(
        xalim=[0, 1], yalim=[0, 1],
        awidthininches=5, aheightininches=5,
        xlabel=r"1-Specificity", ylabel=r"Sensitivity",
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        mask_inner_region=False, mask_outer_region=False,
        lmwidthininches=0.75, bmheightininches=0.75, tmheightininches=0.5,
        title=r"HGG classification in BRATS"
    )
    styles = [":", "-.", "--", "-"]
    alphas = [0.25, 0.50, 0.75, 1]
    colors = ["grey", "b", "r"]
    for experiment, color in zip(experiments, colors):
        for batch_size, style in zip(batch_sizes, styles):
            pred_vols, labels = [], []
            for subset in ["hgg+", "lgg+"]:
                tps = table.at[("brats_hgg_x", batch_size), (subset, experiment)][:, 0]
                tns = table.at[("brats_hgg_x", batch_size), (subset, experiment)][:, 1]
                fps = table.at[("brats_hgg_x", batch_size), (subset, experiment)][:, 2]
                fns = table.at[("brats_hgg_x", batch_size), (subset, experiment)][:, 3]
                pred_vols.append((fps + tps) * voxel_volume)
                labels.append([subset == "hgg+"] * len(tps))

            pred_vols, labels = np.concatenate(pred_vols), np.concatenate(labels)
            x, y = [], []
            for threshold in np.linspace(-1, 1000, 1002):
                labels_ = pred_vols >= threshold
                x.append(1 - np.sum(np.logical_not(labels_) * np.logical_not(labels)) / (np.sum(np.logical_not(labels)) + 1e-7))
                y.append(np.sum(labels_ * labels) / (np.sum(labels) + 1e-7))

            auc = np.abs(np.round(np.trapz(y, x), 2))
            # legend.append(r"{}, $|\mathcal{}|$={}: {:.2f}".format(legend_dict[experiment], r"{B}", batch_size, auc))
            legend.append(r"{}, B={}: {:.2f}".format(legend_dict[experiment], batch_size, auc))
            roc.lineplot(x, y, color=color, marker=None, linestyle=style, alpha=1)

    roc.legend(legend, fontsize=15)
    roc.savefig("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/git/micpy/scripts/jberte3/dicegrad/roc_hgg.pdf")
    roc.show()

    # sokkes ROCs
    data = loadmat("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/git/micpy/scripts/jberte3/dicegrad/sensitivity_specificity.mat")
    legend = []
    roc = Figure(
        xalim=[0, 1], yalim=[0, 1],
        awidthininches=5, aheightininches=5,
        xlabel=r"1-Specificity", ylabel=r"Sensitivity",
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        mask_inner_region=False, mask_outer_region=False,
        lmwidthininches=0.75, bmheightininches=0.75, tmheightininches=0.5,
        title=r"MYO\textsubscript{ED} classification in ACDC"
    )
    styles = ["-."]
    for i, color in enumerate(colors):
        for j, style in enumerate(styles):
            # y = data["sensitivity_all"][:, i]
            # x = 1 - data["specificity_all"][:, i]
            y = data["specificity_all"][:, i]
            x = 1 - data["sensitivity_all"][:, i]
            auc = np.abs(np.round(np.trapz(y, x), 2))
            # legend.append(r"{}, $|\mathcal{}|$={}: {:.2f}".format(legend_dict[experiments[i]], r"{B}", 2, auc))
            legend.append(r"{}, B={}: {:.2f}".format(legend_dict[experiments[i]], 2, auc))
            roc.lineplot(x, y, color=color, marker=None, linestyle=style, alpha=1)

    roc.legend(legend, fontsize=15)
    roc.savefig("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/git/micpy/scripts/jberte3/dicegrad/roc_cardio.pdf")
    roc.show()


datasets_dir = "/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets"
figures_dir = "/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/git/micpy/scripts/jberte3/dicegrad/figures"


def draw_example(case_id, slice_id=34, crop_size=(100, 100), line_thickness=1, line_type="-"):
    mr_path = f"/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/BRATS/2018/Training/{case_id}/FLAIR_2mm.nii"
    gt_path = f"/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/BRATS/2018/Training/{case_id}/GT_W_2mm.nii"
    # gt_path = f"/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/BRATS/2018/Training/{case_id}/GT_W_2mm_empty.nii"
    mr = np.transpose(nib.load(mr_path).get_fdata(), (1, 0, 2))[..., slice_id]
    gt = np.transpose(nib.load(gt_path).get_fdata(), (1, 0, 2))[..., slice_id]
    fig, axs = plt.subplots(4, 4)
    for i, batch_size in enumerate([1, 2, 4, 8]):
        for j, setup in enumerate(["GT", "perimage", "perbatch", "perimage-eps"]):
            if setup == "GT":
                pred = gt

            else:
                pred_path = glob.glob(f"/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/Runs/brats_hgg_x/{case_id}/record_0/Validation/{setup}_auto-1e-3-30_bs-{batch_size}_f-100_Round_0_Fold_*/full_val__s0__b0__x.nii.gz")[0]
                pred = np.transpose(nib.load(pred_path).get_fdata(), (1, 0, 2, 3))[..., slice_id, 0]

            overlay = draw_figure(mr, mask_array=pred, shift=0, scale=255/1000, line_color=(0, 255, 255), line_thickness=line_thickness, line_type=line_type if setup == "GT" else line_type)
            overlay = crop(overlay, crop_size).astype("uint8")
            out_dir = os.path.join(figures_dir, case_id, str(slice_id))
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            cv2.imwrite(os.path.join(out_dir, f"{setup}_{batch_size}.png"), overlay)
            axs[i, j].imshow(overlay[..., ::-1])
            axs[i, j].axis('off')

    plt.show()


if __name__ == "__main__":
    print_table()
    draw_plots()
    # draw_example("case_211", slice_id=34)
    draw_example("case_191", slice_id=30)
