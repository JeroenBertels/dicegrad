import os
import json
import numpy as np
from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, NiftyFileModality, ArrayModality
from matplotlib import pyplot as plt


DATASET_DIR = None
assert DATASET_DIR is not None, "Please provide a dataset directory. It is assumed that the data is organized as /DATASET_DIR/case_<i>/<modality><tag>.nii.gz."


def create_dataset(data=None, fold_i=None, subset=None, nb_folds=5, tag="_2mm", fraction=1, mask_subset=None):
    dataset_dir = DATASET_DIR
    all_cases = [dir_name for dir_name in sorted(os.listdir(dataset_dir)) if dir_name.startswith("case")]
    np.random.seed(0)
    np.random.shuffle(all_cases)
    if data is None:
        assert fold_i is None
        case_ids = all_cases

    else:
        assert isinstance(fold_i, int) and fold_i < nb_folds
        max_nb_cases_per_fold = int(np.ceil(len(all_cases) / nb_folds))
        val_cases = all_cases[fold_i * max_nb_cases_per_fold:(fold_i + 1) * max_nb_cases_per_fold]
        train_cases = [case_i for case_i in all_cases if case_i not in val_cases]
        if data == "train":
            case_ids = train_cases

        elif data == "val":
            case_ids = val_cases

        else:
            raise ValueError("data must be 'train' or 'val'")

    case_ids = case_ids[:int(np.round(len(case_ids) * fraction))]
    median_wt_vol = 87  # 87.527 at native resolution; 86.807 at 2mm resolution
    dataset = Dataset("BRATS2018")
    for case_id in case_ids:
        case = Case(case_id)
        record = Record("record_0")
        record.add(NiftyFileModality("flair", os.path.join(dataset_dir, case_id, f"FLAIR{tag}.nii")))
        record.add(NiftyFileModality("t1", os.path.join(dataset_dir, case_id, f"T1{tag}.nii")))
        record.add(NiftyFileModality("t1ce", os.path.join(dataset_dir, case_id, f"T1_CE{tag}.nii")))
        record.add(NiftyFileModality("t2", os.path.join(dataset_dir, case_id, f"T2{tag}.nii")))
        record.add(NiftyFileModality("wt", os.path.join(dataset_dir, case_id, f"GT_W{tag}.nii")))
        record.add(NiftyFileModality("wt_orig", os.path.join(dataset_dir, case_id, f"GT_W{tag}.nii")))
        with open(os.path.join(dataset_dir, case_id, f"GT_W_volume{tag}.txt"), "r") as f:
            wt_vol = json.load(f)

        record.add(ArrayModality("wt_vol", wt_vol))
        record.add(ArrayModality("wt_vol_orig", wt_vol))
        with open(os.path.join(dataset_dir, case_id, f"subject_id.txt"), "r") as f:
            text = f.read()

        subject_id = text.split("\n")[0].split(": ")[1]
        if wt_vol >= median_wt_vol and mask_subset == "large":
            record["wt"] = NiftyFileModality("wt", os.path.join(dataset_dir, case_id, f"GT_W{tag}_empty.nii"))
            record["wt_vol"] = ArrayModality("wt_vol", 0)

        if wt_vol >= median_wt_vol and subset == "small":
            continue

        if wt_vol < median_wt_vol and mask_subset == "small":
            record["wt"] = NiftyFileModality("wt", os.path.join(dataset_dir, case_id, f"GT_W{tag}_empty.nii"))
            record["wt_vol"] = ArrayModality("wt_vol", 0)

        if wt_vol < median_wt_vol and subset == "large":
            continue

        lgg_subject_ids = os.listdir("/usr/local/micapollo01/MIC/DATA/SHARED/STAFF/jberte3/BRATS_Challenge/2018/Raw_data/MICCAI_BraTS_2018_Data_Training/LGG")
        if subject_id in lgg_subject_ids and mask_subset == "lgg":
            record["wt"] = NiftyFileModality("wt", os.path.join(dataset_dir, case_id, f"GT_W{tag}_empty.nii"))
            record["wt_vol"] = ArrayModality("wt_vol", 0)

        if subject_id in lgg_subject_ids and subset == "hgg":
            continue

        if subject_id not in lgg_subject_ids and mask_subset == "hgg":
            record["wt"] = NiftyFileModality("wt", os.path.join(dataset_dir, case_id, f"GT_W{tag}_empty.nii"))
            record["wt_vol"] = ArrayModality("wt_vol", 0)

        if subject_id not in lgg_subject_ids and subset == "lgg":
            continue

        case.add(record)
        dataset.add(case)

    return dataset


if __name__ == "__main__":
    brats_dataset = create_dataset(subset="lgg")
    mirc = Mirc(brats_dataset)
    volumes = []
    for case_id in brats_dataset:
        volumes.append(brats_dataset[case_id]["record_0"]["wt_vol"].load().item())

    print(np.mean(volumes), np.mean(volumes) / 0.008)
    plt.figure()
    plt.hist(volumes, bins=range(0, 500, 25))
    plt.show()
    print(np.median(volumes))
