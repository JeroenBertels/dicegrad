import os
import numpy as np
import nibabel as nib
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, Modality, NiftiFileMultiModality


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class NiftiFileEmptyModality(Modality):
    def __init__(self, modality_id, file_path):
        super(NiftiFileEmptyModality, self).__init__(modality_id, os.path.dirname(file_path))
        self.file_path = file_path

    def load(self):
        nii = nib.load(self.file_path)
        return Sample(np.zeros(nii.shape), nii.affine)


class NiftiFileBinaryModality(Modality):
    def __init__(self, modality_id, file_path):
        super(NiftiFileBinaryModality, self).__init__(modality_id, os.path.dirname(file_path))
        self.file_path = file_path

    def load(self):
        nii = nib.load(self.file_path)
        return Sample(nii.get_fdata().astype(bool), nii.affine)


def create_dataset(data=None, fold_i=None, subset=None, nb_folds=5, fraction=1., mask_subset=None):
    dataset_dir = os.path.join(BASE_DIR, "MICCAI_BraTS_2018_Data_Training")
    hgg_dir = os.path.join(dataset_dir, "HGG")
    lgg_dir = os.path.join(dataset_dir, "LGG")
    hgg_cases = [f"HGG/{dir_name}" for dir_name in sorted(os.listdir(hgg_dir)) if dir_name.startswith("Brats18")]
    lgg_cases = [f"LGG/{dir_name}" for dir_name in sorted(os.listdir(lgg_dir)) if dir_name.startswith("Brats18")]
    all_cases = hgg_cases + lgg_cases
    np.random.seed(0)
    np.random.shuffle(all_cases)
    if data is None:
        assert fold_i is None, "When all data is requested it does not make sense to provide a fold number."
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
    dataset = Dataset("BRATS2018")
    for case_id in case_ids:
        tumor_type, subject_id = case_id.split('/')
        case = Case(case_id)
        record = Record("record_0")
        record.add(NiftiFileMultiModality(
            "input",
            [
                os.path.join(dataset_dir, case_id, f"{subject_id}_flair.nii.gz"),
                os.path.join(dataset_dir, case_id, f"{subject_id}_t1.nii.gz"),
                os.path.join(dataset_dir, case_id, f"{subject_id}_t1ce.nii.gz"),
                os.path.join(dataset_dir, case_id, f"{subject_id}_t2.nii.gz")
            ]))
        record.add(NiftiFileBinaryModality("output", os.path.join(dataset_dir, case_id, f"{subject_id}_seg.nii.gz")))
        record.add(NiftiFileBinaryModality("output_orig", os.path.join(dataset_dir, case_id, f"{subject_id}_seg.nii.gz")))
        if tumor_type == "LGG":
            if subset == "hgg":
                continue

            if mask_subset == "lgg":
                record["output"] = NiftiFileEmptyModality("output", os.path.join(dataset_dir, case_id, f"{subject_id}_seg.nii.gz"))

        else:
            if subset == "lgg":
                continue

            if mask_subset == "hgg":
                record["output"] = NiftiFileEmptyModality("output", os.path.join(dataset_dir, case_id, f"{subject_id}_seg.nii.gz"))

        case.add(record)
        dataset.add(case)

    return dataset


if __name__ == "__main__":
    brats_dataset = create_dataset(mask_subset="lgg", fraction=0.1)
    mirc = Mirc(brats_dataset)
    mirc.inspect(["input", "output"], ns=0)
    mean_wt_size = np.mean([brats_dataset[case_id]["record_0"]["output"].load().sum() for case_id in brats_dataset])
    print("Mean whole tumor size: ", mean_wt_size)
