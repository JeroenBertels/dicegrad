import os
from deepvoxnet2.components.mirc import Dataset, Case, Record, NiftyFileModality
import scipy.io
import numpy as np

def create_dataset(data_path,data="train",fold_i=0):
    fold_file = os.path.join(data_path, 'folds.mat')
    met = scipy.io.loadmat(fold_file)
    folds_images = np.array(met.get('folds_images'))
    folds_images = np.squeeze(folds_images)
    folds = []
    for f in range(1, 6):
        folds.append([idx for idx, fold in enumerate(folds_images) if fold == f])

    subjects = range(1,len(folds_images)+1)
    test_idx = folds[fold_i]
    subjects_val = [p for i, p in enumerate(subjects) if i in test_idx]
    subjects_train = [p for i, p in enumerate(subjects) if not i in test_idx]

    if data == "train":
        case_names = subjects_train

    elif data == "val":
        case_names = subjects_val

    elif data == "test":
        case_names = []

    else:
        raise ValueError("data must be 'train', 'val' or 'test'")

    dataset = Dataset("ACDC_" + data, data_path)
    for case_name in case_names:
        case = Case('case_{:04d}'.format(case_name))
        record = Record("record_0")
        record.add(NiftyFileModality("MR", os.path.join(data_path, "ACDC_{:04d}.nii.gz".format(case_name))))
        record.add(NiftyFileModality("GT", os.path.join(data_path, "ACDC_{:04d}_gt_categorical.nii.gz".format(case_name))))
        case.add(record)
        dataset.add(case)

    return dataset

if __name__ == '__main__':
    data_path = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MICCAI_2022_scripts/Preprocessed_images'
    create_dataset(data_path, data="train", fold_i=0)
