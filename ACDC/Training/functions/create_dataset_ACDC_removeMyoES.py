import os
from deepvoxnet2.components.mirc import Dataset, Case, Record, NiftyFileModality
import scipy.io
import numpy as np
import nibabel as nib

def create_dataset(data_path, data="train",fold_i=0,removeMyoES = True,removeES = False,ED_fraction = 0.5):
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
    ED_sampling = np.max([1,ED_fraction/(1-ED_fraction)])
    ES_sampling = np.max([1, (1-ED_fraction) / (ED_fraction)])
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
        mr_file = os.path.join(data_path, "ACDC_{:04d}.nii.gz".format(case_name))
        gt_file_init = os.path.join(data_path, "ACDC_{:04d}_gt_categorical.nii.gz".format(case_name))
        if case_name>100 and removeES:
            valid_datafile = False
        elif case_name>100 and removeMyoES:
            gt_file = gt_file_init.replace('.nii.gz','_removeMyoES.nii.gz')
            if not os.path.exists(gt_file):
                gt = nib.load(gt_file_init)
                gt_img = gt.get_fdata()
                gt_img[gt_img == 2] = 0
                nib.save(nib.Nifti1Image(gt_img, gt.affine), gt_file)
            valid_datafile = True
        else:
            gt_file = gt_file_init
            valid_datafile = True

        if case_name>100:
            no_sampling = np.int(ES_sampling)
        else:
            no_sampling = np.int(ED_sampling)

        if valid_datafile:
            for i in range(no_sampling):
                case = Case('case_{:04d}_{}'.format(case_name,i))
                record = Record("record_0")
                record.add(NiftyFileModality("MR",mr_file))
                record.add(NiftyFileModality("GT",gt_file))
                case.add(record)
                dataset.add(case)

    return dataset


if __name__ == '__main__':
    data_path = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MICCAI_2022_scripts/Preprocessed_images'
    create_dataset(data_path, data="train", fold_i=0)
