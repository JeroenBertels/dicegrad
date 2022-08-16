import numpy as np
import nibabel as nib

def calculateEpsPerClass(dataset,nb_classes=1,onlyPositives = False):
    gt_files = [dataset[dataset_id][case_id][record_id]["GT"].file_path for dataset_id in dataset for case_id in dataset[dataset_id] for
     record_id in dataset[dataset_id][case_id]]

    vol = np.zeros((nb_classes,1))
    no_images_used = np.zeros((nb_classes,1))
    for gt_file in gt_files:
        gt = nib.load(gt_file)
        gt_img = gt.get_fdata()
        for i in range(nb_classes):
            vol_=np.sum([gt_img==i])
            if vol_ != 0 or not onlyPositives:
                vol[i] += vol_
                no_images_used[i] += 1

    mean_vol = vol/len(gt_files)
    eps = mean_vol

    return eps


if __name__ == '__main__':
    print('no main initizalized')