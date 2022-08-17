import os
import pickle
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.keras.models.unet_generalized import create_generalized_unet_model
from deepvoxnet2.components.transformers import Resample, GeometricCrop, NormalizeIndividual, IntensityTransform, AffineDeformation, MircInput, ElasticDeformation, Flip, Put, KerasModel
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.losses import get_loss
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.keras.callbacks import DvnModelEvaluator, DvnModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from deepvoxnet2.factories.directory_structure import MircStructure
from BRATS.dataset import create_dataset


def train(dataset_dir, base_dir, run_name, fold_i, mask_subset, fraction, phi, epsilon, batch_size):
    # first create the datasets
    train_data = Mirc(create_dataset(dataset_dir, data="train", fold_i=fold_i, mask_subset=mask_subset, fraction=fraction))
    val_data = Mirc(create_dataset(dataset_dir, data="val", fold_i=fold_i, mask_subset=mask_subset, fraction=1))
    full_val_data = Mirc(create_dataset(dataset_dir, data="val", fold_i=fold_i, mask_subset=None, fraction=1))
    #
    # create samplers from the created datasets
    train_sampler = MircSampler(train_data, shuffle=True)
    val_sampler = MircSampler(val_data)
    full_val_sampler = MircSampler(full_val_data)
    #
    # create DeepVoxNet2 network
    segment_size = (80, 80, 48)
    keras_model = create_generalized_unet_model(
        number_input_features=4,
        output_size=segment_size,
        padding="same"
    )
    keras_model_transformer = KerasModel(keras_model)
    x_input, y_input, y_orig_input = MircInput(["input"]), MircInput(["output"]), MircInput(["output_orig"])
    x_input_2, y_input_2, y_orig_input_2 = Resample((2, 2, 2))(x_input), Resample((2, 2, 2))(y_input), Resample((2, 2, 2))(y_orig_input)
    # used for training and is on the level of patches
    x_path, y_path, y_orig_path = NormalizeIndividual(ignore_value=0)(x_input_2), y_input_2, y_orig_input_2
    x_path, y_path = AffineDeformation(x_path, translation_window_width=(6, 6, 4), rotation_window_width=(3.14 / 12, 0, 0), width_as_std=True)(x_path, y_path)
    x_path, y_path = ElasticDeformation(x_path, shift=(1, 1, 1))(x_path, y_path)
    x_path, y_path = GeometricCrop(y_orig_path, segment_size)(x_path, y_path)
    x_path, y_path = Flip((0.5, 0, 0))(x_path, y_path)
    x_path = IntensityTransform(std_shift=0.05, std_scale=0.05)(x_path)
    x_train, y_train = keras_model_transformer(x_path), y_path
    # used for validation and is on the level of patches
    x_path, y_path, y_orig_path = NormalizeIndividual(ignore_value=0)(x_input_2), y_input_2, y_orig_input_2
    x_path, y_path = GeometricCrop(y_orig_path, segment_size)(x_path, y_path)
    x_val, y_val = keras_model_transformer(x_path), y_path
    # used for validation of the full images and thus is on the level of the input
    x_full_val, y_full_val = Put(x_input_2)(x_val), y_orig_input_2
    # create DVN2 model
    dvn_model = DvnModel(
        outputs={
            "train": [x_train, y_train],
            "val": [x_val, y_val],
            "full_val": [x_full_val, y_full_val],
            "full_test": [x_full_val]
        }
    )
    #
    # create correct loss function based on options specified and metrics
    soft_dice = get_loss("dice_loss", reduce_along_batch="B" in phi, reduce_along_features="C" in phi, eps=epsilon)
    dice_score = get_metric("dice_coefficient", threshold=0.5)
    abs_vol_diff = get_metric("absolute_volume_error", voxel_volume=0.008)
    #
    # compile DVN2 model (and Keras model that sits inside it)
    dvn_model.compile("train", optimizer=Adam(learning_rate=1e-3), losses=[soft_dice], metrics=[[dice_score, abs_vol_diff]])
    dvn_model.compile("val", losses=[soft_dice], metrics=[[dice_score, abs_vol_diff]])
    dvn_model.compile("full_val", losses=[soft_dice], metrics=[[dice_score, abs_vol_diff]])
    #
    # directory structure
    output_structure = MircStructure(
        base_dir=base_dir,
        run_name=run_name,
        experiment_name=f"{mask_subset}_{fraction}_{phi}_{epsilon}_{batch_size}",
        fold_i=fold_i,
        round_i=None,
        training_mirc=train_data,
        validation_mirc=val_data
    )
    output_structure.create()
    #
    # create some callbacks and start training
    callbacks = [
        DvnModelEvaluator(dvn_model, "full_val", full_val_sampler, freq=10, output_dirs=output_structure.val_images_output_dirs),
        DvnModelCheckpoint(dvn_model, output_structure.models_dir, freq=10),
        ReduceLROnPlateau(monitor="val__loss", factor=0.2, patience=30, mode="min", min_delta=0.0001),
        EarlyStopping(monitor="val__loss", patience=60, mode="min", min_delta=0.0001)
    ]
    history = dvn_model.fit("train", train_sampler, batch_size=batch_size, validation_key="val", validation_sampler=val_sampler, callbacks=callbacks, epochs=1000, logs_dir=output_structure.logs_dir)
    dvn_model.evaluate("full_val", val_sampler, output_dirs=output_structure.val_images_output_dirs)
    with open(output_structure.history_path, "wb") as f:
        pickle.dump(history.history, f)

    dvn_model.save(os.path.join(output_structure.models_dir, "dvn_model_final"))


if __name__ == '__main__':
    train(
        dataset_dir="/usr/local/micapollo01/MIC/DATA/SHARED/STAFF/jberte3/BRATS_Challenge/2018/Raw_data/MICCAI_BraTS_2018_Data_Training",
        base_dir="/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/Runs",
        run_name="experiments_of_paper",
        fold_i=0,
        mask_subset=None,
        fraction=1,
        phi="BIC",
        epsilon=1e-7,
        batch_size=2
    )
