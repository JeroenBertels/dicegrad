import os
import argparse
import pickle
import numpy as np
from scripts.jberte3.dicegrad.brats_datasets import create_dataset
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.keras.models.unet_generalized import create_generalized_unet_model
from deepvoxnet2.components.transformers import GeometricCrop, NormalizeIndividual, IntensityTransform, AffineDeformation, MircInput, ElasticDeformation, Flip, Put, KerasModel, Concat
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.losses import get_loss
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.keras.callbacks import DvnModelEvaluator, DvnModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from deepvoxnet2.factories.directory_structure import MircStructure


def main(run_name, experiment_name, fold_i=0, batch_size=1, fraction=1):
    # load datasets
    train_data = Mirc(create_dataset("train", fold_i=fold_i, fraction=fraction, mask_subset="lgg"))
    val_data = Mirc(create_dataset("val", fold_i=fold_i, mask_subset="lgg"))
    full_val_data = Mirc(create_dataset("val", fold_i=fold_i))
    train_sampler = MircSampler(train_data, shuffle=True)
    val_sampler = MircSampler(val_data)
    full_val_sampler = MircSampler(full_val_data)
    # create dvn network
    keras_model = create_generalized_unet_model(
        number_input_features=4,
        output_size=(80, 80, 48),
        padding="same"
    )
    keras_model_transformer = KerasModel(keras_model)
    x_input = Concat()([
        MircInput(["flair"], output_shapes=[(1, 120, 120, 78, 1)]),
        MircInput(["t1"], output_shapes=[(1, 120, 120, 78, 1)]),
        MircInput(["t1ce"], output_shapes=[(1, 120, 120, 78, 1)]),
        MircInput(["t2"], output_shapes=[(1, 120, 120, 78, 1)])
    ])
    y_input = MircInput(["wt"], output_shapes=[(1, 120, 120, 78, 1)])
    y_orig_input = MircInput(["wt_orig"], output_shapes=[(1, 120, 120, 78, 1)])
    x_input = NormalizeIndividual(ignore_value=0)(x_input)
    # used for training and is on the level of patches
    x_path, y_path = AffineDeformation(x_input, translation_window_width=(6, 6, 4), rotation_window_width=(3.14 / 12, 0, 0), width_as_std=True)(x_input, y_input)
    x_path, y_path = ElasticDeformation(x_path, shift=(1, 1, 1))(x_path, y_path)
    x_path, y_path = GeometricCrop(y_orig_input, keras_model.output_shape[1:4])(x_path, y_path)
    x_path, y_path = Flip((0.5, 0, 0))(x_path, y_path)
    x_path = IntensityTransform(std_shift=0.05, std_scale=0.05)(x_path)
    x_train = keras_model_transformer(x_path)
    y_train = y_path
    # used for validation and is on the level of patches
    x_path, y_path = GeometricCrop(y_orig_input, keras_model.output_shape[1:4])(x_input, y_input)
    x_val = keras_model_transformer(x_path)
    y_val = y_path
    # used for validation of the full images and thus is on the level of the input
    x_full_val = Put(x_input)(x_val)
    y_full_val = y_orig_input
    # dvn model and training
    dvn_model = DvnModel(
        outputs={
            "train": [x_train, y_train],
            "val": [x_val, y_val],
            "full_val": [x_full_val, y_full_val],
            "full_test": [x_full_val]
        }
    )
    soft_dice = get_loss("dice_loss", reduce_along_batch=False, reduce_along_features=False, eps=8789)
    dice_score = get_metric("dice_coefficient", threshold=0.5)
    abs_vol_diff = get_metric("absolute_volume_error", voxel_volume=0.008)
    dvn_model.compile("train", optimizer=Adam(learning_rate=1e-3), losses=[soft_dice], metrics=[[dice_score, abs_vol_diff]])
    dvn_model.compile("val", losses=[soft_dice], metrics=[[dice_score, abs_vol_diff]])
    dvn_model.compile("full_val", losses=[soft_dice], metrics=[[dice_score, abs_vol_diff]])
    # directory structure
    output_structure = MircStructure(
        base_dir=os.path.join("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/Runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=None,
        training_mirc=train_data,
        validation_mirc=val_data
    )
    output_structure.create()
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
    parser = argparse.ArgumentParser(description='brats_hgg_perimage_eps')
    parser.add_argument('fold_i', type=int, default=0, nargs="?")
    parser.add_argument('batch_size', type=int, default=1, nargs="?")
    parser.add_argument('fraction', type=float, default=1, nargs="?")
    args = parser.parse_args()
    main(
        run_name="brats_hgg_x",
        experiment_name=f"perimage-eps_auto-1e-3-30_bs-{args.batch_size}_f-{int(np.round(args.fraction * 100))}",
        fold_i=args.fold_i,
        batch_size=args.batch_size,
        fraction=args.fraction
    )
