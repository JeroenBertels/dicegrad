import os
import pickle
import sys
sys.path.append('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Code/GitHub/deepvoxnet2')
sys.path.append('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/tmp/pymirc-master')
sys.path.append(os.path.join(os.getcwd(),'functions'))
from create_dataset_ACDC import create_dataset as create_dataset_ACDC
from calculateEpsPerClass import calculateEpsPerClass
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.keras.models.unet_generalized import create_generalized_unet_model
from deepvoxnet2.components.transformers import AffineDeformation, MircInput, GridCrop, Put, KerasModel, Buffer, GaussianNoise, ToCategorical
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.components.creator import Creator
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.losses import get_loss
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.keras.callbacks import DvnModelEvaluator, LearningRateScheduler, DvnModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from deepvoxnet2.factories.directory_structure import MircStructure
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

main_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MICCAI_2022_scripts'

def train(run_name, experiment_name, fold_i=0, base_dir = None,reduce_along_batch = False,reduce_along_features =False,eps_corr = 1,onlyPositives = False,noEpsLVRV=False):
    nb_classes = 4

    train_data = Mirc()
    train_data.add(create_dataset_ACDC(os.path.join(main_dir,'Preprocessed_images'),"train", fold_i=fold_i))
    val_data = Mirc()
    val_data.add(create_dataset_ACDC(os.path.join(main_dir,'Preprocessed_images'),"val",fold_i=fold_i))

    adapt_epochs = 160/len(train_data.get('ACDC_train'))

    eps_estimate = calculateEpsPerClass(train_data,nb_classes,onlyPositives=onlyPositives)
    eps_estimate = eps_estimate*eps_corr
    if noEpsLVRV:
        eps_estimate = np.array([tf.keras.backend.epsilon(),tf.keras.backend.epsilon(),eps_estimate[2][0],tf.keras.backend.epsilon()])

    print('{}_{}_{}'.format(eps_estimate[1],eps_estimate[2],eps_estimate[3]))

    # building mirc samplers: here the sampler randomly samples a record out of train_data or val_data --> depending on what objects these samplers return, you must choose an appropriate Input later on when building your Dvn network/creator
    train_sampler = MircSampler(train_data, shuffle=True)
    val_sampler = MircSampler(val_data)

    # let's create a keras model and put it in a Transformer layer to be used in our Dvn network (see "create_samples.py" for other examples on Transformers and sample creation)
    # have a look at the print-outs when this model is created; you'll see some interesting properties like # parameters, field of view, output/input sizes, etc.
    keras_model = create_generalized_unet_model(
        number_input_features=1,
        output_size=(192,192,48),
        padding="same",
        number_features_common_pathway=(nb_classes,),
        activation_final_layer='softmax',
        number_features_per_pathway=(
            ((16,16), (16,16)),
            ((32,32), (32,32)),
            ((64,64), (64,64)),
            ((128,128), (128,128)),
            ((256, 256), ())
        ),
        subsample_factors_per_pathway=(
            (1, 1, 1),
            (2, 2, 1),
            (4, 4, 2),
            (8, 8, 4),
            (16, 16, 8)
        ),
        kernel_sizes_per_pathway=(
            (((3, 3, 3), (3, 3, 1)), ((3, 3, 3), (3, 3, 1))),
            (((3, 3, 3), (3, 3, 1)), ((3, 3, 3), (3, 3, 1))),
            (((3, 3, 3), (3, 3, 1)), ((3, 3, 3), (3, 3, 1))),
            (((3, 3, 3), (3, 3, 1)), ((3, 3, 3), (3, 3, 1))),
            (((3, 3, 3), (3, 3, 1)), ())
        ),
        batch_normalization = True,
        instance_normalization = False,
        activation='prelu',
    )
    keras_model_transformer = KerasModel(keras_model)

    # similar to the demo on sample creation, let's make our processing network(s) (keep in mind that the following network could be made in different ways; we just show one way)
    # inputs (we have samplers that sample identifier objects, so we can use MircInputs here; they now what to do with the sampled identifier objects (have a look at their load method)
    # n = number of samples that is taken from the input (None -> keeps going on forever, except for GridCrop)
    x_input = MircInput(["MR"], output_shapes=[(1, None, None, None, 1)])
    y_input = MircInput(["GT"], output_shapes=[(1, None, None, None, 1)], n=None)

    # every transformer knows how the 'affine' of the image should be adapted (especially useful if you use the Put transformer in the end)

    # used for training and is on the level of patches
    y_path = ToCategorical(nb_classes = nb_classes)(y_input)
    x_path, y_path = AffineDeformation(x_input, translation_window_width=(0, 0, 0),
                                       rotation_window_width=(2, 0, 0), scaling_window_width = (0.05,0.05,0.05),cval = [0,[1,0,0,0]])(x_input, y_path)
    x_path, y_path = GridCrop(x_path, (192,192,48), n=1, nonzero=True)(x_path, y_path)  # x_path_0 is used as a reference volume to determine the coordinate around which to crop (here also constrained to nonzero flair voxels)
    x_path = GaussianNoise(std=0.1)(x_path)
    x_train = keras_model_transformer(x_path) # x_train is the predicted output
    y_train = y_path

    # used for validation and is on the level of patches
    y_path = ToCategorical(nb_classes=nb_classes)(y_input)
    x_path, y_path = GridCrop(x_input, (192,192,48), nonzero=True)(x_input, y_path)  # notice that there is no n specified --> this will sample the complete grid
    x_val = keras_model_transformer(x_path)
    y_val = y_path

    # used for validation of the full images and thus is on the level of the input
    x_path = x_val
    x_path = Buffer()(x_path)
    x_full_val = Put(x_input)(x_path)  # x_val is on the patch level and the put transformers brings the patch back to the reference space; have a look why y_input is used (with n=None) and think about why this is
    y_full_val = ToCategorical(nb_classes = nb_classes)(y_input)

    # you can use Creator.summary() method to visualize your designed architecture
    # when constructing your pathway, you can also name your transformers. If you don't do this, the creator will name the transformers.
    # ff you want to have unique transformer names for your entire network, you can first make one large creator and afterwards make the individual creators
    # This step is not necessary however. When you make a DvnModel later there will be built one large creator inside and thus names are given automagically.
    creator = Creator([x_train, y_train, x_val, y_val, x_full_val, y_full_val]) # -> generates samples
    x_train, y_train, x_val, y_val, x_full_val, y_full_val = creator.outputs
    creator.summary()

    # we make a DvnModel, which allows to give [x], [x, y] or [x, y, sample_weight] as "outputs". Here, the x's must be after the keras_model_transformer, thus referring to the predicted y.
    # what is a DvnModel? Similar to a keras model, but including the processing pipeline. If you inspect the DvnModel code you'll see it has fit, evaluate and predict methods
    # to apply a method, one needs to choose which configuration (keys; see below)
    dvn_model = DvnModel(
        outputs={
            "train": [x_train, y_train],
            "val": [x_val, y_val],
            "full_val": [x_full_val, y_full_val],
            "full_test": [x_full_val]
        }
    )

    # similar to keras, we can compile the model
    # here lists (of lists) can be used to apply different losses/metrics to different outputs of the network
    # the losses can also be lists of lists if you want a linear combination of losses to be applied to a certain output (when no weights specified, everything is weighted uniformly)
    soft_dice = get_loss("dice_loss",reduce_along_batch=reduce_along_batch,reduce_along_features=reduce_along_features,weights = [0,1,1,1],eps=eps_estimate)
    dice_score = get_metric("dice_coefficient")
    dvn_model.compile("train", optimizer=Adam(lr=1e-3), losses=[[soft_dice]],metrics=[[dice_score]])
    # although the following outputs are not used for fitting your model, you must compile it as well to know what metrics to calculate
    dvn_model.compile("val", losses=[[soft_dice]], metrics=[[dice_score]])
    dvn_model.compile("full_val", losses=[[soft_dice]], metrics=[[dice_score]])

    # typically one needs to organize everything on their local disks, e.g. a dir to save (intermediate) models, a logs directory to save intermediate log files to view via e.g. Tensorboad, some output dirs to save (intermediate) predictions, etc.
    # we have provided one way of doing so under deepvoxnet2/factories/directory_structure based on the samplers you created
    output_structure = MircStructure(
        base_dir=base_dir,
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=None,  # when None a new round will be created
        training_mirc=train_data,
        validation_mirc=val_data
    )
    output_structure.create()  # only now the non-existing output dirs are created

    callbacks = [
        DvnModelEvaluator(dvn_model, "full_val", val_sampler, freq=np.int(30*adapt_epochs), output_dirs=output_structure.val_images_output_dirs),  # watch out, here you do need to make sure that the order of the val_sampler (shuffle=False by default) is te same as the order of your output_dirs...
        DvnModelCheckpoint(dvn_model, output_structure.models_dir, freq=np.int(30*adapt_epochs)),  # every 30 epochs the model will be saved (for e.g. parallel offline use for testing some things)
        ReduceLROnPlateau(monitor="val__loss",factor=0.2,patience=np.int(40*adapt_epochs),mode="min",min_delta=0.0001),
        EarlyStopping(monitor="val__loss",patience=np.int(80*adapt_epochs),mode="min",min_delta=0.0001)
    ]

    # let's train :-)
    history = dvn_model.fit("train", train_sampler, batch_size=2, validation_key="val", validation_sampler=val_sampler,
                            callbacks=callbacks, epochs=1000,
                            logs_dir=output_structure.logs_dir)  # ideally choose the batch size as a whole multiple of the number of samples your processing pipeline produces
    with open(output_structure.history_path, "wb") as f:
        pickle.dump(history.history, f)

    dvn_model.save(os.path.join(output_structure.models_dir, "dvn_model_final"))

if __name__ == '__main__':
    for f in range(0,5):
        train('Trained_models','full_image_eps_DSC', fold_i=f,
              base_dir=main_dir, reduce_along_batch=False)
        tf.keras.backend.clear_session()
