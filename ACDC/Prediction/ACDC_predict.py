import os
import sys

main_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/MICCAI_2022_scripts'

# sys.path.append('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/tmp/deepvoxnet2-main_21032022')
sys.path.append('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/data/Code/GitHub/deepvoxnet2')
sys.path.append('/usr/local/micapollo01/MIC/DATA/STAFF/stilbo0/tmp/pymirc-master')
sys.path.append(os.path.join(main_dir,'Training','functions'))
from create_dataset_ACDC_removeMyoES import create_dataset as create_dataset_ACDC_removeMyoES
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.factories.directory_structure import MircStructure

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def predict(run_name,experiment_name, fold_i=0, round_i=0,base_dir = None):
    test_data = Mirc()
    test_data.add(create_dataset_ACDC_removeMyoES(os.path.join(main_dir,'Preprocessed_images'),"val", fold_i=fold_i))
    test_sampler = MircSampler(test_data)
    output_structure = MircStructure(
        base_dir=base_dir,
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=round_i,  # when None a new round will be created
        validation_mirc = test_data
    )
    dvn_model = DvnModel.load_model(os.path.join(output_structure.models_dir, "dvn_model"))
    predictions = dvn_model.predict("full_val", test_sampler, output_dirs=output_structure.val_images_output_dirs)

if __name__ == '__main__':
    for f in range(0,5):#5):
        # predict('Trained_models', 'full_image_DSC', fold_i=f,
        #         base_dir=main_dir)

        predict('Trained_models', 'full_batch_DSC', fold_i=f,
                base_dir=main_dir)

        # predict('Trained_models', 'full_image_eps_DSC', fold_i=f,
        #         base_dir=main_dir)
        #
        # predict('Trained_models', 'partial_image_DSC', fold_i=f,
        #         base_dir=main_dir)
        #
        # predict('Trained_models', 'partial_batch_DSC', fold_i=f,
        #         base_dir=main_dir)
        #
        # predict('Trained_models','partial_image_eps_DSC', fold_i=f,
        #         base_dir=main_dir)
        #
        # predict('Trained_models', 'partial_leaf_DSC', fold_i=f,
        #         base_dir=main_dir)
        #
        # predict('Trained_models', 'partial_marginal_DSC', fold_i=f,
        #         base_dir=main_dir)