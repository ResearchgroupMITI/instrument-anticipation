import os
import yaml
from datetime import datetime
from pathlib import Path
import configargparse
import wandb
from modules.utils.configargparse_arguments import build_configargparser
from modules.utils.utils import (
    get_class_by_path,
)
from modules.utils.seed import seed

wandb_mode_global = None
sdfsdf = 3

def train_sweep(hparams):
    '''Training routine for sweep'''
    #Create empty parameter dict
    sweep_config = {}

    #load parameters from normal config file into dicts
    parameters_dict = {}
    for key, value in vars(hparams).items():
        parameters_dict.update({
            key: {'value': value}
        })
    global wandb_mode_global
    wandb_mode_global = parameters_dict["wandb_mode"]["value"]

    #load sweep config and sweep parameters
    with open(hparams.sweep_config_path, 'r') as file:
        sweep_config_dict = yaml.safe_load(file)

        sweep_run_count = sweep_config_dict["sweep_run_count"]
        sweep_config = {
            'method': sweep_config_dict["method"]
            }
        metric = {
            'name': sweep_config_dict["name"],
            'goal': sweep_config_dict["goal"]
        }
        sweep_config['metric'] = metric

        for key, value in sweep_config_dict.items():
            if key not in ['method', 'name', 'goal', 'sweep_run_count']:
                assert isinstance(value, list)
                parameters_dict.update({
                    key: {'values': value}
                })

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, entity="sasha_or_ins_ant", project="TemporalModule_MSTCN_Sweeps")
    wandb.agent(sweep_id, train, count=sweep_run_count)


def train(hparams=None):
    """Main training routine

    Args:
        hparams: parameters for training
        ModuleClass (LightningModule): Contains training routine, etc.
        ModelClass: Contains network to train
        DatasetClass: Contains dataset on which to train
    """
    
    if hparams:
        wandb.init(entity="miti", config=hparams, project=hparams.wandbprojectname, name=hparams.name, mode=hparams.wandb_mode, dir=hparams.output_path)
    else:
        wandb.init(mode=wandb_mode_global)
        hparams = wandb.config
    seed(19, hparams)
    # LOAD MODULE
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    # LOAD MODEL
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    # LOAD DATASET
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)


    # load model
    model = ModelClass(hparams=hparams)
    # load dataset
    dataset = DatasetClass(hparams=hparams)
    # load module
    module = ModuleClass(hparams, model, dataset)

    if hparams.testmode is True:
        module.test()
    else: 
        module.trainval()
    wandb.finish()


def main():
    """Main"""
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # LOAD MODULE
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)

    # LOAD MODEL
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)

    # LOAD DATASET
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)

    # PRINT PARAMS & INIT LOGGER
    hparams = parser.parse_args()

    date_str = datetime.now().strftime("%y%m%d-%H%M%S")

    hparams.name = hparams.subproject_name + '_' + date_str #+ exp_name
    #hparams.output_path = Path(hparams.output_path).absolute() / hparams.name
    hparams.output_path = os.path.join(hparams.output_path, hparams.name)
    os.makedirs(hparams.output_path, exist_ok=True)

    if hparams.do_sweep:
        train_sweep(hparams)
    else:
        train(hparams)


if __name__ == "__main__":
    main()
