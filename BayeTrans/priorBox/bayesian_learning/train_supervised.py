"""This file training a bayesian deep supervised learning"""
from typing import Dict

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch

from ..prior.prior_ssl_utils.data_utils import get_tests_final
from ..prior.prior_ssl_utils.utils import get_sample_dir, load_weights
from ..prior.prior_ssl_utils.cifar10_1 import load_cifar10_1
from ..prior.prior_ssl_utils.loggers import load_loggers_callbacks
from ..prior.prior_ssl_utils.utils import get_backbone
from ..sghmc.sghmc_model import SGLDModel
from ..sghmc.losses import GaussianPriorCELossShifted, CustomCEN, LaplaceApproxPrior
from ..sghmc.utils import load_prior, run_and_log_bma


"""
def train_main(config: dict, train_loader: DataLoader = None, test_loader: DataLoader = None,
               test_loaders: Dict = None):
    "Main function for training with the given loaders and config"
    seed_everything(config['seed'])
    # Path to store the samples
    samples_dir = get_sample_dir(config['samples_dir'])
    config['samples_dir'] = samples_dir
    if config['prior_type'] in ['shifted_gaussian', 'gaussian', 'normal']:
        prior_params = load_prior(config['prior_path'], config,
                                  number_of_samples_prior=config['number_of_samples_prior'])
        criterion = GaussianPriorCELossShifted(prior_params)
    elif config['prior_type'] == 'laplace':
        criterion = LaplaceApproxPrior(config['prior_path'], config)
    else:
        criterion = CustomCEN()
    config['criterion'] = criterion
    config['raw_params'] = config['prior_type'] != 'shifted_gaussian_diag'
    backbone = get_backbone(config)
    model = SGLDModel(backbone=backbone, **config)
    if config['prior_path'] is not None:
        load_weights(model, load_classifier=False, path=config['prior_path'])
    if config['weights_path_classifier'] is not None:
        load_weights(model, load_classifier=True, path=config['weights_path_classifier'])


    loggers, callbacks = load_loggers_callbacks(config['ignore_wandb'], config['use_mlflow'], config['use_tune'], config,
                                                model)
    trainer = pl.Trainer(gpus=config['gpus'], max_epochs=config['epochs'],
                         logger=loggers,
                         callbacks=callbacks,
                         resume_from_checkpoint=config['lightning_ckpt_path'],
                        #   strategy='dp',
                        #  accelerator='ddp',
                         # limit_train_batches=1,
                         # limit_val_batches=10,
                        #  num_sanity_val_steps=0
                         )
    trainer.fit(model, train_loader, test_loader)
    if config['run_bma']:
        run_and_log_bma(model, test_loaders, samples_dir, config, loggers)
"""

def train_main(config: dict, train_loader: DataLoader = None, val_loader: DataLoader = None,
               test_loader: Dict = None):
    """
    Main function for training with the given loaders and config. 
    Rewrite of original train_main with better naming scheme and ability to handle unified model
    from upstream task. 
    """
    seed_everything(config['seed'])
    # Path to store the samples
    samples_dir = get_sample_dir(config['samples_dir'])
    config['samples_dir'] = samples_dir

    # Grab prior parameters for use in model
    # Criterion used to define loss function
    if config['prior_type'] in ['shifted_gaussian', 'gaussian', 'normal']:
        prior_ssl_model = torch.load(config['prior_path'])
        print("SSL Model Loaded")
        prior_params = load_prior(config, prior_ssl_model, config['number_of_samples_prior'])
        print("Prior Loaded")
        criterion = GaussianPriorCELossShifted(prior_params)
        print("Loss Specified")
    elif config['prior_type'] == 'laplace':
        criterion = LaplaceApproxPrior(config['prior_path'], config)
    else:
        criterion = CustomCEN()

    # Send loss object back to config
    config['criterion'] = criterion
    config['raw_params'] = config['prior_type'] != 'shifted_gaussian_diag'

    # Grab Model structure
    if config['prior_path'] is not None:
        backbone = prior_ssl_model.base_model
    else:
        backbone = get_backbone(config)
    
    print("MODEL CONFIG:")
    print(config)
    model = SGLDModel(backbone=backbone, cyclic_lr=True, **config)
    print("Model specified")
    # weights_path_classifier is if you have pretrained weights for your classification layer located at some path
    if config['weights_path_classifier'] is not None:
        load_weights(model, load_classifier=True, path=config['weights_path_classifier'])


    loggers, callbacks = load_loggers_callbacks(config['ignore_wandb'], config['use_mlflow'], config['use_tune'], config,
                                                model)
    trainer = pl.Trainer(gpus=config['gpus'], max_epochs=config['epochs'],
                         logger=loggers,
                         callbacks=callbacks,
                         resume_from_checkpoint=config['lightning_ckpt_path'],
                        #   strategy='dp',
                        #  accelerator='ddp',
                        #  num_sanity_val_steps=0
                         )
    trainer.fit(model, train_loader, val_loader)
    if config['run_bma']:
        run_and_log_bma(model, [test_loader], samples_dir, config, loggers)


"""
def training_function(config: dict, train_loader: DataLoader, test_loader: DataLoader):
    "Train supervised learning with the given train and test loaders and config.
        Load cifar10.1 for evaluation"
        
    # todo - change this after fixing the bug of ray tune
    # cifar10_dataset = load_cifar10_1(root=config['data_dir'])
    # test_datasets = {'cifar10_1': cifar10_dataset}
    test_loaders = get_tests_final(test_loader, test_datasets, config['num_workers'], config['batch_size'])
    train_main(config, train_loader=train_loader, test_loader=test_loader, test_loaders=test_loaders)
"""

def training_function(config: dict, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
    """
    Rewrite of the training_function above. Instead of using CIFAR10.1 as the test set, we're 
    using a holdout set from our loaded data.
    """
    if config['just_eval']:
        bma_evaluation(config, config['lightning_ckpt_path'], test_loader)
    else:
        train_main(config, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def bma_evaluation(config: dict, ckpt_path: str, 
               test_loader: Dict = None):
    """
    Just runs evaluation for the Bayesian Model Average!
    """
    seed_everything(config['seed'])
    # Path to store the samples
    samples_dir = get_sample_dir(config['samples_dir'])
    config['samples_dir'] = samples_dir

    # Grab prior parameters for use in model
    # Criterion used to define loss function
    if config['prior_type'] in ['shifted_gaussian', 'gaussian', 'normal']:
        prior_ssl_model = torch.load(config['prior_path'])
        print("SSL Model Loaded")
        prior_params = load_prior(config, prior_ssl_model, config['number_of_samples_prior'])
        print("Prior Loaded")
        criterion = GaussianPriorCELossShifted(prior_params)
        print("Loss Specified")
    elif config['prior_type'] == 'laplace':
        criterion = LaplaceApproxPrior(config['prior_path'], config)
    else:
        criterion = CustomCEN()

    # Send loss object back to config
    config['criterion'] = criterion
    config['raw_params'] = config['prior_type'] != 'shifted_gaussian_diag'

    # Grab Model structure
    if config['prior_path'] is not None:
        backbone = prior_ssl_model.base_model
    else:
        backbone = get_backbone(config)
    
    model = SGLDModel(backbone=backbone, **config)
    print("Model specified")
    
    checkpoint = torch.load(ckpt_path)
    latest_state = checkpoint['state_dict']
    model.load_state_dict(latest_state)

    loggers, callbacks = load_loggers_callbacks(config['ignore_wandb'], config['use_mlflow'], config['use_tune'], config,
                                                model)
    
    run_and_log_bma(model, {'AG Test': test_loader}, samples_dir, config, loggers)