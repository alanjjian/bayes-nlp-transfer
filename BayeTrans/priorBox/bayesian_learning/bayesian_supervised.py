"""The main supervised class for running bayesian deep learning model (including ray tune)"""

import os
from .bayesian_base import _bayesianBase
from .train_supervised import training_function
from .run_tune import run_ray

from datasets import load_from_disk
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)

# from ..solo_learn.utils.classification_dataloader import prepare_data



def prepare_data(train_dataset=None,
            val_dataset=None,
            data_dir=None,
            train_dir=None,
            val_dir=None,
            batch_size=None,
            num_workers=None):
    """
    Rewrite of ..solo_learn.utils.classification_dataloader.prepare_data
    
    Handles data-loading of AGNews data, and also returns the amount of data you have to train on,
    and the number of batches in your training process
    """
    # Load data
    dataset = load_from_disk(data_dir).with_format('torch')
    
    # Create eval data
    split_train_dataset = dataset['train'].train_test_split(test_size=0.1)
    dataset['train'] = split_train_dataset['train']
    dataset['eval'] = split_train_dataset['test']
    
    # Use tokenizer to generate and process data
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_agnews = dataset.map(preprocess_function, remove_columns=['text'], batched=True)
    
    # Create collator with dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(tokenized_agnews['train'], collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(tokenized_agnews['eval'], collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(tokenized_agnews['test'], collate_fn=data_collator, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, len(dataset['train']), len(train_dataloader)
    
class bayesianSupervised(_bayesianBase):

    def __init__(self, args):
        super(bayesianSupervised, self).__init__(args)


    def learn(self):
        config = self.get_config(self.args)
        """train_loader, test_loader, config['N'], config['num_of_batches'] = prepare_data(
            # train_dataset=config['train_dataset'],
            # val_dataset=config['val_dataset'],
            data_dir=config['data_dir'],
            # train_dir=config['train_dir'],
            # val_dir=config['val_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
        )"""

        # N is the number of rows in your training data, num_of_batches is self-explanatory
        train_loader, val_loader, test_loader, config['N'], config['num_of_batches'] = prepare_data(
            # train_dataset=config['train_dataset'],
            # val_dataset=config['val_dataset'],
            data_dir=config['data_dir'],
            # train_dir=config['train_dir'],
            # val_dir=config['val_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
        )

        if self.args.use_tune:
            run_ray(self.args, config, train_loader, val_loader, test_loader)
        else:
            training_function(config, train_loader, val_loader, test_loader)




    def get_config(self, args):
        # Set wandb
        if not args.ignore_wandb:
            os.environ['WANDB_API_KEY'] = args.wandb_key
        config = vars(args)
        # Adjust the paths to the server
        if args.server_path:
            paths = ['local_dir', 'data_dir', 'weights_path', 'wandb_save_dir', 'samples_dir', 'prior_path']
            if args.g_run == 'greence':
                args.g_path = '/scratch/rs8020/'
            elif args.g_run == 'gauss':
                args.g_path = '/path/to/your/dir'
            elif args.g_path == 'local':
                args.g_path = '/data2/ssl_priors'
            else:
                args.g_path = '/path/to/your/dir'
            for path in paths:
                config[path] = args.g_path + config[path]

        print("These are my configs:")
        print(config)
        return config