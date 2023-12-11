"""SSL prior class"""
import os
import sys


### TODO This should be fixed
# sys.path.insert(0, "../")
###
from pprint import pprint

import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from .prior_base import _PriorBase
from .prior_ssl_utils import load_weights2
from ..swag.swag_callback import StochasticWeightAveraging
from ..solo_learn.methods import METHODS
from ..solo_learn.utils.checkpointer import Checkpointer
from ..solo_learn.utils.classification_dataloader import prepare_data as prepare_data_classification
from ..solo_learn.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)

import datasets
from datasets import load_dataset
from datasets import load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    XLNetConfig,
    XLNetLMHeadModel,
)

from torch.utils.data import DataLoader


class XLNet_pl(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = XLNetLMHeadModel.from_pretrained("xlnet-base-cased")
        self.encoder = self.model.base_model
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs  # delete when `use_cache` is removed in XLNetModel
    ):
        return self.model(input_ids,
        attention_mask,
        mems,
        perm_mask,
        target_mapping,
        token_type_ids,
        input_mask,
        head_mask,
        inputs_embeds,
        labels,
        use_mems,
        output_attentions,
        output_hidden_states,
        return_dict,
        **kwargs)
    
    def training_step(self, batch, batch_idx):
        xlnetoutput_obj = self.forward(input_ids=batch['input_ids'], 
                                       perm_mask=batch['perm_mask'], 
                                       target_mapping=batch['target_mapping'],
                                       labels=batch['labels'])
        loss = xlnetoutput_obj.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        xlnetoutput_obj = self.forward(input_ids=batch['input_ids'], 
                                       perm_mask=batch['perm_mask'], 
                                       target_mapping=batch['target_mapping'],
                                       labels=batch['labels'])
        loss = xlnetoutput_obj.loss
        return loss
    
    def train_dataloader(self):
        train_data = load_from_disk("~/projects/bayes-nlp-transfer/bookcorpus_train.hf").with_format("torch")
        
        tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
        collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=1/6,
            max_span_length=5)
        
        train_loader = DataLoader(train_data, collate_fn=collator, batch_size=4, num_workers=8)
        return train_loader
    
    def val_dataloader(self):
        val_data = load_from_disk("~/projects/bayes-nlp-transfer/bookcorpus_val.hf").with_format("torch")
        
        tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
        collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=1/6,
            max_span_length=5)
        
        val_loader = DataLoader(val_data, collate_fn=collator, batch_size=4, num_workers=8)
        return val_loader
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.encoder.parameters(), lr=4e-4)

class PriorSSL(_PriorBase):

    def __init__(self, args):
        super(PriorSSL, self).__init__(args)


    def learn_prior(self):
        # self.check_args(self.args)
        model = self.get_model(self.args)
        # train_loader, val_loader = self.contrastive_dataloader(self.args)
        train_loader, val_loader = self.plm_dataloader(self.args)
        self.posterior_training(self.args, model, train_loader, val_loader)
        


    def check_args(self, args):
        assert args.method in METHODS, f"Choose from {METHODS.keys()}"
        print("MY METHOD IS", args.method)
        
        """
        if args.num_large_crops and args.num_large_crops != 2:
            assert args.method == "wmse"
        """

    def get_model(self, args):
        """
        MethodClass = METHODS[args.method]
        model = MethodClass(**args.__dict__)
        if args.load_weights:
            load_weights2(model, path=args.path, eval_model=False)
        """
        
        model = XLNet_pl()
        
        return model
    
    def plm_dataloader(self, args):
        
        train_data = load_from_disk("~/projects/bayes-nlp-transfer/bookcorpus_train.hf").with_format("torch")
        val_data = load_from_disk("~/projects/bayes-nlp-transfer/bookcorpus_val.hf").with_format("torch")
        
        tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
        collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=1/6,
            max_span_length=5)
        
        train_loader = DataLoader(train_data, collate_fn=collator, batch_size=8, num_workers=8)
        val_loader = DataLoader(val_data, collate_fn=collator, batch_size=8, num_workers=8)
        
        return train_loader, val_loader
    
    def contrastive_dataloader(self, args):
        if not args.dali:
            # asymmetric augmentations
            if args.unique_augs > 1:
                transform = [
                    prepare_transform(args.dataset, **kwargs) for kwargs in args.transform_kwargs
                ]
            else:
                transform = [prepare_transform(args.dataset, **args.transform_kwargs)]
            transform = prepare_n_crop_transform(transform, num_crops_per_aug=args.num_crops_per_aug)
            if args.debug_augmentations:
                print("Transforms:")
                pprint(transform)

            train_dataset = prepare_datasets(
                args.dataset,
                transform,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                no_labels=args.no_labels,
            )
            train_loader = prepare_dataloader(
                train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )

        # normal dataloader for when it is available
        if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
            val_loader = None
        elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
            val_loader = None
        else:
            _, val_loader, _, _ = prepare_data_classification(
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_dataset=args.dataset,
                val_dataset=args.dataset,

            )
        
        return train_loader, val_loader

    
    def posterior_training(self, args, model, train_loader, val_loader):
        callbacks = []
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        source_path = os.path.join(args.logdir_init, suffix)
        Path(source_path).mkdir(parents=True, exist_ok=True)
        args.source_path = source_path
        print('Args: \n', args)
        # wandb logging
        if args.wandb:
            os.environ['WANDB_API_KEY'] = args.wandb_key
            wandb_logger = WandbLogger(
                name=suffix,
                project=args.project,
                entity=args.entity,
                offline=args.offline,
                save_dir=source_path

            )
            #wandb_logger.watch(model, log="gradients", log_freq=100)
            wandb_logger.log_hyperparams(args)
            # lr logging
            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks.append(lr_monitor)
        
        if args.save_checkpoint:
            # save checkpoint on last epoch only
            ckpt = Checkpointer(
                args,
                logdir=source_path,
                per_step= True,
                frequency=args.checkpoint_frequency,
            )
            callbacks.append(ckpt)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        swa = StochasticWeightAveraging(swa_step_start=args.start_swag, device=device, swa_lrs=args.swa_lrs,
                                        annealing_epochs=args.annealing_epochs, source_path=source_path,
                                        interval_steps=args.interval_steps,
                                        scheduler = args.swag_scheduler, n_cycles = args.n_cycles,
                                        n_samples = args.n_samples
        )
        callbacks.append(swa)
        
        torch.cuda.empty_cache()
        
        trainer = Trainer.from_argparse_args(
            args,
            logger=wandb_logger,
            callbacks=callbacks,
            checkpoint_callback=False,
            terminate_on_nan=True,
            log_every_n_steps = 10,
            gpus=1,
            limit_train_batches = 40,
            limit_val_batches = 20,
        )
        trainer.fit(model) #, train_loader, val_loader)      