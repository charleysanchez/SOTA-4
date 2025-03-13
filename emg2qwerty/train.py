# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import types


import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform

print("Imports Done")
log = logging.getLogger(__name__)

class GradualUnfreezing(pl.Callback):
    """Gradually unfreeze layers during training."""
    def __init__(self, unfreeze_epochs=None):
        super().__init__()
        if unfreeze_epochs is None:
            unfreeze_epochs = {10: ['3.'], 20: ['2.'], 30: ['1.'], 40: ['0.']}
        self.unfreeze_epochs = unfreeze_epochs
        
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # Check if we need to unfreeze any layers at this epoch
        for epoch, layer_prefixes in self.unfreeze_epochs.items():
            if current_epoch == epoch:
                for name, param in pl_module.model.named_parameters():
                    if any(name.startswith(prefix) for prefix in layer_prefixes):
                        param.requires_grad = True
                        print(f"Epoch {current_epoch}: Unfrozen layer {name}")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)
   

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )

    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location="cpu")

        # Load the pretrained model state dict
        pretrained_state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Create a new state dict for the target model
        adapted_state_dict = {}
        
        # Current model state dict (for reference to get the correct shapes)
        current_state_dict = module.state_dict()
        
        for name, param in pretrained_state_dict.items():
            if name in current_state_dict:
                # Get the target parameter shape
                target_shape = current_state_dict[name].shape
                
                # Handle batch norm parameters (32 -> 16)
                if 'model.0.batch_norm' in name and len(param.shape) == 1:
                    if param.shape[0] == 32 and target_shape[0] == 16:
                        # Take first half of parameters (assuming this makes sense for your model)
                        adapted_state_dict[name] = param[:16]
                        print(f"Adapted {name}: {param.shape} -> {adapted_state_dict[name].shape}")
                
                # Handle MLP weights (528 -> 264 in input dimension)
                elif 'model.1.mlps' in name and 'weight' in name:
                    if param.shape[1] == 528 and target_shape[1] == 264:
                        # Take first half of the input dimensions
                        adapted_state_dict[name] = param[:, :264]
                        print(f"Adapted {name}: {param.shape} -> {adapted_state_dict[name].shape}")
                
                # For parameters with matching shapes, copy directly
                elif param.shape == target_shape:
                    adapted_state_dict[name] = param
                    print(f"Copied {name} directly: {param.shape}")
                
                # Skip parameters that don't match and aren't handled above
                else:
                    print(f"Skipped {name}: shape mismatch ({param.shape} vs {target_shape})")

        # Load the modified weights
        module.load_state_dict(adapted_state_dict, strict=False)
        
        # Freeze all layers by default
        for param in module.model.parameters():
            param.requires_grad = False

        # Get the original layer's parameters
        if hasattr(module.model, '4'):
            # If layer 4 is a single layer in current model
            original_layer = module.model[4]
            in_features = original_layer.in_features
            out_features = original_layer.out_features

        # Create a new Sequential module with dropout
        module.model[4] = nn.Linear(in_features, 7)

        # Define layer groups with their learning rates
        layer_groups = {
            'output': ['4.'],      # Output layer (starts unfrozen)
            'lstm': ['3.'],        # LSTM layers (unfreeze later)
            'features_mid': ['1.'],   # Mid-level features
            'features_low': ['0.']    # Low-level features
        }

        # Learning rates for each group (when unfrozen)
        group_learning_rates = {
            'output': 1e-3,
            'lstm': 2e-4,
            'features_mid': 1e-4,
            'features_low': 5e-5
        }

       # Initially, only unfreeze output layer
        param_groups_info = []
        for group_name, prefixes in layer_groups.items():
            group_params = {'params': [], 'lr': group_learning_rates[group_name]}
            for name, param in module.model.named_parameters():
                if any(name.startswith(prefix) for prefix in prefixes):
                    param.requires_grad = True if group_name == 'output' else False
                    group_params['params'].append(param)
                    if group_name == 'output':
                        print(f"Initially unfrozen {group_name} layer: {name}")
            if group_params['params']:
                param_groups_info.append(group_params)

        # Store the param groups in the module
        module.param_groups = param_groups_info

        # Set up base optimizer config (without parameters)
        config.optimizer = {
            "_target_": "torch.optim.AdamW",
            "weight_decay": 0.1,  # L2 regularization
            "betas": (0.9, 0.999),
            "eps": 1e-8
        }

        # Learning rate scheduler configuration
        steps_per_epoch = 29
        total_steps = steps_per_epoch * config.trainer.max_epochs

        # Get max learning rates for groups that have parameters
        max_lrs = []
        for group in ['output', 'lstm', 'features_mid', 'features_low']:
            for param_group in param_groups_info:
                if param_group['lr'] == group_learning_rates[group] and param_group['params']:
                    max_lrs.append(float(group_learning_rates[group]))
                    break

        config.lr_scheduler = {
            "_target_": "torch.optim.lr_scheduler.OneCycleLR",
            "max_lr": max_lrs,  # Explicitly created list of floats
            "pct_start": 0.3,  # Warm up for 30% of training
            "total_steps": int(total_steps),  # Ensure integer type
            "anneal_strategy": "cos",
            "div_factor": 25.0,  # Initial lr = max_lr/25
            "final_div_factor": 10000  # Final lr = initial_lr/1e4
        }

        # Add a configure_optimizers method to the module if it doesn't exist
        if not hasattr(module, 'configure_optimizers_orig'):
            module.configure_optimizers_orig = module.configure_optimizers

        def new_configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.param_groups,
                weight_decay=float(config.optimizer.weight_decay),
                betas=tuple(float(x) for x in config.optimizer.betas),
                eps=float(config.optimizer.eps)
            )
            
            
            max_lr_list = [float(lr) for lr in config.lr_scheduler.max_lr]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=max_lr_list,
                pct_start=float(config.lr_scheduler.pct_start),
                total_steps=int(config.lr_scheduler.total_steps),
                anneal_strategy=str(config.lr_scheduler.anneal_strategy),
                div_factor=float(config.lr_scheduler.div_factor),
                final_div_factor=float(config.lr_scheduler.final_div_factor)
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        
        module.configure_optimizers = types.MethodType(new_configure_optimizers, module)

        # Add callbacks for monitoring
        if "callbacks" not in config:
            config["callbacks"] = []

        # Learning rate monitoring
        lr_monitor = {
            "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
            "logging_interval": "step"
        }
        config.callbacks.append(lr_monitor)

        # Add gradual unfreezing callback as a config dict
        gradual_unfreeze_config = {
            "_target_": "emg2qwerty.train.GradualUnfreezing",
            "unfreeze_epochs": {
                50: ['3.'],           # Unfreeze LSTM at epoch 10
                100: ['1.'],           # Unfreeze mid-level features at epoch 30
                150: ['0.']            # Unfreeze low-level features at epoch 40
            }
        }
        config.callbacks.append(OmegaConf.create(gradual_unfreeze_config))

        # Early stopping
        early_stopping = {
            "_target_": "pytorch_lightning.callbacks.EarlyStopping",
            "monitor": "val/loss",
            "patience": 30,
            "mode": "min",
            "min_delta": 1e-4
        }
        config.callbacks.append(early_stopping)

        # Model checkpointing
        checkpoint_callback = {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": "val/loss",
            "save_top_k": 3,
            "mode": "min",
            "filename": "epoch={epoch}-val_loss={val/loss:.4f}"
        }
        config.callbacks.append(checkpoint_callback)

        # Log parameter counts
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        log.info(f"Total parameters: {total_params:,}")
        log.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")


    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
        )

    log_file = "training.log"
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Keep logging to console as well
    ]
)
    
    

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    log.info(f"Validation Metrics: {val_metrics}")


    test_metrics = trainer.test(module, datamodule)
    log.info(f"Test Metrics: {test_metrics}")


    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    }
    pprint.pprint(results, sort_dicts=False)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
