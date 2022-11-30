#!/usr/bin/env python
# main.py
# Base script to invoke for everything

import os, sys
import argparse, argcomplete
import logging
from omegaconf import OmegaConf

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from modules.GraphBinsLM import GraphBinsLM

import misc_utils


def main(args):
    logger = logging.getLogger(__name__)

    # Define model (pytorch lightning module that contains model definition)
    model = GraphBinsLM(args)

    if args.basic.get("from_checkpoint") is not None:
        logger.info(f"Loading from checkpoint: {args.basic.from_checkpoint}")
        model = model.load_from_checkpoint(checkpoint_path=args.basic.from_checkpoint, args=args)

    callbacks_list = []

    # For monitoring the learning rate as determined by the scheduler
    callbacks_list.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    
    if "gradient_clip_val" in args.optimizer:
        gradient_clip_val = args.optimizer.gradient_clip_val
    else:
        gradient_clip_val = 0

    # If set, remove 1-cycle LR scheduler and use Stochastic Weight Averaging.
    if "use_swa" in args.optimizer and args.optimizer.use_swa:
        logger.info("Using StochasticWeightAveraging")
        callbacks_list.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2))
    
    tb_writer = pl.loggers.TensorBoardLogger(
        save_dir=args.paths.run_dir,
        name=args.basic.name,
        # log_graph=True,   # When using yolo, this won't work due to data-dependent control flow.
        default_hp_metric=False,
        max_queue=1,
        )

    if args.get("validate"):
        logger.info("==== RUNNING VALIDATION ====")
        assert args.basic.get("val_checkpoint") is not None, "Error: no validation checkpoint set."
        logger.info(f"Checkpoint file used: {args.basic.val_checkpoint}")
        
        args.basic.batch_size = 1   # Override batch size
        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False, static_graph=True),
            devices=1,
            # logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False
        )
        
        val_output = trainer.validate(
            model=model,
            verbose=True,
            ckpt_path=args.basic.val_checkpoint,
        )
        
        with open(os.path.join(args.val_output_dir, "validation_output.txt"), 'w') as f:
            f.write(args.basic.name)
            f.write(str(val_output))
            log_str = f"\nabs_rel, sq_rel, rms, rmsl, log10, d1, d2, d3:  \n{val_output[0]['metrics/abs_rel']}, {val_output[0]['metrics/sq_rel']}, {val_output[0]['metrics/rmse']}, {val_output[0]['metrics/rmse_log']}, {val_output[0]['metrics/log10']}, {val_output[0]['metrics/acc_1']}, {val_output[0]['metrics/acc_2']}, {val_output[0]['metrics/acc_3']}  \n ==#==  \nabs_rel_ra, sq_rel_ra, rms_ra, rmsl_ra, log10_ra, d1_ra, d2_ra, d3_ra:  \n{val_output[0]['metrics_ra/abs_rel_ra']}, {val_output[0]['metrics_ra/sq_rel_ra']}, {val_output[0]['metrics_ra/rmse_ra']}, {val_output[0]['metrics_ra/rmse_log_ra']}, {val_output[0]['metrics_ra/log10_ra']}, {val_output[0]['metrics_ra/acc_1_ra']}, {val_output[0]['metrics_ra/acc_2_ra']}, {val_output[0]['metrics_ra/acc_3_ra']}"
            f.write(log_str)

        print(str(val_output))
        print(log_str)
    
    elif args.get("inference"):
        logger.info("==== RUNNING INFERENCE ====")
        assert args.basic.get("val_checkpoint") is not None, "Error: no validation checkpoint set."
        logger.info(f"Checkpoint file used: {args.basic.val_checkpoint}")
        
        args.basic.batch_size = 1   # Override batch size.
        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            limit_predict_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            devices=1,
            logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False
        )
        
        predictions = trainer.predict(
            model=model,
            ckpt_path=args.basic.val_checkpoint,
        )
        logger.info(f"Done, results and metrics saved to {args.predict_output_dir}")
    else:
        # Define pytorch lightning trainer
        # Checkpointing behaviours
        callbacks_list.append(pl.callbacks.ModelCheckpoint(monitor="metrics/abs_rel", save_last=True, save_top_k=1, mode="min"))

        trainer = pl.Trainer(
            limit_train_batches=1 if args.debug else None,
            limit_val_batches=1 if args.debug else None,
            max_epochs=1 if args.debug else args.basic.max_epochs,
            check_val_every_n_epoch=args.basic.validate_every,
            gradient_clip_val=gradient_clip_val,
            accelerator="auto",
            strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True, static_graph=False),
            # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False, static_graph=True),
            devices=args.devices,
            logger=tb_writer,
            log_every_n_steps=1 if args.debug else 50,
            callbacks=callbacks_list,
            enable_model_summary=False
        )
        trainer.fit(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_file", required=True, type=argparse.FileType('r', encoding='UTF-8'), help="Path to the config/params YAML file.")
    parser.add_argument("-v", "--validate", action="store_true", help="""
        Runs validation using the latest available checkpoint that shares a name with the params file, unless the checkpoint file is specified by the
        params file in args.basic.val_checkpoint, in which case that checkpoint is evaluated instead. Uses only one device and a batch size of 1.
        Can also be used on the hparams.yaml automatically saved in each experiment's run directory (as args.basic.name will be present in this file).
        """)
    parser.add_argument("-i", "--inference", action="store_true", help="Run inference (like validation but with bigger batches and no saved metrics file)")
    parser.add_argument("--debug", action="store_true", help="""
        Sets debug mode. Force single-device training with no spawned dataloader workers, to allow breakpoints to work.
        Also forces maximum 50 training batches, for speed of debugging.
        """)
    parser.add_argument("--log_debug", action="store_true", help="""
        If set sets log level to logging.DEBUG. Separate from --debug because sometimes debug output isn't helpful.
        """)
    
    argcomplete.autocomplete(parser)
    cl_args = parser.parse_args()

    # Parse args
    args = OmegaConf.load(cl_args.config_file)
    if "args" in args:
        args = args.args    # This is to allow loading of auto-saved hparams.yaml files
    args.config_file = cl_args.config_file.name
    args.debug = cl_args.debug
    args.log_debug = cl_args.log_debug
    args.validate = cl_args.validate
    args.inference = cl_args.inference

    assert not (args.get("validate") and args.get("inference"))

    # Set up params for the debug mode (1 device, don't spawn extra workers in dataloader to let breakpoint() still work)
    if args.debug:
        logging.info("Debug mode active (--debug)")
    args.devices = 1 if args.debug or args.validate or args.inference else None
    args.hardware.num_workers = 0 if args.debug else args.hardware.num_workers

    # Handle overrides and defaults, do some checking
    args = misc_utils.check_and_validate_args(args)

    logging.basicConfig(level=logging.DEBUG if args.log_debug else logging.INFO, force=True, format="[%(levelname)s][%(name)s] %(message)s")
    logging.info("Starting")
    logging.debug("Debug log active")
    logging.debug(args)

    pl.seed_everything(42, workers=True)

    main(args)