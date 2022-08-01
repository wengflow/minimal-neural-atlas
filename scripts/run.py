import sys
import os
import pathlib
import shutil
import subprocess
import argparse
import yaml
import easydict
import pytorch_lightning as pl

# insert the project / script parent directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PROJECT_DIR)
import minimal_neural_atlas as mna

STAGES = [ "train", "val", "test" ]
METRICS_FILENAME = "metrics.yaml"


def main(args):
    # load the config from the config file
    with open(args.config) as f:
        config = easydict.EasyDict(yaml.full_load(f))

    # obtain the git HEAD hash
    config.git_head_hash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR
    ).decode('ascii').strip()

    # seed all pseudo-random generators
    config.seed = pl.seed_everything(config.seed, workers=True)
    
    # instantiate the data module & model
    datamodule = mna.data.datamodule.DataModule(
        config.seed,
        config.input,
        config.target,
        config.trainer.num_nodes,
        config.trainer.gpus,
        **config.data
    )
    model = mna.models.minimal_neural_atlas.MinimalNeuralAtlas(
        config.git_head_hash,
        config.input,
        config.target,
        config.model.num_charts,
        config.model.train_uv_sample_size,
        config.model.eval_uv_presample_size,
        config.model.min_interior_ratio,
        config.model.prob_occ_threshold,
        config.model.checkpoint_filepath,
        config.model.encoder,
        config.model.cond_sdf,
        config.model.cond_homeomorphism,
        config.loss,
        config.metric,
        config.optimizer,
        config.lr_scheduler,
        config.data.uv_space_scale,
        config.data.pcl_normalization_scale,
        config.data.train_uv_max_sample_size,
        config.data.eval_uv_max_sample_size,
        config.data.train_target_pcl_nml_size,
        config.data.eval_target_pcl_nml_size
    )

    # instantiate the trainer & its components
    if getattr(config.trainer, "checkpoint_callback", True):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.checkpoint)
        callbacks = [ checkpoint_callback ]
    else:
        callbacks = None

    if getattr(config.trainer, "logger", True):
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            default_hp_metric=False, **config.logger
        )   
    else:
        logger = False
    if hasattr(config.trainer, "logger"):
        config.trainer.pop("logger")

    plugins = {
        None: None,
        "ddp_cpu": pl.plugins.DDPPlugin(find_unused_parameters=False),
        "ddp": pl.plugins.DDPPlugin(find_unused_parameters=False),
        "ddp_spawn": pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)
    }[config.trainer.accelerator]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        replace_sampler_ddp=True,
        sync_batchnorm=True,
        terminate_on_nan=True,
        multiple_trainloader_mode="min_size",
        **config.trainer
    )

    # save the config to the log directory, if logging is done & not resuming
    # training from a checkpoint
    if (
        logger is not False
        and getattr(config.trainer, "resume_from_checkpoint", None) is None
    ):
        pathlib.Path(trainer.logger.log_dir).mkdir(parents=True)
        shutil.copy2(args.config, trainer.logger.log_dir)

    # train, validate or test the model
    if args.stage == "train":
        trainer.fit(model, datamodule)
    elif args.stage == "val":
        metrics = trainer.validate(model, datamodule)
    elif args.stage == "test":
        metrics = trainer.test(model, datamodule)

    # save the validation / test metrics to the log dir, if logging is done
    if args.stage != "train" and logger is not False:
        metrics_filepath = os.path.join(
            trainer.logger.log_dir, METRICS_FILENAME
        )
        with open(metrics_filepath, 'w') as f:
            yaml.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training, validation & testing script of Neural Atlas"
    )
    parser.add_argument(
        "stage", type=str, choices=STAGES,
        help="Train, validation or test mode."
    )
    parser.add_argument(
        "config", type=str, help="Path to a configuration file in yaml format."
    )
    args = parser.parse_args()

    main(args)
