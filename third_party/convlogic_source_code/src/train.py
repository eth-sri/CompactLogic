import os
import sys

# torch bundles an older libstdc++ that may lack GLIBCXX_3.4.30 needed by scipy.
# Re-exec with the system libstdc++ preloaded if not already done.
_SYSTEM_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if "LD_PRELOAD" not in os.environ and os.path.exists(_SYSTEM_LIBSTDCXX):
    os.environ["LD_PRELOAD"] = _SYSTEM_LIBSTDCXX
    os.execv(sys.executable, [sys.executable] + sys.argv)

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.profiler import ProfilerActivity, schedule

from convlogic.data import ConvLogicDataModule
from convlogic.model import ConvLogicModel
from utils.seed import set_seed

class PtModelCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".pt"

# Experiment with other options (medium, high, highest)
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # print full config
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.general.seed, reproducible=cfg.trainer.deterministic)

    dm = ConvLogicDataModule(
        dataset_name=cfg.model.dataset_name,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.data.num_workers,
        threshold_levels=cfg.data.threshold_levels,
        train_val_split=cfg.data.train_val_split,
        threshold_type=cfg.data.threshold_type,
    )

    # infer input channels
    cfg.model.input_channels = dm.input_channels
    model = ConvLogicModel(cfg.model)

    # Fallback to CSVLogger if wandb is not selected
    loggers = []
    if cfg.logging.wandb:
        loggers.append(WandbLogger(project="convlogic"))
    else:
        loggers.append(CSVLogger("logs", name="convlogic"))

    # Checkpoint callback
    checkpoint = PtModelCheckpoint(
        monitor="val_eval/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath="checkpoints/",
        filename="best-{epoch:02d}-{val_eval_acc:.4f}",
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint]
    if cfg.get("early_stopping", {}).get("use", False):
        early_stop = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            mode=cfg.early_stopping.mode,
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            verbose=True,
        )
        callbacks.append(early_stop)

    # Accuracy threshold stopping
    if cfg.get("accuracy_threshold_stop", {}).get("use", False):

        class AccuracyThresholdStop(pl.Callback):
            def on_validation_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                target_epoch = cfg.accuracy_threshold_stop.epoch
                threshold = cfg.accuracy_threshold_stop.threshold

                if epoch == target_epoch:
                    logs = trainer.callback_metrics
                    acc = logs.get(cfg.accuracy_threshold_stop.monitor)
                    if acc is not None and acc < threshold:
                        print(f"Early stop: accuracy {acc:.4f} < threshold {threshold} at epoch {epoch}")
                        trainer.should_stop = True

        callbacks.append(AccuracyThresholdStop())

    # Choose profiler
    profiler = None
    if cfg.general.get("profile", False):
        profiler_type = cfg.general.get("profile_type", "simple")
        if profiler_type == "simple":
            profiler = SimpleProfiler()
        elif profiler_type == "advanced":
            profiler = AdvancedProfiler(dirpath=".", filename="advanced_profiler.txt")
        elif profiler_type == "pytorch":
            profiler = PyTorchProfiler(
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_profiler"),
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=1,  # skip first step
                    warmup=1,  # warm up for 1 step
                    active=5,  # profile next 5 steps
                    repeat=1,  # only once
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

    # Trainer
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        profiler=profiler,
        **cfg.trainer,
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
