import lightning.pytorch as pl
import torch
import gc
from lightning.pytorch.loggers import WandbLogger

from pl_wrapper import STR_VAL_LOSS


def pl_wdb_train(
        dir_runtime_results, 
        patience: int, 
        max_epoch: int, 
        accumulate_grad_batches: int,
        device: str, 
        proj_name: str, 
        save_top_k: int, 
        datamodule: pl.LightningDataModule, 
        model_lit: pl.LightningModule, 
        module_to_log=None,
        log_every_n_batches=1,
        inference_mode=False,
    ):
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    logger = WandbLogger(project=proj_name, log_model=True, save_dir=dir_runtime_results)
    if module_to_log:
        logger.watch(module_to_log, log_freq=log_every_n_batches)

    early_stop = pl.callbacks.EarlyStopping(monitor=STR_VAL_LOSS, patience=patience)
    chkpt_min_val_loss = pl.callbacks.ModelCheckpoint(monitor=STR_VAL_LOSS, save_top_k=save_top_k)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        default_root_dir=dir_runtime_results,
        logger=logger,
        accelerator=device,
        max_epochs=max_epoch,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
        callbacks=[early_stop, lr_monitor, chkpt_min_val_loss],
        inference_mode=inference_mode,
        log_every_n_steps=10
    )

    trainer.fit(
        model_lit,
        datamodule
    )

    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()