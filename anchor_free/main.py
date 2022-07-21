from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from model.detection_model import DetectionModel
from data.detection_data import DetectionDataModule
from consts import *


def main():
    seed_everything(seed=42, workers=True)
    model = DetectionModel()
    data = DetectionDataModule()
    trainer = Trainer(
        accelerator=TRAINER_ACCELERATOR,
        devices=TRAINER_DEVICES,
        precision=TRAINER_PRECISION,
        deterministic=False,
        max_epochs=TRAINER_MAX_EPOCHS,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(
                monitor=TRAINER_MONITOR,
                mode=TRAINER_MONITOR_MODE,
                save_top_k=1,
                auto_insert_metric_name=True,
                verbose=True,
            ),
            EarlyStopping(
                monitor=TRAINER_MONITOR,
                mode=TRAINER_MONITOR_MODE,
                patience=TRAINER_EARLY_STOPPING_PATIENCE,
                verbose=True,
            ),
        ],
        logger=WandbLogger(
            save_dir="logs",
            project=TRAINER_EXPERIMENT_NAME,
            name=TRAINER_EXPERIMENT_VERSION,
        ),
        fast_dev_run=TRAINER_FAST_DEV_RUN,
    )
    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    if not TRAINER_FAST_DEV_RUN:
        trainer.test(ckpt_path="best", datamodule=data)


if __name__ == "__main__":
    main()
