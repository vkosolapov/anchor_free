from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model.segmentation_model import SegmentationModel
from data.segmentation_data import SegmentationDataModule
from consts import *


def main():
    seed_everything(seed=42, workers=True)
    model = SegmentationModel()
    data = SegmentationDataModule()
    trainer = Trainer(
        accelerator=TRAINER_ACCELERATOR,
        devices=TRAINER_DEVICES,
        precision=TRAINER_PRECISION,
        deterministic=True,
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
        logger=TensorBoardLogger(
            save_dir="logs",
            name=TRAINER_EXPERIMENT_NAME,
            version=TRAINER_EXPERIMENT_VERSION,
        ),
        fast_dev_run=TRAINER_FAST_DEV_RUN,
    )
    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    if not TRAINER_FAST_DEV_RUN:
        trainer.test(ckpt_path="best", datamodule=data)


if __name__ == "__main__":
    main()
