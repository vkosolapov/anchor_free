from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model.classification_model import ClassificationModel
from data.classification_data import ClassificationDataModule


def main():
    seed_everything(seed=42, workers=True)
    model = ClassificationModel()
    data = ClassificationDataModule()
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        deterministic=True,
        max_epochs=1000,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(
                monitor="accuracy/val",
                mode="max",
                save_top_k=1,
                auto_insert_metric_name=True,
                verbose=True,
            ),
            EarlyStopping(
                monitor="accuracy/val",
                mode="max",
                patience=10,
                verbose=True,
            ),
        ],
        logger=TensorBoardLogger(
            save_dir="logs", name="001_test", version="002_sub_test"
        ),
        fast_dev_run=False,
    )
    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    trainer.test(ckpt_path="best", datamodule=data)


if __name__ == "__main__":
    main()
