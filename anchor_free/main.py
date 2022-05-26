from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import Model
from data import Imagenette


def main():
    seed_everything(seed=42, workers=True)
    model = Model()
    data = Imagenette()
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        # amp_backend="apex",
        precision=16,
        deterministic=True,
        auto_scale_batch_size="binsearch",
        max_epochs=1000,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(monitor="val_accuracy"),
            EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                min_delta=0.00,
                mode="max",
                verbose=False,
            ),
        ],
        logger=TensorBoardLogger(
            save_dir="logs", name="001_test", version="001_sub_test"
        ),
        track_grad_norm=2,
        fast_dev_run=False,
    )
    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    trainer.test(ckpt_path="best", datamodule=data)


if __name__ == "__main__":
    main()
