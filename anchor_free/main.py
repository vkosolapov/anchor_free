from pytorch_lightning.utilities.cli import LightningCLI

from data import Imagenette
from model import Model


def main():
    cli = LightningCLI(
        model_class=Model,
        datamodule_class=Imagenette,
        seed_everything_default=42,
        save_config_filename="001_test.json",
        save_config_overwrite=True,
        run=False,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
