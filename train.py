import argparse
import os

from cycle_gan.configs import Config
from cycle_gan.trainers import get_trainer_cls
from cycle_gan.utils.ddp import cleanup, setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)

    global_rank, local_rank, world_size = setup()

    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train, cfg.wandb, cfg.model, global_rank, local_rank, world_size
    )

    trainer.train()
    cleanup()
