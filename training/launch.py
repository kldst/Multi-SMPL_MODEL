# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse

# Make `import vggt` (repo root) and local modules importable regardless of how
# launch.py is invoked, so no PYTHONPATH env var is required.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../yian_vggt_smpl/training
_REPO_ROOT = os.path.dirname(_THIS_DIR)                   # .../yian_vggt_smpl
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


