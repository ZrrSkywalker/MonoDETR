"""
Entry point for MonoDETR training and evaluation.

Replaces train.sh / test.sh.  Usage:

    python main.py --mode train --config configs/monodetr.yaml
    python main.py --mode train --config configs/monodetr.yaml --cuda 2
    python main.py --mode train --config configs/monodetr.yaml --cuda 2,3

    python main.py --mode test --config configs/monodetr.yaml
    python main.py --mode test --config configs/monodetr.yaml --cuda 2

All stdout/stderr is automatically tee'd to logs/<timestamp>_<backbone>.log.
"""

import os
import sys
from enum import Enum
from typing import Annotated

import typer

_ROOT = os.path.dirname(os.path.abspath(__file__))


class Mode(str, Enum):
    train = "train"
    test = "test"


def main(
    mode: Annotated[Mode, typer.Option(help="'train' or 'test'")],
    config: Annotated[str, typer.Option(help="Path to YAML config file")],
    cuda: Annotated[str, typer.Option(help="CUDA device ID(s) to use, e.g. '0' or '2,3'")] = "0",
) -> None:
    """MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection."""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    # Remap to 0-indexed IDs as seen by PyTorch after CUDA_VISIBLE_DEVICES is applied
    n_gpus = len(cuda.split(","))
    gpu_ids = ",".join(str(i) for i in range(n_gpus))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from tools.train_val import run  # noqa: PLC0415
    run(config=config, evaluate_only=(mode == Mode.test), gpu_ids=gpu_ids)


if __name__ == "__main__":
    typer.run(main)
