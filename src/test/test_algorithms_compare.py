#
import pytest
import os
import json
import torch
from typing import Dict, Union, Type
from veritas.frameworks import (
    FrameworkCompareHMMCategoricalBasic,
    FrameworkCompareHMMCategoricalStream,
    FrameworkCompareHMMGaussianBasic,
    FrameworkCompareHMMGaussianStream,
)


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]
LOG_ROOT = CONSTANTS["log_root"]
NUM_EPOCHS = CONSTANTS["num_epochs"]


#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#
DATA_DIR0 = os.path.join(DATA_ROOT, "HMMNaiveCategorical")
DATA_DIR1 = os.path.join(DATA_ROOT, "HMMPriorCategorical")
DATA_DIR2 = os.path.join(DATA_ROOT, "HMMNaiveGaussian")
DATA_DIR3 = os.path.join(DATA_ROOT, "HMMPriorGaussian")


#
FRAMEWORKS: Dict[
    str,
    Union[
        Type[FrameworkCompareHMMCategoricalBasic],
        Type[FrameworkCompareHMMCategoricalStream],
        Type[FrameworkCompareHMMGaussianBasic],
        Type[FrameworkCompareHMMGaussianStream],
    ],
]


#
PARAMETERS = [
    ("categorical.basic", DATA_DIR0, False),
    ("categorical.basic", DATA_DIR1, True),
    ("categorical.stream", DATA_DIR0, False),
    ("categorical.stream", DATA_DIR1, True),
    ("gaussian.basic", DATA_DIR2, False),
    ("gaussian.basic", DATA_DIR3, True),
    ("gaussian.stream", DATA_DIR2, False),
    ("gaussian.stream", DATA_DIR3, False),
]
FRAMEWORKS = {
    "categorical.basic": FrameworkCompareHMMCategoricalBasic,
    "categorical.stream": FrameworkCompareHMMCategoricalStream,
    "gaussian.basic": FrameworkCompareHMMGaussianBasic,
    "gaussian.stream": FrameworkCompareHMMGaussianStream,
}


@pytest.mark.parametrize(("kind", "directory", "jit"), PARAMETERS)
def test_correct(*, kind: str, directory: str, jit: bool) -> None:
    R"""
    Test basic HMM learning correctness comparing with convention HMM.

    Args
    ----
    - directory
        Directory of a dataset.
    - jit
        If True, use JIT in testing.

    Returns
    -------
    """
    #
    args = []
    args.extend(["--dataset", directory])
    args.extend(["--device", DEVICE])
    args.extend(["--jit"] if jit else [])
    args.extend(["--num-epochs", str(NUM_EPOCHS)])

    #
    framework = FRAMEWORKS[kind](disk=LOG_ROOT, clean=True)
    framework(args)


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for (kind, directory, jit) in PARAMETERS:
        #
        test_correct(kind=kind, directory=directory, jit=jit)


#
if __name__ == "__main__":
    #
    main()
