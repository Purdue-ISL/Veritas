#
import pytest
import os
import json
import torch
from typing import Sequence, Dict, Union, Type
from veritas.frameworks import FrameworkFitHMMCategorical, FrameworkFitHMMGaussian, FrameworkFitHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]
LOG_ROOT = CONSTANTS["log_root"]


#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#
DATA_DIR0 = os.path.join(DATA_ROOT, "HMMNaiveCategorical")
DATA_DIR1 = os.path.join(DATA_ROOT, "HMMPriorCategorical")
DATA_DIR2 = os.path.join(DATA_ROOT, "HMMNaiveGaussian")
DATA_DIR3 = os.path.join(DATA_ROOT, "HMMPriorGaussian")
DATA_DIR4 = os.path.join(DATA_ROOT, "Stream2")


def args_categorical(directory: str, /) -> Sequence[str]:
    R"""
    Arguments of categorical HMM.

    Args
    ----
    - directory
        Directory.

    Returns
    -------
    - args
        Arguments.
    """
    #
    args = []
    args.extend(["--dataset", directory])
    args.extend(["--train", "7"])
    args.extend(["--valid", "1"])
    args.extend(["--test", "2"])
    args.extend(["--device", DEVICE])
    args.extend(["--initial", "generic"])
    args.extend(["--transition", "generic"])
    args.extend(["--emission", "categorical"])
    args.extend(["--num-epochs", "0"])
    return args


def args_gaussian(directory: str, /) -> Sequence[str]:
    R"""
    Arguments of Gaussian HMM.

    Args
    ----
    - directory
        Directory.

    Returns
    -------
    - args
        Arguments.
    """
    #
    args = []
    args.extend(["--dataset", directory])
    args.extend(["--train", "7"])
    args.extend(["--valid", "1"])
    args.extend(["--test", "2"])
    args.extend(["--device", DEVICE])
    args.extend(["--initial", "generic"])
    args.extend(["--transition", "generic"])
    args.extend(["--emission", "gaussian"])
    args.extend(["--num-epochs", "0"])
    return args


def args_stream(directory: str, /) -> Sequence[str]:
    R"""
    Arguments of video streaming HMM.

    Args
    ----
    - directory
        Directory.

    Returns
    -------
    - args
        Arguments.
    """
    #
    args = []
    args.extend(["--dataset", directory])
    args.extend(["--train", os.path.join(directory, "train0.json")])
    args.extend(["--valid", os.path.join(directory, "valid0.json")])
    args.extend(["--test", os.path.join(directory, "test.json")])
    args.extend(["--device", DEVICE])
    args.extend(["--initial", "generic"])
    args.extend(["--transition", "generic"])
    args.extend(["--emission", "v0"])
    args.extend(["--num-epochs", "0"])
    args.extend(["--capacity-max", "10.0"])
    args.extend(["--filter-capmax"])
    args.extend(["--capacity-unit", "0.5"])
    args.extend(["--transition-unit", "5.0"])
    args.extend(["--initeta", "1.0"])
    args.extend(["--transeta", "1.0"])
    args.extend(["--vareta", "1.0"])
    args.extend(["--varinit", "0.25"])
    args.extend(["--varmax-head", "2.0"])
    args.extend(["--varmax-rest", "1.0"])
    args.extend(["--head-by-time", "1.0"])
    args.extend(["--head-by-chunk", "1"])
    args.extend(["--smooth", "0.05"])
    return args


#
FRAMEWORKS: Dict[
    str,
    Union[Type[FrameworkFitHMMCategorical], Type[FrameworkFitHMMGaussian], Type[FrameworkFitHMMStream]],
]


#
PARAMETERS = [
    ("categorical", DATA_DIR0),
    ("categorical", DATA_DIR1),
    ("gaussian", DATA_DIR2),
    ("gaussian", DATA_DIR3),
    ("stream", DATA_DIR4),
]
ARGS = {"categorical": args_categorical, "gaussian": args_gaussian, "stream": args_stream}
FRAMEWORKS = {
    "categorical": FrameworkFitHMMCategorical,
    "gaussian": FrameworkFitHMMGaussian,
    "stream": FrameworkFitHMMStream,
}


@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_annotate(*, kind: str, directory: str) -> None:
    R"""
    Test categorical HMM annotation.

    Args
    ----
    - kind
        Task kind.
    - directory
        Directory of a dataset.

    Returns
    -------
    """
    #
    framework = FRAMEWORKS[kind](disk=LOG_ROOT, clean=True, strict=False)
    framework.phase3(ARGS[kind](directory))
    framework.erase()

    #
    framework._algorithm.__annotate__()


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for (kind, directory) in PARAMETERS:
        #
        test_annotate(kind=kind, directory=directory)


#
if __name__ == "__main__":
    #
    main()
