#
import pytest
import os
import json
from typing import Sequence, Dict, Union, Type
from veritas.frameworks import FrameworkLoadHMMCategorical, FrameworkLoadHMMGaussian, FrameworkLoadHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]
LOG_ROOT = CONSTANTS["log_root"]


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
    args.extend(["--capacity-max", "10.0"])
    args.extend(["--filter-capmax"])
    return args


#
FRAMEWORKS: Dict[
    str,
    Union[Type[FrameworkLoadHMMCategorical], Type[FrameworkLoadHMMGaussian], Type[FrameworkLoadHMMStream]],
]


#
PARAMETERS = [
    ("categorical", DATA_DIR0),
    ("categorical", DATA_DIR1),
    ("gaussian", DATA_DIR2),
    ("gaussian", DATA_DIR3),
    ("stream", DATA_DIR4),
]
FRAMEWORKS = {
    "categorical": FrameworkLoadHMMCategorical,
    "gaussian": FrameworkLoadHMMGaussian,
    "stream": FrameworkLoadHMMStream,
}
ARGS = {"categorical": args_categorical, "gaussian": args_gaussian, "stream": args_stream}


@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_annotate(*, kind: str, directory: str) -> None:
    R"""
    Test dataset annotations.

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
    framework = FRAMEWORKS[kind](disk=LOG_ROOT, clean=True)
    framework.phase1(ARGS[kind](directory))
    framework.erase()

    #
    framework._dataset_full.__annotate__()


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
