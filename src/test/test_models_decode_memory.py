#
import pytest
import os
import json
import torch
from typing import Sequence, Dict, Union, Type
from veritas.frameworks import FrameworkFitHMMCategorical, FrameworkFitHMMGaussian, FrameworkFitHMMStream
from veritas.models import ModelHMM
from veritas.types import THNUMS


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


def args_categorical(directory: str, initial: str, transition: str, emission: str, /) -> Sequence[str]:
    R"""
    Arguments of categorical HMM.

    Args
    ----
    - directory
        Directory.
    - initial
        Initial model name.
    - transition
        Transition model name.
    - emission
        Emission model name.

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
    args.extend(["--initial", initial])
    args.extend(["--transition", transition])
    args.extend(["--emission", emission])
    args.extend(["--num-epochs", "0"])
    return args


def args_gaussian(directory: str, initial: str, transition: str, emission: str, /) -> Sequence[str]:
    R"""
    Arguments of Gaussian HMM.

    Args
    ----
    - directory
        Directory.
    - initial
        Initial model name.
    - transition
        Transition model name.
    - emission
        Emission model name.

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
    args.extend(["--initial", initial])
    args.extend(["--transition", transition])
    args.extend(["--emission", emission])
    args.extend(["--num-epochs", "0"])
    return args


def args_stream(directory: str, initial: str, transition: str, emission: str, /) -> Sequence[str]:
    R"""
    Arguments of video streaming HMM.

    Args
    ----
    - directory
        Directory.
    - initial
        Initial model name.
    - transition
        Transition model name.
    - emission
        Emission model name.

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
    args.extend(["--initial", initial])
    args.extend(["--transition", transition])
    args.extend(["--emission", emission])
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
    args.extend(["--transextra", "5"])
    args.extend(["--include-beyond"])
    args.extend(["--smooth", "0.05"])
    return args


def decode_categorical(inputs: Sequence[THNUMS], outputs: Sequence[THNUMS], model: ModelHMM, /) -> None:
    R"""
    Decode categorical HMM inputs and outputs.

    Args
    ----
    - inputs
        Full inputs.
    - Outputs
        Full outputs.
    - model
        Model.

    Returns
    -------
    """
    #
    model.inputs(inputs)
    model.outputs(outputs)
    model.initial.inputs([])
    model.initial.outputs([outputs[0]])
    model.transition.inputs([])
    model.transition.outputs([outputs[1]])
    model.emission.inputs([inputs[0]])
    model.emission.outputs([outputs[2]])


def decode_gaussian(inputs: Sequence[THNUMS], outputs: Sequence[THNUMS], model: ModelHMM, /) -> None:
    R"""
    Decode Gaussian HMM inputs and outputs.

    Args
    ----
    - inputs
        Full inputs.
    - Outputs
        Full outputs.
    - model
        Model.

    Returns
    -------
    """
    #
    model.inputs(inputs)
    model.outputs(outputs)
    model.initial.inputs([])
    model.initial.outputs([outputs[0]])
    model.transition.inputs([])
    model.transition.outputs([outputs[1]])
    model.emission.inputs([inputs[0]])
    model.emission.outputs([outputs[2]])


def decode_stream(inputs: Sequence[THNUMS], outputs: Sequence[THNUMS], model: ModelHMM, /) -> None:
    R"""
    Decode video streaming HMM inputs and outputs.

    Args
    ----
    - inputs
        Full inputs.
    - Outputs
        Full outputs.
    - model
        Model.

    Returns
    -------
    """
    #
    model.inputs(inputs)
    model.outputs(outputs)
    model.initial.inputs([])
    model.initial.outputs([outputs[0]])
    model.transition.inputs([])
    model.transition.outputs([outputs[1]])
    model.emission.inputs([inputs[0]])
    model.emission.outputs([outputs[2], outputs[3]])


#
FRAMEWORKS: Dict[
    str,
    Union[Type[FrameworkFitHMMCategorical], Type[FrameworkFitHMMGaussian], Type[FrameworkFitHMMStream]],
]


#
PARAMETERS = [
    ("categorical", DATA_DIR0, "generic", "generic", "categorical"),
    ("categorical", DATA_DIR1, "generic", "generic", "categorical"),
    ("gaussian", DATA_DIR2, "generic", "generic", "gaussian"),
    ("gaussian", DATA_DIR3, "generic", "generic", "gaussian"),
    ("stream", DATA_DIR4, "generic", "generic", "v0"),
    ("stream", DATA_DIR4, "generic", "diag.sym", "v0"),
    ("stream", DATA_DIR4, "generic", "diag.asym", "v0"),
]
ARGS = {"categorical": args_categorical, "gaussian": args_gaussian, "stream": args_stream}
FRAMEWORKS = {
    "categorical": FrameworkFitHMMCategorical,
    "gaussian": FrameworkFitHMMGaussian,
    "stream": FrameworkFitHMMStream,
}
DECODES = {"categorical": decode_categorical, "gaussian": decode_gaussian, "stream": decode_stream}


@pytest.mark.parametrize(("kind", "directory", "initial", "transition", "emission"), PARAMETERS)
def test_decode(*, kind: str, directory: str, initial: str, transition: str, emission: str) -> None:
    R"""
    Test HMM model decoding.

    Args
    ----
    - kind
        Task kind.
    - directory
        Directory of a dataset.
    - initial
        Initial model name.
    - transition
        Transition model name.
    - emission
        Emission model name.

    Returns
    -------
    """

    #
    framework = FRAMEWORKS[kind](disk=LOG_ROOT, clean=True, strict=False)
    framework.phase3(ARGS[kind](directory, initial, transition, emission))
    framework.erase()

    #
    for loader in (framework._loader_train, framework._loader_valid, framework._loader_test):
        #
        (infos, *memory) = next(iter(loader))
        for (begins, ends) in zip(infos[0], infos[-1]):
            #
            inputs = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]
            outputs = framework._model.forward(inputs)
            DECODES[kind](inputs, outputs, framework._model)


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for (kind, directory, initial, transition, emission) in PARAMETERS:
        #
        test_decode(kind=kind, directory=directory, initial=initial, transition=transition, emission=emission)


#
if __name__ == "__main__":
    #
    main()
