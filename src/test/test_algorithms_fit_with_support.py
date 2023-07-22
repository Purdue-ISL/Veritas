#
import pytest
import os
import json
import torch
from typing import Sequence, Dict, Type
from veritas.frameworks import FrameworkFitHMMStream


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
DATA_DIR0 = os.path.join(DATA_ROOT, "Deploy06")


#
CAPMAX = {DATA_DIR0: "100.0"}
CAPFILTER = {DATA_DIR0: False}
CAPUNIT = {DATA_DIR0: "5.0"}


#
INITETA = {DATA_DIR0: 0.0}
TRANSETA = {DATA_DIR0: 1e-2}
VARETA = {DATA_DIR0: 1e-4}


def args_stream(directory: str, jit: bool, initial: str, transition: str, emission: str, /) -> Sequence[str]:
    R"""
    Arguments of video streaming HMM.

    Args
    ----
    - directory
        Directory.
    - jit
        Use JIT.
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
    args.extend(["--jit"] if jit else [])
    args.extend(["--initial", initial])
    args.extend(["--transition", transition])
    args.extend(["--emission", emission])
    args.extend(["--num-epochs", str(NUM_EPOCHS)])
    args.extend(["--capacity-max", CAPMAX[directory]])
    args.extend(["--filter-capmax"] if CAPFILTER[directory] else [])
    args.extend(["--capacity-unit", CAPUNIT[directory]])
    args.extend(["--capacity-min", "0.1"])
    args.extend(["--transition-unit", "5.0"])
    args.extend(["--initeta", str(INITETA[directory])])
    args.extend(["--transeta", str(TRANSETA[directory])])
    args.extend(["--vareta", str(VARETA[directory])])
    args.extend(["--varinit", str((float(CAPMAX[directory]) * 0.05) ** 2)])
    args.extend(["--varmax-head", str(2.0 * (float(CAPMAX[directory]) * 0.1) ** 2)])
    args.extend(["--varmax-rest", str((float(CAPMAX[directory]) * 0.1) ** 2)])
    args.extend(["--head-by-time", "1.0"])
    args.extend(["--head-by-chunk", "1"])
    args.extend(["--transextra", "5"])
    args.extend(["--include-beyond"])
    args.extend(["--smooth", "0.05"])
    args.extend(["--support", os.path.join("supports", "soft0.json")])
    return args


#
FRAMEWORKS: Dict[str, Type[FrameworkFitHMMStream]]


#
PARAMETERS = [
    ("stream", DATA_DIR0, False, "generic", "gaussian.sym", "v1"),
    ("stream", DATA_DIR0, True, "generic", "gaussian.asym", "v1"),
]
ARGS = {"stream": args_stream}
FRAMEWORKS = {"stream": FrameworkFitHMMStream}
STRICTS = {"stream": False}


@pytest.mark.parametrize(("kind", "directory", "jit", "initial", "transition", "emission"), PARAMETERS)
def test_fit(*, kind: str, directory: str, jit: bool, initial: str, transition: str, emission: str) -> None:
    R"""
    Test HMM learning.

    Args
    ----
    - kind
        Task kind
    - directory
        Directory of a dataset.
    - jit
        If True, use JIT in testing.
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
    framework = FRAMEWORKS[kind](disk=LOG_ROOT, clean=True, strict=STRICTS[kind])
    framework(ARGS[kind](directory, jit, initial, transition, emission))

    # Learning must be full batch
    assert len([i for (i, _) in enumerate(framework._loader_train)]) == 1
    assert len([i for (i, _) in enumerate(framework._loader_valid)]) == 1
    assert len([i for (i, _) in enumerate(framework._loader_test)]) == 1


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for (kind, directory, jit, initial, transition, emission) in PARAMETERS:
        #
        test_fit(kind=kind, directory=directory, jit=jit, initial=initial, transition=transition, emission=emission)


#
if __name__ == "__main__":
    #
    main()
