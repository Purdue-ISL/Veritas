#
import pytest
import os
import json
import torch
from veritas.frameworks import FrameworkTransformHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]
LOG_ROOT = CONSTANTS["log_root"]
RESUME_ROOT = CONSTANTS["resume_root"]


#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#
DATA_DIR0 = os.path.join(DATA_ROOT, "Stream2")
RESUME_DIR0 = os.path.join(RESUME_ROOT, "fit", "Example0")
RESUME_DIR1 = os.path.join(RESUME_ROOT, "fit", "Example1")

#
PARAMETERS = [
    (DATA_DIR0, False, RESUME_DIR0, True, True, True),
    (DATA_DIR0, True, RESUME_DIR1, True, True, True),
    (DATA_DIR0, False, RESUME_DIR1, False, True, False),
    (DATA_DIR0, False, RESUME_DIR1, True, False, False),
    (DATA_DIR0, False, RESUME_DIR1, False, False, False),
]


@pytest.mark.parametrize(("directory", "jit", "resume", "disable0", "disable1", "disable2"), PARAMETERS)
def test_fit(*, directory: str, jit: bool, resume: str, disable0: bool, disable1: bool, disable2: bool) -> None:
    R"""
    Test HMM transformation.

    Args
    ----
    - directory
        Directory of a dataset.
    - jit
        If True, use JIT in testing.
    - resume
        Directory of log to resume.
    - disable0
        Disable step size rendering.
    - disable1
        Disable density bar rendering.
    - disable2
        Dsiable true capacity rendering.

    Returns
    -------
    """
    #
    args = []
    args.extend(["--dataset", directory])
    args.extend(["--transform", os.path.join(directory, "test.json")])
    args.extend(["--device", DEVICE])
    args.extend(["--jit"] if jit else [])
    args.extend(["--resume", resume])
    args.extend(["--num-sample-seconds", "700"])
    args.extend(["--disable-step-size"] if disable0 else [])
    args.extend(["--disable-dense-bar"] if disable1 else [])
    args.extend(["--disable-true-capacity"] if disable2 else [])

    #
    framework = FrameworkTransformHMMStream(disk=LOG_ROOT, clean=True)
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
    for (directory, jit, resume, disable0, disable1, disable2) in PARAMETERS:
        #
        test_fit(directory=directory, jit=jit, resume=resume, disable0=disable0, disable1=disable1, disable2=disable2)


#
if __name__ == "__main__":
    #
    main()
