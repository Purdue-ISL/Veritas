#
import pytest
import os
import json
import torch
from veritas.frameworks import FrameworkFitHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]
LOG_ROOT = CONSTANTS["log_root"]


#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#
DATA_DIR0 = os.path.join(DATA_ROOT, "Stream2")


#
PARAMETERS = [("categorical", DATA_DIR0), ("gaussian", DATA_DIR0)]


@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_pseudo(*, kind: str, directory: str) -> None:
    R"""
    Test pseudo emission model.

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
    args = []
    args.extend(["--dataset", directory])
    args.extend(["--train", "7"])
    args.extend(["--valid", "1"])
    args.extend(["--test", "2"])
    args.extend(["--device", DEVICE])
    args.extend(["--initial", "generic"])
    args.extend(["--transition", "generic"])
    args.extend(["--emission", kind])
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

    #
    framework = FrameworkFitHMMStream(disk=LOG_ROOT, clean=True, strict=False)
    framework.phase2(args)
    framework.erase()


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
        test_pseudo(kind=kind, directory=directory)


#
if __name__ == "__main__":
    #
    main()
