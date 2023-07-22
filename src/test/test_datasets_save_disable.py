#
import pytest
import os
import json
import shutil
from veritas.datasets import DataHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]


#
DATA_DIR0 = os.path.join(DATA_ROOT, "__Test__")
DATA_DIR1 = os.path.join(DATA_ROOT, "Stream2")


@pytest.mark.xfail
def test_save() -> None:
    R"""
    Test disabled saving.

    Args
    ----

    Returns
    -------
    """
    # Load dataset from disk.
    with open(os.path.join(DATA_DIR1, "fhash.json"), "r") as file:
        #
        fhashes = json.load(file)
    DataHMMStream.from_csv(
        DATA_DIR1,
        fhashes,
        title_observation="video_session_streams",
        title_hidden="ground_truth_capacity",
        filenames=None,
    ).save_csv(DATA_DIR0, title_observation="video_session_streams", title_hidden="ground_truth_capacity")


def test_clear() -> None:
    R"""
    Clear cache as test.

    Args
    ----

    Returns
    -------
    """
    #
    if os.path.isdir(DATA_DIR0):
        #
        shutil.rmtree(DATA_DIR0)


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    try:
        #
        test_save()
    except RuntimeError:
        #
        pass
    test_clear()


#
if __name__ == "__main__":
    #
    main()
