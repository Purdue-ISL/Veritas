#
import pytest
import os
import json
from veritas.datasets import DataHMMCategorical, DataHMMGaussian, DataHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]


#
DATA_DIR0 = os.path.join(DATA_ROOT, "HMMNaiveCategorical")
DATA_DIR1 = os.path.join(DATA_ROOT, "HMMPriorCategorical")
DATA_DIR2 = os.path.join(DATA_ROOT, "HMMNaiveGaussian")
DATA_DIR3 = os.path.join(DATA_ROOT, "HMMPriorGaussian")
DATA_DIR4 = os.path.join(DATA_ROOT, "Stream2")


#
PARAMETERS = [
    ("categorical", DATA_DIR0),
    ("categorical", DATA_DIR1),
    ("gaussian", DATA_DIR2),
    ("gaussian", DATA_DIR3),
    ("stream", DATA_DIR4),
]
DATAHMMS = {"categorical": DataHMMCategorical, "gaussian": DataHMMGaussian, "stream": DataHMMStream}
OBSERVATIONS = {"categorical": "observation", "gaussian": "observation", "stream": "video_session_streams"}
HIDDENS = {"categorical": "hidden", "gaussian": "hidden", "stream": "ground_truth_capacity"}


@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_partial(*, kind: str, directory: str) -> None:
    R"""
    Test dataset partial loading.

    Args
    ----
    - kind
        Task kind.
    - directory
        Directory of a dataset.

    Returns
    -------
    """
    # Load dataset from disk.
    with open(os.path.join(directory, "fhash.json"), "r") as file:
        #
        fhashes = json.load(file)
    dataset0 = DATAHMMS[kind].from_csv(
        directory,
        fhashes,
        title_observation=OBSERVATIONS[kind],
        title_hidden=HIDDENS[kind],
        filenames=None,
    )
    dataset1 = DATAHMMS[kind].from_csv(
        directory,
        fhashes,
        title_observation=OBSERVATIONS[kind],
        title_hidden=HIDDENS[kind],
        filenames=[dataset0.get_index(i) for i in range(len(dataset0)) if i % 2 == 1],
    )
    assert len(dataset1) == len(dataset1) // 1


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
        test_partial(kind=kind, directory=directory)


#
if __name__ == "__main__":
    #
    main()
