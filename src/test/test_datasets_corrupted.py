#
import pytest
import os
import json
import shutil
from typing import Dict, Callable
from veritas.datasets import DataHMMCategorical, DataHMMGaussian, DataHMMStream


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]


#
DATA_DIR0 = os.path.join(DATA_ROOT, "__Test__")
DATA_DIR1 = os.path.join(DATA_ROOT, "HMMNaiveCategorical")
DATA_DIR2 = os.path.join(DATA_ROOT, "HMMPriorCategorical")
DATA_DIR3 = os.path.join(DATA_ROOT, "HMMNaiveGaussian")
DATA_DIR4 = os.path.join(DATA_ROOT, "HMMPriorGaussian")
DATA_DIR5 = os.path.join(DATA_ROOT, "Stream2")


#
PARAMETERS = [
    ("categorical", DATA_DIR1),
    ("categorical", DATA_DIR2),
    ("gaussian", DATA_DIR3),
    ("gaussian", DATA_DIR4),
    ("stream", DATA_DIR5),
]
DATAHMMS = {"categorical": DataHMMCategorical, "gaussian": DataHMMGaussian, "stream": DataHMMStream}
OBSERVATIONS = {"categorical": "observation", "gaussian": "observation", "stream": "video_session_streams"}
HIDDENS = {"categorical": "hidden", "gaussian": "hidden", "stream": "ground_truth_capacity"}


def fhash_reverse(fhashes: Dict[str, Dict[str, str]], /) -> Dict[str, Dict[str, str]]:
    R"""
    Reverse file hashing.

    Args
    ----
    - fhashes
        File hash values.

    Returns
    -------
    - fhashes
        File hash values.
    """
    #
    return {key0: {key1: val1[::-1] for (key1, val1) in val0.items()} for (key0, val0) in fhashes.items()}


def fhash_wildcard(fhashes: Dict[str, Dict[str, str]], /) -> Dict[str, Dict[str, str]]:
    R"""
    Force wildcard file hashing.

    Args
    ----
    - fhashes
        File hash values.

    Returns
    -------
    - fhashes
        File hash values.
    """
    #
    return {key0: {key1: "*" for key1 in val0} for (key0, val0) in fhashes.items()}


def directory_subdirs(fhashes: Dict[str, Dict[str, str]], /) -> Dict[str, Dict[str, str]]:
    R"""
    Add sub directories to data directory.

    Args
    ----
    - fhashes
        File hash values.

    Returns
    -------
    """
    # We need a non-empty sub directory.
    for (root, subdirs, _) in os.walk(DATA_DIR0):
        #
        if root != DATA_DIR0:
            #
            continue
        for subdir in subdirs:
            #
            os.makedirs(os.path.join(root, subdir, "__Subdir__"))
    return fhashes


def template_test_corrupt(
    *,
    kind: str,
    directory: str,
    corrupt: Callable[[Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]],
) -> None:
    R"""
    Test corner case(s).

    Args
    ----
    - kind
        Task kind.
    - directory
        Directory of a dataset.
    - corrupt
        Function to corrupt dataset.

    Returns
    -------
    """
    # Copy given dataset to a temporary dataset.
    if os.path.isdir(DATA_DIR0):
        #
        shutil.rmtree(DATA_DIR0)
    shutil.copytree(directory, DATA_DIR0)
    with open(os.path.join(directory, "fhash.json"), "r") as file:
        #
        fhashes = json.load(file)

    # Corrupt temporary dataset and file hash values.
    fhashes = corrupt(fhashes)

    # Load corrupted dataset should raise error.
    DATAHMMS[kind].from_csv(
        DATA_DIR0,
        fhashes,
        title_observation=OBSERVATIONS[kind],
        title_hidden=HIDDENS[kind],
        filenames=None,
    )


@pytest.mark.xfail
@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_corrupt_reverse(*, kind: str, directory: str) -> None:
    R"""
    Test corner case for reversed file hashing.

    Args
    ----
    - kind
        Dataset kind.
    - directory
        Directory of a dataset.

    Returns
    -------
    """
    #
    template_test_corrupt(kind=kind, directory=directory, corrupt=fhash_reverse)


@pytest.mark.xfail
@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_corrupt_wildcard(*, kind: str, directory: str) -> None:
    R"""
    Test corner case for wildcard file hashing.

    Args
    ----
    - kind
        Dataset kind.
    - directory
        Directory of a dataset.

    Returns
    -------
    """
    #
    template_test_corrupt(kind=kind, directory=directory, corrupt=fhash_wildcard)


@pytest.mark.xfail
@pytest.mark.parametrize(("kind", "directory"), PARAMETERS)
def test_corrupt_subdir(*, kind: str, directory: str) -> None:
    R"""
    Test corner case for having unexpected sub directories.

    Args
    ----
    - kind
        Dataset kind.
    - directory
        Directory of a dataset.

    Returns
    -------
    """
    #
    template_test_corrupt(kind=kind, directory=directory, corrupt=directory_subdirs)


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
    for test_corrupt in [test_corrupt_reverse, test_corrupt_wildcard, test_corrupt_subdir]:
        #
        for (kind, directory) in PARAMETERS:
            #
            try:
                #
                test_corrupt(kind=kind, directory=directory)
            except RuntimeError:
                #
                pass
    test_clear()


#
if __name__ == "__main__":
    #
    main()
