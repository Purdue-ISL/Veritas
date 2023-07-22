#
from __future__ import annotations


#
import os
import json
from ....datasets import DataHMMCategorical
from ....datasets.utils import collect_fhash
from .hmm import FrameworkLoadHMM


class FrameworkLoadHMMCategorical(FrameworkLoadHMM[DataHMMCategorical]):
    R"""
    Framework to load categorical HMM dataset.
    """

    def dataset_full(self: FrameworkLoadHMMCategorical, /) -> DataHMMCategorical:
        R"""
        Get full HMM dataset.

        Args
        ----

        Returns
        -------
        - dataset
            HMM dataset.
        """
        #
        return load(self._directory_dataset)


def load(directory: str, /) -> DataHMMCategorical:
    R"""
    Load dataset.

    Args
    ----
    - directory
        Directory of dataset.

    Returns
    -------
    - dataset
        Dataset.
    """
    # Load dataset from disk.
    print('Dataset at "{:s}".'.format(directory))
    with open(os.path.join(directory, "fhash.json"), "r") as file:
        #
        fhashes = json.load(file)

    # Ensure file hash values are as defined.
    for (title, fhashes_) in collect_fhash(directory, title_observation="observation", title_hidden="hidden").items():
        #
        for (key, val) in fhashes_.items():
            #
            assert fhashes[title][key] == val

    #
    dataset = DataHMMCategorical.from_csv(
        directory,
        fhashes,
        title_observation="observation",
        title_hidden="hidden",
        filenames=None,
    )
    return dataset
