#
from __future__ import annotations


#
import os
import json
from ....datasets import DataHMMGaussian
from ....datasets.utils import collect_fhash
from .hmm import FrameworkLoadHMM


class FrameworkLoadHMMGaussian(FrameworkLoadHMM[DataHMMGaussian]):
    R"""
    Framework to load Gaussian HMM dataset.
    """

    def dataset_full(self: FrameworkLoadHMMGaussian, /) -> DataHMMGaussian:
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


def load(directory: str, /) -> DataHMMGaussian:
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
    dataset = DataHMMGaussian.from_csv(
        directory,
        fhashes,
        title_observation="observation",
        title_hidden="hidden",
        filenames=None,
    )
    return dataset
