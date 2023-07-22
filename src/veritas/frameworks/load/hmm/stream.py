#
from __future__ import annotations


#
import os
import json
from typing import Optional
from ....datasets import DataHMMStream
from ....datasets.utils import collect_fhash
from .hmm import FrameworkLoadHMM


class FrameworkLoadHMMStream(FrameworkLoadHMM[DataHMMStream]):
    R"""
    Framework to load video streaming HMM dataset.
    """

    def arguments(self: FrameworkLoadHMMStream, /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkLoadHMM.arguments(self)
        self._parser.add_argument("--capacity-max", type=float, required=True, help="Maximum capacity.")
        self._parser.add_argument("--filter-capmax", action="store_true", help="Filter maximum capacity in dataset.")

    def parse(self: FrameworkLoadHMMStream, /) -> None:
        R"""
        Parse argument(s) from given command(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkLoadHMM.parse(self)
        self._capmax = float(self._args.capacity_max)
        self._filter_capmax = bool(self._args.filter_capmax)

    def dataset_full(self: FrameworkLoadHMMStream, /) -> DataHMMStream:
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
        return load(self._directory_dataset, self._capmax if self._filter_capmax else float("inf"))


def load(directory: str, capmax: float, /) -> DataHMMStream:
    R"""
    Load dataset.

    Args
    ----
    - directory
        Directory of dataset.
    - capmax
        Capacity maximum.

    Returns
    -------
    - dataset
        Dataset.
    """
    #
    print('Dataset at "{:s}".'.format(directory))

    # Load file hash values and ensure them are as defined.
    with open(os.path.join(directory, "fhash.json"), "r") as file:
        #
        fhashes = json.load(file)
    for (title, fhashes_) in collect_fhash(
        directory,
        title_observation="video_session_streams",
        title_hidden="ground_truth_capacity",
    ).items():
        #
        for (key, val) in fhashes_.items():
            #
            assert fhashes[title][key] == val

    #
    dataset = DataHMMStream.from_csv(
        directory,
        fhashes,
        title_observation="video_session_streams",
        title_hidden="ground_truth_capacity",
        filenames=None,
    )
    dataset.sanitize_exact_coverage()
    dataset.sanitize_capacity_max(capmax)
    dataset.sanitize_sample_short(2)
    return dataset
