#
from __future__ import annotations

#
import pandas as pd
import more_itertools as xitertools
import numpy as onp
from typing import Type, Tuple, Dict, Sequence, List
from .data import DataHMM
from ...types import NPRECS, NPINTS, NPSTRS


#
Sample = Tuple[NPRECS, NPRECS]


class DataHMMTrivial(DataHMM):
    R"""
    Concrete data of trivial hidden Markov processes.
    """
    #
    TYPES = {"int64": ["int64"], "float64": ["float64"]}

    def __init__(
        self: DataHMMTrivial,
        indices: NPSTRS,
        sections: NPINTS,
        sizes: NPINTS,
        observations: NPRECS,
        hiddens: NPRECS,
        columns: Sequence[Dict[str, List[str]]],
        /,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - indices
            Indices of all samples.
            It can be numeric or string IDs.
        - sections
            Numeric section assignments of all samples.
        - sizes
            Sizes of all samples.
        - observations
            Observation data of all samples.
        - hiddens
            Hidden state data of all samples.
        - columns
            Data column names.

        Returns
        -------
        """
        #
        DataHMM.__init__(self, indices, sections, sizes, observations, hiddens, columns)

        #
        array = self.hiddens["int64"]
        array = onp.unique(array)
        assert (
            onp.ptp(array) == len(array) - 1 and onp.max(array) == len(array) - 1
        ), "Some hidden state(s) never appear in given dataset."
        self.num_hiddens = len(array)

    @classmethod
    def richness(cls: Type[DataHMMTrivial], sample: Sample, /) -> float:
        R"""
        Compute sample information richness.

        Args
        ----
        - sample
            A sample.

        Returns
        -------
        - score
            Richness score.
        """
        #
        (observation, _) = sample
        return float(len(onp.unique(observation["int64"])))

    @classmethod
    def csv_to_dataframe(cls: Type[DataHMMTrivial], path: str, /) -> pd.DataFrame:
        R"""
        Convert csv file to dataframe.

        Args
        ----
        - path
            Path.

        Returns
        -------
        - dataframe
            Dataframe.
        """
        #
        return pd.read_csv(path)

    @classmethod
    def sync_dataframes(
        cls: Type[DataHMMTrivial],
        dataframe_observation: pd.DataFrame,
        dataframe_hidden: pd.DataFrame,
        /,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        R"""
        Synchronize dataframes.
        It is used for data preprocessing across dataframes.

        Args
        ----
        - dataframe_observation
            Dataframe of observation.
        - dataframe_hidden
            Dataframe of hidden state.

        Returns
        -------
        - dataframe_observation
            Dataframe of observation.
        - dataframe_hidden
            Dataframe of hidden state.
        """
        #
        return (dataframe_observation, dataframe_hidden)

    @classmethod
    def recarray_to_csv(
        cls: Type[DataHMMTrivial],
        recarray: NPRECS,
        columns: Dict[str, List[str]],
        path: str,
        /,
    ) -> None:
        R"""
        Convert record array to csv file.

        Args
        ----
        - recarray
            Record array.
        - columns
            Column names.
        - path
            Path.

        Returns
        -------
        """
        #
        assert recarray.dtype.names is not None and set(recarray.dtype.names) == set(
            cls.TYPES.keys(),
        ), "Data record array should have only {:s} fields.".format(
            ", ".join('"{:s}"'.format(name) for name in cls.TYPES.keys()),
        )
        dataframe = pd.concat(
            [pd.DataFrame(recarray[name]) for name in recarray.dtype.names if recarray.dtype[name].itemsize > 0],
            axis=1,
            ignore_index=True,
        )
        flattened = xitertools.flatten(
            [columns[name] for name in recarray.dtype.names if recarray.dtype[name].itemsize > 0]
        )
        dataframe.rename(columns={i: name for (i, name) in enumerate(flattened)}, inplace=True)
        dataframe.to_csv(path, index=False)
