#
from __future__ import annotations


#
import abc
import numpy as onp
import os
import more_itertools as xitertools
import copy
import pandas as pd
from typing import Type, Tuple, Sequence, Dict, Union, cast, List, Optional, TypeVar
from ...types import NPINTS, NPSTRS, NPRECS, NPBOOLS
from ..dataset import DataFinite
from .meta import MetaHMM


#
SelfDataHMM = TypeVar("SelfDataHMM", bound="DataHMM")
Sample = Tuple[NPRECS, NPRECS]


class DataHMM(DataFinite[Sample], MetaHMM):
    R"""
    Concrete data of hidden Markov processes.
    """

    def __init__(
        self: DataHMM,
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
        self.indices = indices
        self.sections = sections
        self.sizes = sizes
        self.observations = observations
        self.hiddens = hiddens
        self.columns = columns
        self.memory = [self.observations, self.hiddens]

        #
        assert all(
            str(onp.dtype(name)) == name for name in self.TYPES
        ), "Keys of data type definition must be the exact numpy dtype name."
        self.types = self.TYPES

        #
        assert self.sizes.shape[0] == 2, "Sample sizes should be defined exactly for observation and hidden state."
        (self.starts, self.ends) = self.sizes_to_bounds(self.sizes)

    @staticmethod
    def sizes_to_bounds(sizes: NPINTS, /) -> Tuple[NPINTS, NPINTS]:
        R"""
        Get sample upper and lower bounds from sizes.

        Args
        ----
        - sizes
            Sample sizes in the block.

        Returns
        -------
        - starts
            Sample start point bounds in the block.
        - end
            Sample end point bounds in the block.
        """
        #
        assert sizes.ndim == 2, "Sample sizes should be separately defined for each memory block."

        #
        bounds = onp.zeros((sizes.shape[0], 1 + sizes.shape[1]), dtype=sizes.dtype)
        onp.cumsum(sizes, axis=1, out=bounds[:, 1:])
        return (bounds[:, :-1], bounds[:, 1:])

    @classmethod
    def from_csv(
        cls: Type[SelfDataHMM],
        directory: str,
        fhashes: Dict[str, Dict[str, str]],
        /,
        *,
        title_observation: str,
        title_hidden: str,
        filenames: Optional[Sequence[str]],
    ) -> SelfDataHMM:
        R"""
        Initialize from loading csv file.

        Args
        ----
        - directory
            directory.
        - fhashes
            File hash values.
        - title_observation
            Observation directory title.
        - title_hidden
            Hidden state directory title.
        - filenames
            Filenames to be loaded.
            If it is None, load all possible files.

        Returns
        -------
        - self
            Instance.
        """
        #
        directory_observation = os.path.join(directory, title_observation)
        directory_hidden = os.path.join(directory, title_hidden)
        if filenames is None:
            #
            filenames_ = cls.collect_indices(directory, title_observation=title_observation, title_hidden=title_hidden)
        else:
            #
            filenames_ = filenames

        #
        indices = onp.array(filenames_)
        sections = onp.zeros((len(filenames_),), dtype=onp.int64)
        buf_size = []
        buf_observation = []
        buf_hidden = []
        for name in filenames_:
            #
            path_observation = os.path.join(directory_observation, name)
            path_hidden = os.path.join(directory_hidden, name)
            cls.validate_path(path_observation, fhashes[title_observation][name])
            cls.validate_path(path_hidden, fhashes[title_hidden][name])

            #
            dataframe_observation = cls.csv_to_dataframe(path_observation)
            dataframe_hidden = cls.csv_to_dataframe(path_hidden)
            (dataframe_observation, dataframe_hidden) = cls.sync_dataframes(dataframe_observation, dataframe_hidden)
            (observation, columns_observation) = cls.dataframe_to_recarray(dataframe_observation)
            (hidden, columns_hidden) = cls.dataframe_to_recarray(dataframe_hidden)

            #
            buf_size.append((len(observation), len(hidden)))
            buf_observation.append(observation)
            buf_hidden.append(hidden)
        sizes = onp.array(buf_size).T
        observations = cast(NPRECS, onp.concatenate(buf_observation))
        hiddens = cast(NPRECS, onp.concatenate(buf_hidden))
        return cls(indices, sections, sizes, observations, hiddens, [columns_observation, columns_hidden])

    @classmethod
    def collect_indices(
        cls: Type[DataHMM],
        directory: str,
        /,
        *,
        title_observation: str,
        title_hidden: str,
    ) -> Sequence[str]:
        R"""
        Collect all file name indices under dataset directory.

        Args
        ----
        - directory
            directory.
        - title_observation
            Observation directory title.
        - title_hidden
            Hidden state directory title.

        Returns
        -------
        - names
            File names as indices.
        """
        #
        directory_observation = os.path.join(directory, title_observation)
        directory_hidden = os.path.join(directory, title_hidden)
        buf_name = []
        for directory in (directory_observation, directory_hidden):
            #
            for (root, _, filenames) in os.walk(directory):
                #
                if root != directory:
                    #
                    raise RuntimeError(
                        'Data directory "{:s}" should not have non-empty sub directories.'.format(directory)
                    )
                else:
                    #
                    buf_name.extend(filenames)
        return list(sorted(set(buf_name)))

    @classmethod
    @abc.abstractmethod
    def csv_to_dataframe(cls: Type[DataHMM], path: str, /) -> pd.DataFrame:
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
        pass

    @classmethod
    @abc.abstractmethod
    def sync_dataframes(
        cls: Type[DataHMM],
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
        pass

    @classmethod
    def dataframe_to_recarray(
        cls: Type[DataHMM],
        dataframe: pd.DataFrame,
        /,
    ) -> Tuple[NPRECS, Dict[str, List[str]]]:
        R"""
        Convert dataframe to record array.

        Args
        ----
        - dataframe
            Dataframe.

        Returns
        -------
        - recarray
            Record array.
        - columns
            Data columns.
        """
        #
        columns: Dict[str, List[str]]
        buf: List[Union[Tuple[str, type], Tuple[str, type, int]]]

        #
        columns = {}
        arraies = {}
        length = len(dataframe)
        buf = []
        for (name, dtypes) in cls.TYPES.items():
            #
            dataframe_ = dataframe.select_dtypes(include=dtypes)
            columns[name] = list(dataframe_.columns)
            ncols = len(columns[name])
            array = dataframe_.to_numpy().astype(name)
            if ncols == 1:
                #
                arraies[name] = onp.reshape(array, (length,))
                buf.append((name, array.dtype))
            else:
                #
                arraies[name] = onp.reshape(array, (length, ncols))
                buf.append((name, array.dtype, ncols))

        #
        unreadables = set(dataframe.columns) - set(xitertools.flatten(columns.values()))
        assert len(unreadables) == 0, "data column(s) {:s} is unreadable as numeric data.".format(
            ", ".join('"{:s}"'.format(name) for name in sorted(unreadables))
        )

        #
        recarray = onp.recarray((length,), dtype=buf)
        for (name, array) in arraies.items():
            #
            recarray[name] = array
        return (recarray, columns)

    def save_csv(
        self: DataHMM,
        directory: str,
        /,
        *,
        title_observation: str,
        title_hidden: str,
    ) -> Dict[str, Dict[str, str]]:
        R"""
        Save to csv file.

        Args
        ----
        - directory
            Directory.
        - title_observation
            Observation directory title.
        - title_hidden
            Hidden state directory title.

        Returns
        -------
        - fhashes
            Saved file hash values.
        """
        #
        directory_observation = os.path.join(directory, title_observation)
        directory_hidden = os.path.join(directory, title_hidden)
        for directory in (directory_observation, directory_hidden):
            #
            if not os.path.isdir(directory):
                #
                os.makedirs(os.path.join(directory))

        #
        fhashes: Dict[str, Dict[str, str]]

        #
        (columns_observation, columns_hidden) = self.columns
        fhashes = {title_observation: {}, title_hidden: {}}
        for i in range(len(self)):
            #
            name = self.indices[i].item()
            path_observation = os.path.join(directory_observation, name)
            path_hidden = os.path.join(directory_hidden, name)
            (observation, hidden) = self[i]
            self.recarray_to_csv(observation, columns_observation, path_observation)
            self.recarray_to_csv(hidden, columns_hidden, path_hidden)
            fhashes[title_observation][name] = self.get_fhash(path_observation)
            fhashes[title_hidden][name] = self.get_fhash(path_hidden)
        return fhashes

    @classmethod
    @abc.abstractmethod
    def recarray_to_csv(
        cls: Type[DataHMM],
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
        pass

    @classmethod
    @abc.abstractmethod
    def richness(self: Type[MetaHMM], sample: Sample, /) -> float:
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
        pass

    def copy_sections(self: SelfDataHMM, takes: Sequence[int], /) -> SelfDataHMM:
        R"""
        Create a subset copy from given section IDs.

        Args
        ----
        - takes
            Section IDs to be taken.

        Returns
        -------
        - self
            Instance.
        """
        #
        return self.copy_mask(onp.isin(self.sections, takes))

    def copy_indices(self: SelfDataHMM, takes: Sequence[str], /) -> SelfDataHMM:
        R"""
        Create a subset copy from given indices.

        Args
        ----
        - takes
            Section IDs to be taken.

        Returns
        -------
        - self
            Instance.
        """
        #
        return self.copy_mask(onp.isin(self.indices, takes))

    def copy_mask(self: SelfDataHMM, masks: NPBOOLS, /) -> SelfDataHMM:
        R"""
        Create a subset copy from given sample mask.

        Args
        ----
        - masks
            Sample masks to be taken.

        Returns
        -------
        - self
            Instance.
        """
        #
        copy_indices = self.indices[masks]
        copy_sections = self.sections[masks]
        copy_sizes = self.sizes[:, masks]

        # Vectorized trick to copy by multiple slices.
        buf = []
        for (i, block) in enumerate((self.observations, self.hiddens)):
            #
            length = len(block)
            starts = self.starts[i, masks]
            ends = self.ends[i, masks]
            hits = onp.zeros((length,), dtype=onp.int64)
            hits[starts] += 1
            hits[ends[ends < length]] += -1
            buf.append(block[onp.cumsum(hits).astype(onp.bool8)])
        (copy_observations, copy_hiddens) = buf

        #
        copy_columns = copy.deepcopy(self.columns)
        return self.__class__(copy_indices, copy_sections, copy_sizes, copy_observations, copy_hiddens, copy_columns)

    def __getitem__(self: DataHMM, id: int, /) -> Sample:
        R"""
        Get item.

        Args
        ----
        - id
            Index.

        Returns
        -------
        - observation
            Observation trace of given index.
        - hidden
            Hidden state trace of given index.
        """
        #
        buf = []
        for (j, block) in enumerate((self.observations, self.hiddens)):
            #
            buf.append(block[self.starts[j, id].item() : self.ends[j, id].item()])
        (observation, hidden) = buf
        return (observation, hidden)
