#
from __future__ import annotations

#
import pandas as pd
import numpy as onp
from typing import Type, Tuple, Dict, Sequence, Optional, List
from .meta import MetaHMM
from .data import DataHMM
from ...types import NPRECS, NPFLOATS


#
Sample = Tuple[NPRECS, NPRECS]


class DataHMMStream(DataHMM):
    R"""
    Concrete data of video streaming hidden Markov processes.
    The relationship from observation to hidden state can be zero-to-one, one-to-one, and many-to-one.
    """

    #
    # \\ TYPES = {"int64": ["int64"], "float64": ["float64"], "datetime64": ["datetime64[ns, UTC]", "datetime64[ns]"]}
    TYPES = {"float64": ["float64"], "datetime64": ["datetime64[ns, UTC]", "datetime64[ns]"]}

    @classmethod
    def richness(cls: Type[DataHMMStream], sample: Sample, /) -> float:
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
        (_, hidden) = sample
        capacity = hidden["float64"][0]
        return float(onp.ptp(capacity).item())

    @classmethod
    def csv_to_dataframe(cls: Type[DataHMMStream], path: str, /) -> pd.DataFrame:
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
        dataframe = pd.read_csv(path)

        #
        for name in ["start_time", "end_time"]:
            #
            if name in dataframe:
                #
                dataframe[name] = pd.to_datetime(dataframe[name])

        # This part seems to be cleaned in raw data.
        # Will be removed in next update.
        # \\:# For the model, we need observation data in the same memory block, thus we have to force potential integer
        # \\:# data into floating data.
        # \\:for name in ["client_trans_time"]:
        # \\:    #
        # \\:    if name in dataframe:
        # \\:        #
        # \\:        dataframe[name] = dataframe[name].astype("float64")

        #
        return dataframe

    @classmethod
    def sync_dataframes(
        cls: Type[DataHMMStream],
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
        columns = cls.TYPES["datetime64"]
        datetimes_observation = {
            name: series.dt.tz_localize(None)
            for (name, series) in dataframe_observation.select_dtypes(include=columns).items()
        }
        datetimes_hidden = {
            name: series.dt.tz_localize(None)
            for (name, series) in dataframe_hidden.select_dtypes(include=columns).items()
        }
        datum_observation = min(series.min() for series in datetimes_observation.values())
        datum_hidden = min(series.min() for series in datetimes_hidden.values())
        datum = min(datum_observation, datum_hidden)

        #
        for (name, series) in datetimes_observation.items():
            #
            dataframe_observation[name] = series
            dataframe_observation["{:s}_elapsed".format(name)] = (series - datum).dt.total_seconds()
        for (name, series) in datetimes_hidden.items():
            #
            dataframe_hidden[name] = series
            dataframe_hidden["{:s}_elapsed".format(name)] = (series - datum).dt.total_seconds()
        return (dataframe_observation, dataframe_hidden)

    @classmethod
    def recarray_to_csv(
        cls: Type[DataHMMStream],
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
        raise NotImplementedError("Video streaming data is read-only since it is collected from other sources.")

    def sanitize_exact_coverage(self: DataHMMStream, /) -> None:
        R"""
        Sanitize samples to ensure that hidden maximum time exactly covers observation maximum time.

        Args
        ----

        Returns
        -------
        """
        #
        k0 = self.columns[0]["float64"].index("end_time_elapsed")
        k1 = self.columns[1]["float64"].index("start_time_elapsed")

        #
        buf_id = []
        buf_size = []
        buf_observation = []
        buf_hidden = []
        for (id, (starts, ends)) in enumerate(zip(self.starts.T, self.ends.T)):
            #
            blocks = [block[start:end] for (block, start, end) in zip(self.memory, starts, ends)]
            (observation, hidden) = blocks

            #
            last_time_elapsed = observation["float64"][-1, k0]
            end_time_elapsed = hidden["float64"][:, k1]
            (indices,) = onp.where(end_time_elapsed > last_time_elapsed)
            breakpoint = onp.min(indices).item() + 1
            hidden = hidden[:breakpoint]

            #
            assert (
                len(observation) > 0 and len(hidden) > 0
            ), 'Exact coverage sanitization removes sample "{:s}" which is not allowed.'.format(self.indices[id])
            buf_id.append(id)
            buf_size.append((len(observation), len(hidden)))
            buf_observation.append(observation)
            buf_hidden.append(hidden)

        #
        self.indices = self.indices[buf_id]
        self.sections = self.sections[buf_id]
        self.sizes = onp.array(buf_size).T
        self.observations = onp.concatenate(buf_observation)
        self.hiddens = onp.concatenate(buf_hidden)
        self.memory = [self.observations, self.hiddens]
        (self.starts, self.ends) = self.sizes_to_bounds(self.sizes)

        #
        for (starts, ends) in zip(self.starts.T, self.ends.T):
            #
            (observation, hidden) = [block[start:end] for (block, start, end) in zip(self.memory, starts, ends)]
            assert onp.max(observation["float64"][:, k0]) <= onp.max(
                hidden["float64"][:, k1],
            ), "Observation covers time which has no collected hidden state."

    def sanitize_capacity_max(self: DataHMMStream, threshold: float, /) -> None:
        R"""
        Sanitize beyond-maximum values in capacity columns.

        Args
        ----
        - threshold
            Maximum threshold.
            Values equal to the threshold is valid (inclusive).

        Returns
        -------
        """
        #
        j = self.columns[1]["float64"].index("bandwidth")
        k1 = self.columns[1]["float64"].index("start_time_elapsed")
        k0 = self.columns[0]["float64"].index("end_time_elapsed")

        #
        buf_id = []
        buf_size = []
        buf_observation = []
        buf_hidden = []
        for (id, (starts, ends)) in enumerate(zip(self.starts.T, self.ends.T)):
            #
            blocks = [block[start:end] for (block, start, end) in zip(self.memory, starts, ends)]
            (observation, hidden) = blocks

            # Truncate according to hidden state.
            bandwidth = hidden["float64"][:, j]
            (indices,) = onp.where(bandwidth > threshold)
            breakpoint = onp.min(indices).item() if len(indices) > 0 else len(hidden) + 1
            hidden = hidden[:breakpoint]

            # Truncate observation to ensure its end time is within hidden state last time.
            assert (
                len(hidden) > 0
            ), 'Capacity maximum ({:.2f}) sanitization removes sample "{:s}" which is not allowed.'.format(
                threshold,
                self.indices[id],
            )
            last_time_elapsed = hidden["float64"][-1, k1].item()
            end_time_elapsed = observation["float64"][:, k0]
            (indices,) = onp.where(end_time_elapsed > last_time_elapsed)
            breakpoint = onp.min(indices).item() if len(indices) > 0 else len(observation) + 1
            observation = observation[:breakpoint]

            #
            assert (
                len(observation) > 0 and len(hidden) > 0
            ), 'Capacity maximum ({:.2f}) sanitization removes sample "{:s}" which is not allowed.'.format(
                threshold, self.indices[id]
            )
            buf_id.append(id)
            buf_size.append((len(observation), len(hidden)))
            buf_observation.append(observation)
            buf_hidden.append(hidden)

        #
        self.indices = self.indices[buf_id]
        self.sections = self.sections[buf_id]
        self.sizes = onp.array(buf_size).T
        self.observations = onp.concatenate(buf_observation)
        self.hiddens = onp.concatenate(buf_hidden)
        self.memory = [self.observations, self.hiddens]
        (self.starts, self.ends) = self.sizes_to_bounds(self.sizes)

        # Ensure the sanitization exlcude all invalid cases.
        assert onp.all(
            self.hiddens["float64"][:, j] <= threshold
        ), "Beyond-maximum bandwidth still be found in sanitized hidden state data."
        for (starts, ends) in zip(self.starts.T, self.ends.T):
            #
            (observation, hidden) = [block[start:end] for (block, start, end) in zip(self.memory, starts, ends)]
            assert onp.max(observation["float64"][:, k0]) <= onp.max(
                hidden["float64"][:, k1],
            ), "Observation covers time which has no collected hidden state."

    def sanitize_sample_short(self: DataHMMStream, num: int, /) -> None:
        R"""
        Sanitize samples without enough observation.

        Args
        ----
        - num
            Minimum number of observations.
            This minimum number is valid (inclusive).

        Returns
        -------
        """
        #
        buf_id = []
        buf_size = []
        buf_observation = []
        buf_hidden = []
        for (id, (starts, ends)) in enumerate(zip(self.starts.T, self.ends.T)):
            #
            blocks = [block[start:end] for (block, start, end) in zip(self.memory, starts, ends)]
            (observation, hidden) = blocks

            #
            if len(observation) >= num:
                #
                buf_id.append(id)
                buf_size.append((len(observation), len(hidden)))
                buf_observation.append(observation)
                buf_hidden.append(hidden)

        #
        self.indices = self.indices[buf_id]
        self.sections = self.sections[buf_id]
        self.sizes = onp.array(buf_size).T
        self.observations = onp.concatenate(buf_observation)
        self.hiddens = onp.concatenate(buf_hidden)
        self.memory = [self.observations, self.hiddens]
        (self.starts, self.ends) = self.sizes_to_bounds(self.sizes)


def get_capacities_discretized(dataset: MetaHMM, capunit: float, /) -> Sequence[NPFLOATS]:
    R"""
    Get discretized capacities from dataset.

    Args
    ----
    - dataset
        Dataset.
    - capunit
        Discretization unit.

    Returns
    -------
    - capicities
        Discretized capacity.
    """
    # Capacity column must be the float64 column with title "bandwidth" in hidden state data.
    j = dataset.columns[1]["float64"].index("bandwidth")

    #
    buf = []
    for i in range(len(dataset)):
        #
        (_, hidden) = dataset[i]
        capacity = hidden["float64"][:, j]
        capacity = onp.round(capacity / capunit) * capunit
        buf.append(capacity)
    return buf


def get_capacities_continuous(dataset: MetaHMM, /) -> Sequence[NPFLOATS]:
    R"""
    Get continuous capacities from dataset.

    Args
    ----
    - dataset
        Dataset.

    Returns
    -------
    - capicities
        Continuous capacity.
    """
    # Capacity column must be the float64 column with title "bandwidth" in hidden state data.
    j = dataset.columns[1]["float64"].index("bandwidth")

    #
    buf = []
    for i in range(len(dataset)):
        #
        (_, hidden) = dataset[i]
        capacity = hidden["float64"][:, j]
        buf.append(capacity)
    return buf
