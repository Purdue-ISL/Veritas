#
from __future__ import annotations

#
import numpy as onp
from hmmlearn import hmm
from typing import Tuple, Dict, Sequence, Type, List
from .trivial import DataHMMTrivial
from ...types import NPFLOATS, NPINTS, NPSTRS, NPRECS


class DataHMMCategorical(DataHMMTrivial):
    R"""
    Concrete data of categorical HMM.
    """

    def __init__(
        self: DataHMMCategorical,
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
        DataHMMTrivial.__init__(self, indices, sections, sizes, observations, hiddens, columns)

        #
        array = self.observations["int64"]
        array = onp.unique(array)
        assert (
            onp.ptp(array) == len(array) - 1 and onp.max(array) == len(array) - 1
        ), "Some observation(s) never appear in given dataset."
        self.num_observations = len(array)

    @classmethod
    def from_simulate(
        cls: Type[DataHMMCategorical],
        initials: NPFLOATS,
        transitions: NPFLOATS,
        emissions: NPFLOATS,
        lengths: NPINTS,
        /,
        *,
        random_state: onp.random.RandomState,
    ) -> DataHMMCategorical:
        R"""
        Initialize from simulation.

        Args
        ----
        - initials
            Initial distribution.
        - transitions
            Transition matrix.
        - emissions
            Emission matrix.
        - lengths
            Random sampling lengths.
        - random_state
            Random state.

        Returns
        -------
        - self
            Instance.
        """
        #
        (num_states, _) = emissions.shape
        model = hmm.CategoricalHMM(n_components=num_states, params="", init_params="", random_state=random_state)
        model.startprob_ = initials
        model.transmat_ = transitions
        model.emissionprob_ = emissions

        #
        buf_size = []
        buf_observation = []
        buf_hidden = []
        for l in lengths.tolist():
            #
            (it_observation, it_hidden) = model.sample(l)
            it_observation = onp.reshape(it_observation, (len(it_observation),))
            it_hidden = onp.reshape(it_hidden, (len(it_hidden),))
            buf_size.append((len(it_observation), len(it_hidden)))
            buf_observation.append(it_observation)
            buf_hidden.append(it_hidden)

        #
        maxlen = len(str(len(lengths) - 1))
        indices = onp.array(["{:0{:d}d}".format(i, maxlen) for i in range(len(lengths))])
        sections = onp.zeros_like(lengths)

        #
        total = onp.sum(lengths).item()
        sizes = onp.array(buf_size).T
        observations = onp.recarray(total, dtype=[("int64", onp.int64), ("float64", onp.float64, 0)])
        hiddens = onp.recarray(total, dtype=[("int64", onp.int64), ("float64", onp.float64, 0)])

        #
        columns: Tuple[Dict[str, List[str]], Dict[str, List[str]]]

        #
        columns = ({"int64": ["observation"], "float64": []}, {"int64": ["hidden"], "float64": []})

        #
        observations["int64"] = onp.concatenate(buf_observation)
        hiddens["int64"] = onp.concatenate(buf_hidden)
        return cls(indices, sections, sizes, observations, hiddens, columns)
