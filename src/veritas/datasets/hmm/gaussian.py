#
from __future__ import annotations

#
import numpy as onp
from hmmlearn import hmm
from typing import Tuple, Dict, Sequence, Type, List
from .trivial import DataHMMTrivial
from ...types import NPFLOATS, NPINTS, NPSTRS, NPRECS


class DataHMMGaussian(DataHMMTrivial):
    R"""
    Concrete data of Gaussian HMM.
    """

    def __init__(
        self: DataHMMGaussian,
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
        array = self.observations["float64"]
        if array.ndim == 1:
            #
            self.num_features = 1
        else:
            #
            (_, self.num_features) = array.shape

    @classmethod
    def from_simulate(
        cls: Type[DataHMMGaussian],
        initials: NPFLOATS,
        transitions: NPFLOATS,
        means: NPFLOATS,
        covars: NPFLOATS,
        lengths: NPINTS,
        /,
        *,
        random_state: onp.random.RandomState,
    ) -> DataHMMGaussian:
        R"""
        Initialize from simulation.

        Args
        ----
        - initials
            Initial distribution.
        - transitions
            Transition matrix.
        - means
            Mean matrix.
        - covars
            Covariance matrix.
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
        (num_states, num_features) = means.shape
        model = hmm.GaussianHMM(n_components=num_states, params="", init_params="", random_state=random_state)
        model.startprob_ = initials
        model.transmat_ = transitions
        model.means_ = means
        model.covars_ = covars

        #
        buf_size = []
        buf_observation = []
        buf_hidden = []
        for l in lengths.tolist():
            #
            (it_observation, it_hidden) = model.sample(l)
            if num_features == 1:
                #
                it_observation = onp.reshape(it_observation, (len(it_observation),))
            else:
                #
                it_observation = onp.reshape(it_observation, (len(it_observation), num_features))
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
        if num_features == 1:
            #
            observations = onp.recarray(total, dtype=[("int64", onp.int64, 0), ("float64", onp.float64)])
        else:
            #
            observations = onp.recarray(total, dtype=[("int64", onp.int64, 0), ("float64", onp.float64, num_features)])
        hiddens = onp.recarray(total, dtype=[("int64", onp.int64), ("float64", onp.float64, 0)])

        #
        columns: Tuple[Dict[str, List[str]], Dict[str, List[str]]]

        #
        if num_features == 1:
            #
            columns = ({"int64": [], "float64": ["observation"]}, {"int64": ["hidden"], "float64": []})
        else:
            #
            maxlen = len(str(num_features))
            columns = (
                {"int64": [], "float64": ["observation{:0{:d}d}".format(i + 1, maxlen) for i in range(num_features)]},
                {"int64": ["hidden"], "float64": []},
            )

        #
        observations["float64"] = onp.concatenate(buf_observation)
        hiddens["int64"] = onp.concatenate(buf_hidden)
        return cls(indices, sections, sizes, observations, hiddens, columns)
