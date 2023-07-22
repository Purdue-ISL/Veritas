#
from __future__ import annotations


#
import torch
from typing import Tuple, Type, Optional
from ...model import Memory
from ....types import THNUMS
from .emission import ModelEmission


#
Inputs = Tuple[THNUMS]
Outputs = Tuple[THNUMS]


class ModelEmissionCategorical(ModelEmission[Inputs, Outputs]):
    R"""
    Categorical emission distribution model.
    """

    def __init__(
        self: ModelEmissionCategorical,
        num_hiddens: int,
        num_observations: int,
        /,
        *,
        dint: Optional[torch.dtype],
        dfloat: Optional[torch.dtype],
        emiteta: float,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - num_hiddens
            Number of hidden states.
        - num_observations
            Number of observations.
        - dint
            Integer precision.
        - dfloat
            Floating precision.
        - emiteta
            Emission distribution update scale.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_hiddens = num_hiddens
        self.num_observations = num_observations
        self.emiteta = emiteta

        #
        self._dint = self.DINT if dint is None else dint
        self._dfloat = self.DFLOAT if dfloat is None else dfloat

        #
        self.emissions = torch.nn.Parameter(torch.zeros(self.num_hiddens, self.num_observations, dtype=self.dfloat))

        #
        self.emissions_: torch.Tensor

        #
        self.register_buffer("emissions_", torch.zeros_like(self.emissions.data))

    def reset(self: ModelEmissionCategorical, rng: torch.Generator, /) -> ModelEmissionCategorical:
        R"""
        Reset parameters.

        Args
        ----
        - rng
            Random state.

        Returns
        -------
        - self
            Instance itself.
        """
        # Initialize by random distribution.
        values = torch.rand(self.num_hiddens, self.num_observations, generator=rng, dtype=self.dfloat)
        values /= torch.sum(values, dim=1, keepdim=True)
        self.emissions.data.copy_(values)
        return self

    def forward(self: ModelEmissionCategorical, inputs: Memory, /) -> Memory:
        R"""
        Forward.

        Args
        ----
        - inputs
            Input memory.

        Returns
        -------
        - outputs
            Output memory.
        """
        #
        (observation, _) = inputs
        return [log_prob(self.emissions)[:, torch.reshape(observation, (len(observation),))]]

    @classmethod
    def inputs(cls: Type[ModelEmissionCategorical], memory: Memory, /) -> Inputs:
        R"""
        Decode memory into exact input form.

        Args
        ----
        - memory
            Decoding memory.

        Returns
        -------
        - outputs
            Exact input form.
        """
        #
        (observation,) = memory
        return (observation,)

    @classmethod
    def outputs(cls: Type[ModelEmissionCategorical], memory: Memory, /) -> Outputs:
        R"""
        Decode memory into exact output form.

        Args
        ----
        - memory
            Decoding memory.

        Returns
        -------
        - outputs
            Exact output form.
        """
        #
        (emissions,) = memory
        return (emissions,)

    def estimate(self: ModelEmissionCategorical, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.emissions_.data.zero_()

    def accumulate(self: ModelEmissionCategorical, inputs: Memory, posterior: THNUMS, /) -> None:
        R"""
        Suffcient statistics accumulation for HMM EM algorithm.

        Args
        ----
        - inputs
            Inputs of a sample.
        - posterior
            Posterior estimation of a sample.

        Returns
        -------
        """
        #
        (observation, _) = inputs
        self.emissions_.index_add_(1, observation, posterior)

    def maximize(self: ModelEmissionCategorical, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.emissions_ /= torch.sum(self.emissions_, dim=1, keepdim=True)

    def backward(self: ModelEmissionCategorical, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        # Gradient should be the difference.
        loss = self.emiteta * 0.5 * torch.sum((self.emissions - self.emissions_) ** 2)
        loss.backward()


def log_prob(prob: THNUMS, /) -> THNUMS:
    R"""
    Compute log probabilities for given probabilities.

    Args
    ----
    - prob
        Probability.

    Returns
    -------
    - prob_log
        Log probability.
    """
    #
    return torch.log(prob)
