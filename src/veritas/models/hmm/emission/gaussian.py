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


class ModelEmissionGaussian(ModelEmission[Inputs, Outputs]):
    R"""
    Gaussian emission distribution model.
    """

    def __init__(
        self: ModelEmissionGaussian,
        num_hiddens: int,
        num_features: int,
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
        - num_features
            Number of features.
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
        self.num_features = num_features
        self.emiteta = emiteta

        #
        self._dint = self.DINT if dint is None else dint
        self._dfloat = self.DFLOAT if dfloat is None else dfloat

        #
        self.means = torch.nn.Parameter(torch.zeros(self.num_hiddens, self.num_features, dtype=self.dfloat))
        self.vars = torch.nn.Parameter(torch.zeros(self.num_hiddens, self.num_features, dtype=self.dfloat))

        #
        self.weights_: torch.Tensor
        self.sums_: torch.Tensor
        self.squares_: torch.Tensor
        self.means_: torch.Tensor
        self.vars_: torch.Tensor

        #
        self.register_buffer("weights_", torch.zeros(self.num_hiddens, dtype=self.dfloat))
        self.register_buffer("sums_", torch.zeros_like(self.means.data))
        self.register_buffer("squares_", torch.zeros_like(self.vars.data))
        self.register_buffer("means_", torch.zeros_like(self.means.data))
        self.register_buffer("vars_", torch.zeros_like(self.vars.data))

    def reset(self: ModelEmissionGaussian, rng: torch.Generator, /) -> ModelEmissionGaussian:
        R"""
        Reset parameters.

        ArgsL
        ----
        - rng
            Random state.

        Returns
        -------
        - self
            Instance itself.
        """
        # Initialize by random distribution.
        means = torch.rand(self.num_hiddens, self.num_features, generator=rng, dtype=self.dfloat)
        vars = torch.rand(self.num_hiddens, self.num_features, generator=rng, dtype=self.dfloat) ** 2
        self.means.data.copy_(means)
        self.vars.data.copy_(vars)
        return self

    def forward(self: ModelEmissionGaussian, inputs: Memory, /) -> Memory:
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
        data = torch.reshape(observation, (1, len(observation), self.num_features))
        mean = torch.reshape(self.means, (self.num_hiddens, 1, self.num_features))
        var = torch.reshape(self.vars, (self.num_hiddens, 1, self.num_features))
        return [log_prob(data, mean, var)]

    @classmethod
    def inputs(cls: Type[ModelEmissionGaussian], memory: Memory, /) -> Inputs:
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
    def outputs(cls: Type[ModelEmissionGaussian], memory: Memory, /) -> Outputs:
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

    def estimate(self: ModelEmissionGaussian, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        # The naive way for Gaussian EM is collecting all weights and observations in the memory, and compute their
        # weighted mean and weighted variance after accumulating all observations.
        # The weighted mean can be simply achieved from weighted sum of observations and sum of weights.
        # The issue is that weighted variance rely on weighted mean, so it seems that there is no way to accumulate it.
        # But if we unravel naive form of weighted variance, we can find that we only need weighted square sum of
        # observations, weighted sum of observations, weighted means and sum of weights.
        # Thus, we only need three accumulators.
        self.weights_.data.zero_()
        self.sums_.data.zero_()
        self.squares_.data.zero_()

    def accumulate(self: ModelEmissionGaussian, inputs: Memory, posterior: THNUMS, /) -> None:
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
        data = torch.reshape(observation, (1, len(observation), self.num_features))
        weight = torch.reshape(posterior, (self.num_hiddens, len(observation), 1))
        self.weights_.add_(torch.sum(weight, dim=(1, 2)))
        self.sums_.add_(torch.sum(weight * data, dim=1))
        self.squares_.add_(torch.sum(weight * data**2, dim=1))

    def maximize(self: ModelEmissionGaussian, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        weights = torch.reshape(self.weights_, (self.num_hiddens, 1))
        self.means_.copy_(self.sums_ / weights)
        self.vars_.copy_((self.squares_ - 2 * self.sums_ * self.means_ + weights * self.means_**2) / weights)

    def backward(self: ModelEmissionGaussian, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        # Gradient should be the difference.
        loss0 = 0.5 * torch.sum((self.means - self.means_) ** 2)
        loss1 = 0.5 * torch.sum((self.vars - self.vars_) ** 2)
        loss = self.emiteta * (loss0 + loss1)
        loss.backward()


def log_prob(data: THNUMS, mean: THNUMS, var: THNUMS, /) -> THNUMS:
    R"""
    Compute log probabilities for data given means and variances.

    Args
    ----
    - data
        Data.
    - mean
        Mean.
    - var
        Variances.
        Square of standard deviations

    Returns
    -------
    - prob_log
        Log probability.
    """
    #
    pi = torch.tensor(torch.pi, dtype=data.dtype, device=data.device)
    eps = torch.tensor(torch.finfo(var.dtype).tiny, dtype=data.dtype, device=data.device)
    var = torch.maximum(var, eps)
    prob_log_ele0 = torch.log(2 * pi)
    prob_log_ele1 = torch.log(var)
    prob_log_ele2 = torch.square(data - mean) / var
    return -0.5 * torch.sum(prob_log_ele0 + prob_log_ele1 + prob_log_ele2, dim=2)
