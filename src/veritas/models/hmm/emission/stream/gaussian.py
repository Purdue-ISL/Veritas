#
from __future__ import annotations


#
import torch
from typing import Tuple, Type
from ....model import Memory
from .....types import THNUMS, THNUMS
from ..emission import ModelEmission
from ..gaussian import ModelEmissionGaussian


#
Inputs = Tuple[THNUMS]
Outputs = Tuple[THNUMS, THNUMS]


class ModelEmissionStreamGaussianPseudo(ModelEmission[Inputs, Outputs]):
    R"""
    Pseudo video streaming Gaussian emission distribution model.
    This can force a convention HMM to be a video streaming like HMM.
    """

    def __init__(
        self: ModelEmissionStreamGaussianPseudo,
        model: ModelEmissionGaussian,
        /,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - model
            Pseudo model.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.model = model

        #
        self._dint = self.model._dint
        self._dfloat = self.model._dfloat

    def reset(
        self: ModelEmissionStreamGaussianPseudo,
        rng: torch.Generator,
        /,
    ) -> ModelEmissionStreamGaussianPseudo:
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
        self.model.reset(rng)
        return self

    def forward(self: ModelEmissionStreamGaussianPseudo, inputs: Memory, /) -> Memory:
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
        gaps = torch.ones(len(observation), dtype=torch.long, device=observation.device)
        gaps[0] = 0
        return [gaps, *self.model.forward(inputs)]

    @classmethod
    def inputs(cls: Type[ModelEmissionStreamGaussianPseudo], memory: Memory, /) -> Inputs:
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
        return ModelEmissionGaussian.inputs(memory)

    @classmethod
    def outputs(cls: Type[ModelEmissionStreamGaussianPseudo], memory: Memory, /) -> Outputs:
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
        (gaps, *memory) = memory
        return (gaps, *ModelEmissionGaussian.outputs(memory))

    def estimate(self: ModelEmissionStreamGaussianPseudo, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.model.estimate()

    def accumulate(self: ModelEmissionStreamGaussianPseudo, inputs: Memory, posterior: THNUMS, /) -> None:
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
        self.model.accumulate(inputs, posterior)

    def maximize(self: ModelEmissionStreamGaussianPseudo, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.model.maximize()

    def backward(self: ModelEmissionStreamGaussianPseudo, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        #
        self.model.backward()
