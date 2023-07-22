#
from __future__ import annotations


#
import torch
from typing import TypeVar, Tuple, Type
from ..model import Model, Memory
from .initial import ModelInitial
from .transition import ModelTransition
from .emission import ModelEmission
from ...types import THNUMS


#
Inputs = Tuple[THNUMS, ...]
Outputs = Tuple[THNUMS, ...]


#
AnyInputsInitial = TypeVar("AnyInputsInitial", bound="Tuple[THNUMS, ...]")
AnyInputsTransition = TypeVar("AnyInputsTransition", bound="Tuple[THNUMS, ...]")
AnyInputsEmission = TypeVar("AnyInputsEmission", bound="Tuple[THNUMS, ...]")
AnyOutputsEmission = TypeVar("AnyOutputsEmission", bound="Tuple[THNUMS, ...]")


class ModelHMM(Model[Inputs, Outputs]):
    R"""
    HMM model.
    """

    def __init__(
        self: ModelHMM,
        initial: ModelInitial[AnyInputsInitial],
        transition: ModelTransition[AnyInputsTransition],
        emission: ModelEmission[AnyInputsEmission, AnyOutputsEmission],
        /,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - model_initial
            Model to generate initial distribution.
        - model_transition
            Model to generate transition matrix.
        - model_emission
            Model to generate emission distribution.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.initial = initial
        self.transition = transition
        self.emission = emission

        #
        self._dint = self.dunique([self.initial.dint, self.transition.dint, self.emission.dint])
        self._dfloat = self.dunique([self.initial.dfloat, self.transition.dfloat, self.emission.dfloat])

        #
        self.num_hiddens = max(self.initial.num_hiddens, self.transition.num_hiddens)

    def reset(self: ModelHMM, rng: torch.Generator, /) -> ModelHMM:
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
        #
        self.initial.reset(rng)
        self.transition.reset(rng)
        self.emission.reset(rng)
        return self

    def forward(self: ModelHMM, inputs: Memory, /) -> Memory:
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
        return [*self.initial.forward(inputs), *self.transition.forward(inputs), *self.emission.forward(inputs)]

    @classmethod
    def inputs(cls: Type[ModelHMM], memory: Memory, /) -> Inputs:
        R"""
        Decode memory into exact input form.

        Args
        ----
        - memory
            Decoding memory.

        Returns
        -------
        - inputs
            Exact input form.
        """
        #
        return tuple(memory)

    @classmethod
    def outputs(cls: Type[ModelHMM], memory: Memory, /) -> Outputs:
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
        return tuple(memory)

    def estimate(self: ModelHMM, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.initial.estimate()
        self.transition.estimate()
        self.emission.estimate()

    def accumulate(self: ModelHMM, inputs: Memory, posteriors: Memory, /) -> None:
        R"""
        Suffcient statistics accumulation for HMM EM algorithm.

        Args
        ----
        - inputs
            Inputs of a sample.
        - posterior
            Posterior estimations of a sample.

        Returns
        -------
        """
        #
        (gammas, xisums) = posteriors
        self.initial.accumulate(gammas[:, 0])
        self.transition.accumulate(xisums)
        self.emission.accumulate(inputs, gammas)

    def maximize(self: ModelHMM, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.initial.maximize()
        self.transition.maximize()
        self.emission.maximize()

    def backward(self: ModelHMM, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        #
        self.initial.backward()
        self.transition.backward()
        self.emission.backward()
