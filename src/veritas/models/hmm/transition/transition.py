#
from __future__ import annotations


#
import abc
from typing import Tuple, Type, TypeVar
from ...model import Model, Memory
from ....types import THNUMS


#
AnyInputs = TypeVar("AnyInputs", bound="Tuple[THNUMS, ...]")
Outputs = Tuple[THNUMS]


class ModelTransition(Model[AnyInputs, Outputs]):
    R"""
    Transition matrix model.
    """

    def __annotate__(self: ModelTransition[AnyInputs], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self.num_hiddens: int
        self.smoother: float

    @classmethod
    def outputs(cls: Type[ModelTransition[AnyInputs]], memory: Memory, /) -> Outputs:
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
        (transitions,) = memory
        return (transitions,)

    @abc.abstractmethod
    def estimate(self: ModelTransition[AnyInputs], /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def accumulate(self: ModelTransition[AnyInputs], posterior: THNUMS, /) -> None:
        R"""
        Suffcient statistics accumulation for HMM EM algorithm.

        Args
        ----
        - posterior
            Posterior estimation of a sample.

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def maximize(self: ModelTransition[AnyInputs], /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def backward(self: ModelTransition[AnyInputs], /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def smooth(self: ModelTransition[AnyInputs], transitions: THNUMS, /) -> THNUMS:
        R"""
        Smooth transition matrix.

        Args
        ----
        - transitions
            Transition matrix.

        Returns
        -------
        - transitions
            Transition matrix.
        """
        #
        transitions = transitions * (1.0 - self.smoother) + 1.0 / float(self.num_hiddens) * self.smoother
        return transitions
