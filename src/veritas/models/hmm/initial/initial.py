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


class ModelInitial(Model[AnyInputs, Outputs]):
    R"""
    Initial distribution model.
    """

    def __annotate__(self: ModelInitial[AnyInputs], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self.num_hiddens: int

    @classmethod
    def outputs(cls: Type[ModelInitial[AnyInputs]], memory: Memory, /) -> Outputs:
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
        (initials,) = memory
        return (initials,)

    @abc.abstractmethod
    def estimate(self: ModelInitial[AnyInputs], /) -> None:
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
    def accumulate(self: ModelInitial[AnyInputs], posterior: THNUMS, /) -> None:
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
    def maximize(self: ModelInitial[AnyInputs], /) -> None:
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
    def backward(self: ModelInitial[AnyInputs], /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        #
        pass
