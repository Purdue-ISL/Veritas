#
from __future__ import annotations


#
import abc
from typing import Tuple, TypeVar
from ...model import Model, Memory
from ....types import THNUMS


#
AnyInputs = TypeVar("AnyInputs", bound="Tuple[THNUMS, ...]")
AnyOutputs = TypeVar("AnyOutputs", bound="Tuple[THNUMS, ...]")


class ModelEmission(Model[AnyInputs, AnyOutputs]):
    R"""
    Emission distribution model.
    """

    def __annotate__(self: ModelEmission[AnyInputs, AnyOutputs], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self.num_hiddens: int

    @abc.abstractmethod
    def estimate(self: ModelEmission[AnyInputs, AnyOutputs], /) -> None:
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
    def accumulate(self: ModelEmission[AnyInputs, AnyOutputs], inputs: Memory, posterior: THNUMS, /) -> None:
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
        pass

    @abc.abstractmethod
    def maximize(self: ModelEmission[AnyInputs, AnyOutputs], /) -> None:
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
    def backward(self: ModelEmission[AnyInputs, AnyOutputs], /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        #
        pass
