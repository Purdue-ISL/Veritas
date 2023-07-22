#
from __future__ import annotations


#
import torch
import abc
from typing import Tuple, Type, Optional
from ....model import Memory
from .....types import THNUMS
from ..transition import ModelTransition


#
Inputs = Tuple[()]


class ModelTransitionCombine(ModelTransition[Inputs]):
    R"""
    Combining transition matrix model.
    """

    def __init__(
        self: ModelTransitionCombine,
        num_hiddens: int,
        num_focuses: int,
        include_beyond: bool,
        smoother: float,
        /,
        *,
        dint: Optional[torch.dtype],
        dfloat: Optional[torch.dtype],
        transeta: float,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - num_hiddens
            Number of hidden states.
        - num_focuses
            Number of focusing transition states for prior knowledge.
        - include_beyond
            Include target transition states which is out of valid states in construction.
        - smoother
            Smoothing coefficient.
        - dint
            Integer precision.
        - dfloat
            Floating precision.
        - transeta
            Transition matrix update scale.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        assert num_focuses > 0 and num_focuses % 2 == 1

        #
        self.num_hiddens = num_hiddens
        self.num_focuses = num_focuses
        self.include_beyond = include_beyond
        self.smoother = smoother
        self.transeta = transeta

        #
        self._dint = self.DINT if dint is None else dint
        self._dfloat = self.DFLOAT if dfloat is None else dfloat

        #
        self.weights = self.allocate_weights()

        #
        self.basises: torch.Tensor
        self.transitions_: torch.Tensor

        #
        self.register_buffer("basises", self.allocate_basises())
        self.register_buffer("transitions_", torch.zeros(self.num_hiddens, self.num_hiddens, dtype=self.weights.dtype))

        #
        assert torch.all(torch.logical_or(self.basises == 0.0, self.basises == 1.0))

    @abc.abstractmethod
    def allocate_weights(self: ModelTransitionCombine, /) -> torch.nn.Parameter:
        R"""
        Allocate weights.

        Args
        ----

        Returns
        -------
        - weights
            Weights.
        """
        #
        pass

    @abc.abstractmethod
    def allocate_basises(self: ModelTransitionCombine, /) -> torch.Tensor:
        R"""
        Allocate basises corresponding to weights.

        Args
        ----

        Returns
        -------
        - basises
            Basises.
        """
        #
        pass

    @classmethod
    def inputs(cls: Type[ModelTransitionCombine], memory: Memory, /) -> Inputs:
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
        assert len(memory) == 0
        return ()

    def estimate(self: ModelTransitionCombine, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.transitions_.data.zero_()

    def accumulate(self: ModelTransitionCombine, posterior: THNUMS, /) -> None:
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
        self.transitions_ += posterior

    def maximize(self: ModelTransitionCombine, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        sums = torch.sum(self.transitions_, dim=1, keepdim=True)
        self.transitions_ /= sums

        # For transition rows without any observation, fill it by uniform.
        sums = torch.reshape(sums, (len(sums),))
        self.transitions_[sums == 0] = 1.0 / self.num_hiddens

    def backward(self: ModelTransitionCombine, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        #
        mask = torch.isnan(self.transitions_)
        defaults = mask.to(self.transitions_.dtype)
        defaults /= torch.sum(defaults, dim=1, keepdim=True)
        self.transitions_[mask] = defaults[mask]

        # Gradient should be the difference.
        (transitions,) = self.forward([])
        loss = self.transeta * 0.5 * torch.sum((transitions - self.transitions_) ** 2)
        loss.backward()
