#
from __future__ import annotations


#
import torch
from typing import Tuple, Type, Optional
from ...model import Memory
from ....types import THNUMS
from .transition import ModelTransition


#
Inputs = Tuple[()]


class ModelTransitionGeneric(ModelTransition[Inputs]):
    R"""
    Generic transition matrix model.
    """

    def __init__(
        self: ModelTransitionGeneric,
        num_hiddens: int,
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
        self.num_hiddens = num_hiddens
        self.smoother = smoother
        self.transeta = transeta

        #
        self._dint = self.DINT if dint is None else dint
        self._dfloat = self.DFLOAT if dfloat is None else dfloat

        #
        self.transitions = torch.nn.Parameter(torch.zeros(self.num_hiddens, self.num_hiddens, dtype=self.dfloat))

        #
        self.register_buffer("transitions_", torch.zeros_like(self.transitions.data))

        #
        self.transitions_: torch.Tensor

    def reset(self: ModelTransitionGeneric, rng: torch.Generator, /) -> ModelTransitionGeneric:
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
        # Initialize by uniform distribution.
        values = torch.ones(self.num_hiddens, self.num_hiddens, dtype=self.dfloat)
        values /= torch.sum(values, dim=1, keepdim=True)
        self.transitions.data.copy_(values)
        return self

    def forward(self: ModelTransitionGeneric, inputs: Memory, /) -> Memory:
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
        transitions = self.transitions
        return [self.smooth(transitions)]

    @classmethod
    def inputs(cls: Type[ModelTransitionGeneric], memory: Memory, /) -> Inputs:
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

    def estimate(self: ModelTransitionGeneric, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.transitions_.data.zero_()

    def accumulate(self: ModelTransitionGeneric, posterior: THNUMS, /) -> None:
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

    def maximize(self: ModelTransitionGeneric, /) -> None:
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
        masks = sums == 0
        if torch.any(masks).item():
            #
            sums = torch.reshape(sums, (len(sums),))
            self.transitions_[sums == 0] = 1.0 / self.num_hiddens
        assert torch.all(
            torch.isclose(
                torch.sum(self.transitions_, dim=1),
                torch.tensor([1.0], dtype=self.transitions_.dtype, device=self.transitions_.device),
            ),
        ).item()

    def backward(self: ModelTransitionGeneric, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        # Gradient should be the difference.
        loss = self.transeta * 0.5 * torch.sum((self.transitions - self.transitions_) ** 2)
        loss.backward()
