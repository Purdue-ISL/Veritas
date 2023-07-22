#
from __future__ import annotations


#
import torch
from typing import Tuple, Type, Optional
from ...model import Memory
from ....types import THNUMS
from .initial import ModelInitial


#
Inputs = Tuple[()]


class ModelInitialGeneric(ModelInitial[Inputs]):
    R"""
    Generic initial distribution model.
    """

    def __init__(
        self: ModelInitialGeneric,
        num_hiddens: int,
        /,
        *,
        dint: Optional[torch.dtype],
        dfloat: Optional[torch.dtype],
        initeta: float,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - num_hiddens
            Number of hidden states.
        - dint
            Integer precision.
        - dfloat
            Floating precision.
        - initeta
            Initial distribution update scale.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_hiddens = num_hiddens
        self.initeta = initeta

        #
        self._dint = self.DINT if dint is None else dint
        self._dfloat = self.DFLOAT if dfloat is None else dfloat

        #
        self.initials = torch.nn.Parameter(torch.zeros(self.num_hiddens, dtype=self.dfloat))

        #
        self.initials_: torch.Tensor

        #
        self.register_buffer("initials_", torch.zeros_like(self.initials.data))

    def reset(self: ModelInitialGeneric, rng: torch.Generator, /) -> ModelInitialGeneric:
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
        values = torch.ones(self.num_hiddens, dtype=self.dfloat)
        values /= torch.sum(values)
        self.initials.data.copy_(values)
        return self

    def forward(self: ModelInitialGeneric, inputs: Memory, /) -> Memory:
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
        return [self.initials]

    @classmethod
    def inputs(cls: Type[ModelInitialGeneric], memory: Memory, /) -> Inputs:
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

    def estimate(self: ModelInitialGeneric, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.initials_.data.zero_()

    def accumulate(self: ModelInitialGeneric, posterior: THNUMS, /) -> None:
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
        self.initials_ += posterior

    def maximize(self: ModelInitialGeneric, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        sums = torch.sum(self.initials_)
        self.initials_ /= sums

        # For initial states without any observation, fill it by uniform.
        masks = sums == 0
        if torch.any(masks).item():
            #
            self.initials_[:] = 1.0 / self.num_hiddens
        assert torch.isclose(
            torch.sum(self.initials_),
            torch.tensor(1.0, dtype=self.initials_.dtype, device=self.initials_.device),
        ).item()

    def backward(self: ModelInitialGeneric, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        # Gradient should be the difference.
        loss = self.initeta * 0.5 * torch.sum((self.initials - self.initials_) ** 2)
        loss.backward()
