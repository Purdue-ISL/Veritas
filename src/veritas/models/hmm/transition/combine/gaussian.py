#
from __future__ import annotations


#
import torch
import math
from ....model import Memory
from .combine import ModelTransitionCombine
from .....types import THNUMS


class ModelTransitionGaussian(ModelTransitionCombine):
    R"""
    Gaussian transition matrix model.
    """

    def reset(self: ModelTransitionGaussian, rng: torch.Generator, /) -> ModelTransitionGaussian:
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
        weights = torch.rand(len(self.weights), generator=rng)
        self.weights.data.copy_(weights)
        return self

    def allocate_basises(self: ModelTransitionGaussian, /) -> torch.Tensor:
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
        bufups = [torch.diagflat(torch.ones(self.num_hiddens, dtype=self.dfloat))]
        buflws = [torch.diagflat(torch.ones(self.num_hiddens, dtype=self.dfloat))]
        for i in range(1, self.num_hiddens):
            #
            bufups.append(torch.diagflat(torch.ones(self.num_hiddens - i, dtype=self.dfloat), i))
            buflws.append(torch.diagflat(torch.ones(self.num_hiddens - i, dtype=self.dfloat), -i))
        return torch.stack((torch.stack(bufups), torch.stack(buflws)))


def uidf(num: int, focus: int, scale: THNUMS, /) -> THNUMS:
    R"""
    Unit interval distribution function.

    Args
    ----
    - num
        Number of unit intervals.
    - focus
        Number of focusing intervals.
    - scale
        Standard deviation.

    Returns
    -------
    - uidists
        Unit interval distributions.
    """
    #
    bounds = torch.zeros(num + 1, dtype=scale.dtype, device=scale.device)
    bounds[1:].copy_(-torch.arange(num, dtype=scale.dtype, device=scale.device) - 0.5)

    #
    cdists = 0.5 * (1.0 + torch.erf(bounds / scale / math.sqrt(2.0)))
    uidists = cdists[:-1] - cdists[1:]
    uidists[-1].add_(cdists[-1])

    # Average on uninteresting transitions.
    focus = (focus - 1) // 2 + 1
    if focus < len(uidists):
        #
        uidists[focus:].fill_(torch.mean(uidists[focus:]))
    return uidists


class ModelTransitionGaussianSym(ModelTransitionGaussian):
    R"""
    Symmetric Gaussian transition matrix model.
    """

    def allocate_weights(self: ModelTransitionGaussianSym, /) -> torch.nn.Parameter:
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
        return torch.nn.Parameter(torch.zeros(1, dtype=self.dfloat))

    def forward(self: ModelTransitionGaussianSym, inputs: Memory, /) -> Memory:
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
        # Utilize Gaussian CDF whose scale is positive.
        (scale,) = torch.nn.functional.softplus(self.weights)
        weights = uidf(self.num_hiddens, self.num_focuses, scale)
        weights_up = torch.reshape(weights, (self.num_hiddens, 1, 1))
        weights_lw = torch.reshape(weights, (self.num_hiddens, 1, 1))
        transitions = torch.sum(weights_up * self.basises[0] + weights_lw * self.basises[1], dim=0)

        # Deal with boundary truncation.
        if self.include_beyond:
            # Decreasing states.
            for i in range(self.num_hiddens - 1):
                #
                transitions[i, 0].add_(torch.sum(weights_lw[i + 1 :]))

            # Increasing states.
            for i in range(1, self.num_hiddens):
                #
                transitions[i, -1].add_(torch.sum(weights_up[self.num_hiddens - i :]))

        #
        transitions /= torch.sum(transitions, dim=1, keepdim=True)
        return [self.smooth(transitions)]


class ModelTransitionGaussianAsym(ModelTransitionGaussian):
    R"""
    Asymmetric Gaussian transition matrix model.
    """

    def allocate_weights(self: ModelTransitionGaussianAsym, /) -> torch.nn.Parameter:
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
        return torch.nn.Parameter(torch.zeros(2, dtype=self.dfloat))

    def forward(self: ModelTransitionGaussianAsym, inputs: Memory, /) -> Memory:
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
        # Utilize Gaussian CDF whose scale is positive.
        (scale_up, scale_lw) = torch.nn.functional.softplus(self.weights)
        (basises_up, basises_lw) = self.basises
        weights_up = torch.reshape(uidf(self.num_hiddens, self.num_focuses, scale_up), (self.num_hiddens, 1, 1))
        weights_lw = torch.reshape(uidf(self.num_hiddens, self.num_focuses, scale_lw), (self.num_hiddens, 1, 1))
        transitions = torch.sum(weights_up * basises_up + weights_lw * basises_lw, dim=0)

        #
        transitions /= torch.sum(transitions, dim=1, keepdim=True)
        return [self.smooth(transitions)]
