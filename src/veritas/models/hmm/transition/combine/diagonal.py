#
from __future__ import annotations


#
import torch
from ....model import Memory
from .combine import ModelTransitionCombine


class ModelTransitionDiag(ModelTransitionCombine):
    R"""
    Diagonal transition matrix model.
    """

    def reset(self: ModelTransitionDiag, rng: torch.Generator, /) -> ModelTransitionDiag:
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
        weights = torch.ones(len(self.weights))
        self.weights.data.copy_(weights)
        return self


class ModelTransitionDiagSym(ModelTransitionDiag):
    R"""
    Symmetric 3-diagonal transition matrix model.
    """

    def allocate_weights(self: ModelTransitionDiagSym, /) -> torch.nn.Parameter:
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
        half = (self.num_focuses - 1) // 2
        return torch.nn.Parameter(torch.zeros(2 + half, dtype=self.dfloat))

    def allocate_basises(self: ModelTransitionDiagSym, /) -> torch.Tensor:
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
        buf = [torch.diagflat(torch.ones(self.num_hiddens, dtype=self.dfloat))]
        half = (self.num_focuses - 1) // 2
        for i in range(half):
            #
            buf.append(
                torch.diagflat(torch.ones(self.num_hiddens - 1 - i, dtype=self.dfloat), i + 1)
                + torch.diagflat(torch.ones(self.num_hiddens - 1 - i, dtype=self.dfloat), -(i + 1)),
            )
        buf.append(
            torch.triu(torch.ones(self.num_hiddens, self.num_hiddens, dtype=self.dfloat), half + 1)
            + torch.tril(torch.ones(self.num_hiddens, self.num_hiddens, dtype=self.dfloat), -(half + 1)),
        )
        return torch.stack(buf)

    def forward(self: ModelTransitionDiagSym, inputs: Memory, /) -> Memory:
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
        # Weights are not guaranterd to be positive, thus softplus is required.
        weights = torch.reshape(torch.nn.functional.softplus(self.weights), (len(self.weights), 1, 1))
        transitions = torch.sum(weights * self.basises, dim=0)

        # Deal with boundary truncation.
        if self.include_beyond:
            #
            half = (self.num_focuses - 1) // 2
            rest = self.num_hiddens - 1 - half
            offs = torch.reshape(weights[1:], (half + 1,))

            # Decreasing states.
            for i in range(self.num_hiddens):
                #
                for j in range(half - i):
                    #
                    transitions[i, 0].add_(offs[j + i])
                transitions[i, 0].add_(offs[half], alpha=rest - max(i - half, 0))

            # Increasing states.
            for i in range(self.num_hiddens):
                #
                for j in range(half - self.num_hiddens + 1 + i):
                    #
                    transitions[i, self.num_hiddens - 1].add_(offs[j + self.num_hiddens - i - 1])
                transitions[i, self.num_hiddens - 1].add_(
                    offs[half],
                    alpha=rest - max(self.num_hiddens - i - 1 - half, 0),
                )

        #
        transitions /= torch.sum(transitions, dim=1, keepdim=True)
        return [self.smooth(transitions)]


class ModelTransitionDiagAsym(ModelTransitionDiag):
    R"""
    Asymmetric 3-diagonal transition matrix model.
    """

    def allocate_weights(self: ModelTransitionDiagAsym, /) -> torch.nn.Parameter:
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
        half = (self.num_focuses - 1) // 2
        return torch.nn.Parameter(torch.zeros(3 + half * 2, dtype=self.dfloat))

    def allocate_basises(self: ModelTransitionDiagAsym, /) -> torch.Tensor:
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
        buf = [torch.diagflat(torch.ones(self.num_hiddens, dtype=self.dfloat))]
        half = (self.num_focuses - 1) // 2
        for i in range(half):
            #
            buf.append(torch.diagflat(torch.ones(self.num_hiddens - 1 - i, dtype=self.dfloat), i + 1))
            buf.append(torch.diagflat(torch.ones(self.num_hiddens - 1 - i, dtype=self.dfloat), -(i + 1)))
        buf.append(torch.triu(torch.ones(self.num_hiddens, self.num_hiddens, dtype=self.dfloat), half + 1))
        buf.append(torch.tril(torch.ones(self.num_hiddens, self.num_hiddens, dtype=self.dfloat), -(half + 1)))
        return torch.stack(buf)

    def forward(self: ModelTransitionDiagAsym, inputs: Memory, /) -> Memory:
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
        # Weights are not guaranterd to be positive, thus softplus is required.
        weights = torch.reshape(torch.nn.functional.softplus(self.weights), (len(self.weights), 1, 1))
        weights = torch.reshape(self.weights, (len(self.weights), 1, 1))
        transitions = torch.sum(weights * self.basises, dim=0)

        # Deal with boundary truncation.
        if self.include_beyond:
            #
            half = (self.num_focuses - 1) // 2
            rest = self.num_hiddens - 1 - half
            offs = torch.reshape(weights[1:], (half + 1, 2))

            # Decreasing states.
            for i in range(self.num_hiddens):
                #
                for j in range(half - i):
                    #
                    transitions[i, 0].add_(offs[j + i, 1])
                transitions[i, 0].add_(offs[half, 1], alpha=rest - max(i - half, 0))

            # Increasing states.
            for i in range(self.num_hiddens):
                #
                for j in range(half - self.num_hiddens + 1 + i):
                    #
                    transitions[i, self.num_hiddens - 1].add_(offs[j + self.num_hiddens - i - 1, 0])
                transitions[i, self.num_hiddens - 1].add_(
                    offs[half, 0],
                    alpha=rest - max(self.num_hiddens - i - 1 - half, 0),
                )

        #
        transitions /= torch.sum(transitions, dim=1, keepdim=True)
        return [self.smooth(transitions)]
