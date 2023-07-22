#
from __future__ import annotations


#
import torch
import numpy as onp
from typing import Sequence, Tuple
from ..gradient import AlgorithmGradient
from ...models import ModelHMM
from ...loaders import LoaderHMM
from ._jit.gradient import baum_welch_forward_log, baum_welch_backward_log, baum_welch_posterior_log
from ...types import NPNUMS, NPFLOATS, THNUMS


class AlgorithmGradientHMM(AlgorithmGradient[ModelHMM, LoaderHMM]):
    R"""
    Gradient-descent-like HMM algorithm.
    """

    def __init__(self: AlgorithmGradientHMM, model: ModelHMM, /, *, jit: bool) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - model
            HMM model.
        - flag
            If True, use JIT.

        Returns
        -------
        """
        #
        self.losses = ["NLL.Sum", "NLL.Mean"]
        self.metrics = ["NLL.Sum", "NLL.Mean"]
        self.model = model

        #
        self.forward = baum_welch_forward_log[jit]
        self.backward = baum_welch_backward_log[jit]
        self.posterior = baum_welch_posterior_log[jit]

    def train(
        self: AlgorithmGradientHMM,
        model: ModelHMM,
        loader: LoaderHMM,
        /,
    ) -> Tuple[NPFLOATS, NPFLOATS]:
        R"""
        Train model by data of a single iteration from loader.

        Args
        ----
        - model
            Model.
        - loader
            Loader.

        Returns
        -------
        - loss
            Loss.
        - size
            Number of samples considered for the loss.
        """
        #
        (infos, *memory) = next(iter(loader))
        (_, num_smaples, _) = infos.shape

        #
        with torch.no_grad():
            #
            prob_log_sum = torch.tensor(0.0, dtype=model.dfloat, device=model.device)
            prob_log_mean = torch.tensor(0.0, dtype=model.dfloat, device=model.device)
            model.estimate()
            for (begins, ends) in zip(infos[0], infos[-1]):
                #
                sample = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]
                (prob_log, length, posteriors) = self.statistics(model, sample)
                prob_log_sum += prob_log
                prob_log_mean += prob_log / float(length)
                model.accumulate(sample, posteriors)
            model.maximize()

        #
        model.optimizer.zero_grad()
        model.backward()
        model.optimizer.step()
        return (
            onp.array([float(-prob_log_sum.item()), float(-prob_log_mean.item()) / float(num_smaples)]),
            onp.array([float(num_smaples), float(num_smaples)]),
        )

    def statistics(
        self: AlgorithmGradientHMM,
        model: ModelHMM,
        sample: Sequence[THNUMS],
        /,
    ) -> Tuple[THNUMS, int, Sequence[THNUMS]]:
        R"""
        Get posterior sufficient statistics memory to be accumulated for EM.

        Args
        ----
        - model
            Model.
        - loader
            Loader.
        - optimizer
            Optimizer used to update model from gradient descent.

        Returns
        -------
        - prob_log
            Log probability of given sample under current model.
        - length
            Length of given sample under current model.
        - memory
            Sufficient statistics memory.
        """
        #
        (initials, transitions, emissions_log) = model.forward(sample)
        (_, length) = emissions_log.shape
        initials_log = torch.log(initials)
        transitions_log = torch.log(transitions)

        #
        (alphas_log, prob_log) = self.forward(initials_log, transitions_log, emissions_log)
        betas_log = self.backward(transitions_log, emissions_log)
        (gammas_log, xis_log) = self.posterior(transitions_log, emissions_log, alphas_log, betas_log)
        gammas = torch.exp(gammas_log)
        xisums = torch.exp(torch.logsumexp(xis_log, dim=2))
        return (prob_log, length, [gammas, xisums])

    @torch.no_grad()
    def evaluate(
        self: AlgorithmGradientHMM,
        model: ModelHMM,
        loader: LoaderHMM,
        /,
    ) -> Tuple[NPFLOATS, NPFLOATS]:
        R"""
        Evaluate model by data of a single iteration from loader.

        Args
        ----
        - model
            Model.
        - loader
            Loader.

        Returns
        -------
        - metric
            Metric.
        - size
            Number of samples considered for the metric.
        """
        #
        (infos, *memory) = next(iter(loader))
        (_, num_smaples, _) = infos.shape

        #
        prob_log_sum = torch.tensor(0.0, dtype=model.dfloat, device=model.device)
        prob_log_mean = torch.tensor(0.0, dtype=model.dfloat, device=model.device)
        for (begins, ends) in zip(infos[0], infos[-1]):
            #
            sample = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]

            #
            (initials, transitions, emissions_log) = model.forward(sample)
            (_, length) = emissions_log.shape
            initials_log = torch.log(initials)
            transitions_log = torch.log(transitions)
            (_, prob_log) = self.forward(initials_log, transitions_log, emissions_log)
            prob_log_sum += prob_log
            prob_log_mean += prob_log / float(length)
        return (
            onp.array([float(-prob_log_sum.item()), float(-prob_log_mean.item()) / float(num_smaples)]),
            onp.array([float(num_smaples), float(num_smaples)]),
        )

    def details(
        self: AlgorithmGradientHMM,
        model: ModelHMM,
        loader: LoaderHMM,
        /,
    ) -> Sequence[NPNUMS]:
        R"""
        Collect detail information.

        Args
        ----
        - model
            Model.
        - loader
            Loader.

        Returns
        -------
        - memory
            Memory of detail information.
        """
        #
        (infos, *memory) = next(iter(loader))

        #
        with torch.no_grad():
            #
            buf_emissions_log = []
            buf_alphas_log = []
            buf_prob_log = []
            buf_betas_log = []
            buf_gammas = []
            buf_xisums = []
            for (begins, ends) in zip(infos[0], infos[-1]):
                #
                sample = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]
                (initials, transitions, emissions_log) = model.forward(sample)

                #
                initials_log = torch.log(initials)
                transitions_log = torch.log(transitions)
                (alphas_log, prob_log) = self.forward(initials_log, transitions_log, emissions_log)
                betas_log = self.backward(transitions_log, emissions_log)
                (gammas_log, xis_log) = self.posterior(transitions_log, emissions_log, alphas_log, betas_log)
                gammas = torch.exp(gammas_log)
                xisums = torch.exp(torch.logsumexp(xis_log, dim=2))

                #
                buf_emissions_log.append(emissions_log.cpu().numpy())
                buf_alphas_log.append(alphas_log.cpu().numpy())
                buf_prob_log.append(prob_log.cpu().numpy())
                buf_betas_log.append(betas_log.cpu().numpy())
                buf_gammas.append(gammas.cpu().numpy())
                buf_xisums.append(xisums.cpu().numpy())

        #
        return [
            infos.cpu().numpy(),
            onp.concatenate(buf_emissions_log, axis=1),
            onp.concatenate(buf_alphas_log, axis=1),
            onp.stack(buf_prob_log),
            onp.concatenate(buf_betas_log, axis=1),
            onp.concatenate(buf_gammas, axis=1),
            onp.stack(buf_xisums, axis=2),
        ]
