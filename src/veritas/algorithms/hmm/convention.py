#
from __future__ import annotations


#
import numpy as onp
import torch
from hmmlearn.base import BaseHMM
from hmmlearn import _hmmc
from typing import Sequence, Tuple
from ..algorithm import Algorithm
from ...loaders import LoaderHMM
from ...types import NPNUMS, NPFLOATS


class AlgorithmConventionHMM(Algorithm[BaseHMM, LoaderHMM]):
    R"""
    Conventional HMM algorithm.
    """

    def __init__(self: AlgorithmConventionHMM, model: BaseHMM, /) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - model
            HMM model

        Returns
        -------
        """
        #
        self.losses = ["NLL.Sum", "NLL.Mean"]
        self.metrics = ["NLL.Sum", "NLL.Mean"]
        self.model = model

        #
        assert (
            self.model.n_iter == 1
        ), "Number of training epochs is controlled by our variable, thus `n_iter` should be 1."

    def fit(
        self: AlgorithmConventionHMM,
        loader_train: LoaderHMM,
        loader_valid: LoaderHMM,
        /,
        *,
        num_epochs: int,
    ) -> Tuple[NPFLOATS, NPFLOATS]:
        R"""
        Tune parameters on given datasets.

        Args
        ----
        - loader_train
            Training dataset loader for updating parameters.
        - loader_valid
            Validation dataset loader for selecting parameters.
        - num_epochs
            Number of updating epochs.

        Returns
        -------
        - losses
            Collection of training losses.
        - metrics
            Collection of validation metrics.
        """
        #
        losses = onp.zeros((len(self.losses), num_epochs, 2))
        metrics = onp.zeros((len(self.metrics), num_epochs, 2))

        #
        for epc in range(num_epochs):
            #
            (losses[:, epc, 0], losses[:, epc, 1]) = self.train(self.model, iter(loader_train))
            (metrics[:, epc, 0], metrics[:, epc, 1]) = self.evaluate(self.model, iter(loader_valid))
        return (losses, metrics)

    def train(
        self: AlgorithmConventionHMM,
        model: BaseHMM,
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
        (infos, observations, _) = next(iter(loader))

        #
        sizes = infos[1, :, 0].cpu().numpy()
        samples = onp.reshape(
            observations.cpu().numpy(),
            (len(observations), observations.numel() // len(observations)),
        )

        #
        model.fit(samples, lengths=sizes)
        return (
            onp.array([float(-model.monitor_.history[-1]), float("nan")]),
            onp.array([float(infos.shape[1]), float("nan")]),
        )

    @torch.no_grad()
    def evaluate(
        self: AlgorithmConventionHMM,
        model: BaseHMM,
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
        (infos, observations, _) = next(iter(loader))

        #
        begins = infos[0, :, 0].cpu().numpy()
        ends = infos[2, :, 0].cpu().numpy()
        samples = onp.reshape(
            observations.cpu().numpy(),
            (len(observations), observations.numel() // len(observations)),
        )

        #
        prob_log_sum = onp.array(0.0, dtype=onp.float64)
        for (begin, end) in zip(begins, ends):
            #
            sample = samples[begin:end]

            #
            emissions_log = model._compute_log_likelihood(sample)
            (prob_log, _) = _hmmc.forward_log(model.startprob_, model.transmat_, emissions_log)
            prob_log_sum += prob_log
        return (onp.array([-prob_log_sum.item(), float("nan")]), onp.array([float(infos.shape[1]), float("nan")]))

    def details(
        self: AlgorithmConventionHMM,
        model: BaseHMM,
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
        (infos, observations, _) = next(iter(loader))

        #
        begins = infos[0, :, 0].cpu().numpy()
        ends = infos[2, :, 0].cpu().numpy()
        samples = onp.reshape(
            observations.cpu().numpy(),
            (len(observations), observations.numel() // len(observations)),
        )

        #
        buf_emissions_log = []
        buf_alphas_log = []
        buf_prob_log = []
        buf_betas_log = []
        buf_gammas = []
        buf_xisums = []
        for (begin, end) in zip(begins, ends):
            #
            sample = samples[begin:end]

            #
            (emissions_log, prob_log, gammas, alphas_log, betas_log) = model._fit_log(sample)
            xisums_log = _hmmc.compute_log_xi_sum(alphas_log, model.transmat_, betas_log, emissions_log)
            xisums = onp.exp(xisums_log)

            #
            buf_emissions_log.append(emissions_log.T)
            buf_alphas_log.append(alphas_log.T)
            buf_prob_log.append(prob_log)
            buf_betas_log.append(betas_log.T)
            buf_gammas.append(gammas.T)
            buf_xisums.append(xisums)

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
