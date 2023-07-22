#
from __future__ import annotations


#
import torch
import numpy as onp
from typing import Sequence, Tuple
from ..hmm.gradient import AlgorithmGradientHMM
from ...models import ModelHMM
from ...models.hmm.emission.stream.estimate import discretize_chunk_times_start
from ...loaders import LoaderHMM
from ._jit.gradient import (
    stack_matrix_power,
    baum_welch_forward_log,
    baum_welch_backward_log,
    baum_welch_posterior_log,
    sample_hidden_traces,
    fill_between_traces,
    fill_after_traces,
)
from ...types import NPNUMS, THNUMS, NPFLOATS


class AlgorithmGradientStreamHMM(AlgorithmGradientHMM):
    R"""
    Gradient-descent-like streaming Hidden Markov Model algorithm.
    """

    def __init__(self: AlgorithmGradientStreamHMM, model: ModelHMM, /, *, jit: bool) -> None:
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
        self.stack_matrix_power = stack_matrix_power[jit]
        self.forward = baum_welch_forward_log[jit]
        self.backward = baum_welch_backward_log[jit]
        self.posterior = baum_welch_posterior_log[jit]
        self.sample_hidden_traces = sample_hidden_traces[jit]
        self.fill_between_traces = fill_between_traces[jit]
        self.fill_after_traces = fill_after_traces[jit]

    def statistics(
        self: AlgorithmGradientStreamHMM,
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
        (initials, transitions, gaps, emissions_log) = model.forward(sample)
        (_, length) = emissions_log.shape
        initials_log = torch.log(initials)

        # Allocate all possible transition matrix powers ahead of computation.
        transpowers = self.stack_matrix_power(transitions, torch.max(gaps).item())
        transpowers_log = torch.log(transpowers)[:, :, gaps]

        #
        assert not torch.any(torch.isnan(initials_log)).item()
        assert not torch.any(torch.isnan(transpowers_log)).item()
        assert not torch.any(torch.isnan(emissions_log)).item()
        (alphas_log, prob_log) = self.forward(initials_log, transpowers_log, emissions_log)
        betas_log = self.backward(transpowers_log, emissions_log)

        #
        assert not torch.any(torch.isnan(alphas_log)).item()
        assert not torch.any(torch.isnan(betas_log)).item()
        (gammas_log, xis_log) = self.posterior(transpowers_log, emissions_log, alphas_log, betas_log)

        #
        assert not torch.any(torch.isnan(gammas_log)).item()
        assert not torch.any(torch.isnan(xis_log)).item()
        gammas = torch.exp(gammas_log)
        xisums = torch.exp(torch.logsumexp(xis_log, dim=2))
        return (prob_log, length, [gammas, xisums])

    @torch.no_grad()
    def evaluate(
        self: AlgorithmGradientStreamHMM,
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
            (initials, transitions, gaps, emissions_log) = model.forward(sample)
            (_, length) = emissions_log.shape
            initials_log = torch.log(initials)
            transpowers = self.stack_matrix_power(transitions, torch.max(gaps).item())
            transpowers_log = torch.log(transpowers)[:, :, gaps]
            assert not torch.any(torch.isnan(initials_log)).item()
            assert not torch.any(torch.isnan(transpowers_log)).item()
            assert not torch.any(torch.isnan(emissions_log)).item()
            (_, prob_log) = self.forward(initials_log, transpowers_log, emissions_log)
            prob_log_sum += prob_log
            prob_log_mean += prob_log / float(length)
        return (
            onp.array([float(-prob_log_sum.item()), float(-prob_log_mean.item()) / float(num_smaples)]),
            onp.array([float(num_smaples), float(num_smaples)]),
        )

    def details(
        self: AlgorithmGradientStreamHMM,
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
            buf_gaps = []
            buf_emissions_log = []
            buf_alphas_log = []
            buf_prob_log = []
            buf_betas_log = []
            buf_gammas = []
            buf_xisums = []
            for (begins, ends) in zip(infos[0], infos[-1]):
                #
                sample = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]

                #
                (initials, transitions, gaps, emissions_log) = model.forward(sample)
                initials_log = torch.log(initials)

                # Allocate all possible transition matrix powers ahead of computation.
                transpowers = self.stack_matrix_power(transitions, torch.max(gaps).item())
                transpowers_log = torch.log(transpowers)[:, :, gaps]

                #
                (alphas_log, prob_log) = self.forward(initials_log, transpowers_log, emissions_log)
                betas_log = self.backward(transpowers_log, emissions_log)
                (gammas_log, xis_log) = self.posterior(transpowers_log, emissions_log, alphas_log, betas_log)
                gammas = torch.exp(gammas_log)
                xisums = torch.exp(torch.logsumexp(xis_log, dim=2))

                #
                buf_gaps.append(gaps.cpu().numpy())
                buf_emissions_log.append(emissions_log.cpu().numpy())
                buf_alphas_log.append(alphas_log.cpu().numpy())
                buf_prob_log.append(prob_log.cpu().numpy())
                buf_betas_log.append(betas_log.cpu().numpy())
                buf_gammas.append(gammas.cpu().numpy())
                buf_xisums.append(xisums.cpu().numpy())

        #
        return [
            infos.cpu().numpy(),
            onp.concatenate(buf_gaps),
            onp.concatenate(buf_emissions_log, axis=1),
            onp.concatenate(buf_alphas_log, axis=1),
            onp.stack(buf_prob_log),
            onp.concatenate(buf_betas_log, axis=1),
            onp.concatenate(buf_gammas, axis=1),
            onp.stack(buf_xisums, axis=2),
        ]

    def sample(
        self: AlgorithmGradientStreamHMM,
        num: int,
        seconds_total: float,
        seconds_unit: float,
        times_start: THNUMS,
        times_end: THNUMS,
        gaps: THNUMS,
        transpowers_log: THNUMS,
        emissions_log: THNUMS,
        gammas_log: THNUMS,
        xis_log: THNUMS,
        seed: int,
        /,
    ) -> Tuple[THNUMS, THNUMS]:
        R"""
        Sample hidden state traces.

        Args
        ----
        - num
            Number of sample times.
        - seconds_total
            Total seconds of a single sample.
        - seconds_unit
            Seconds of a unit step in a single sample.
        - times_start
            Observation start times.
        - times_end
            Observation end times.
        - gaps
            HMM transition gaps between observations.
        - transpowers_log
            Stack of log transition matrix powers.
        - emissions_log
            Log distribution of emission.
        - gammas_log
            Posterior log distribution of last state.
        - xis_log
            Posterior log transition matrix starting at each step.
        - seed
            Random seed.

        Returns
        -------
        - samples_crit
            Critical sample traces.
            It is useful for estimation-based evaluation.
        - samples_true
            True sample traces.
        """
        #
        total = int(seconds_total / seconds_unit)
        true_samples = -torch.ones(num, total, dtype=xis_log.dtype, device=xis_log.device)

        # Sample critical points according to throughput estimation function.
        (_, _, length) = xis_log.shape
        rng = torch.Generator(xis_log.device).manual_seed(seed)
        probs = torch.rand((num, length + 1), generator=rng, dtype=xis_log.dtype, device=xis_log.device)
        crit_samples = torch.reshape(self.sample_hidden_traces(probs, gammas_log, xis_log), (num, length + 1))

        # Safely fill critial points to samples.
        steps_start = discretize_chunk_times_start(times_start, seconds_unit)
        step_last = int(torch.max(steps_start).item())
        assert torch.all(gaps[1:] == steps_start[1:] - steps_start[:-1])
        for (i, t) in enumerate(steps_start):
            #
            assert torch.all(true_samples[:, t] < 0) or torch.all(true_samples[:, t] == crit_samples[:, i])
            true_samples[:, t] = crit_samples[:, i]

        # Sample and fill between critical points.
        noncrits = torch.nonzero(gaps > 1)
        noncrits = torch.reshape(noncrits, (len(noncrits),))
        probs = torch.rand(
            (num, int(torch.sum(gaps[noncrits] - 1).item())),
            generator=rng,
            dtype=xis_log.dtype,
            device=xis_log.device,
        )
        self.fill_between_traces(
            probs,
            noncrits,
            steps_start,
            crit_samples,
            transpowers_log,
            emissions_log,
            true_samples,
        )

        # Sample after all critical points.
        probs = torch.rand((num, total - step_last - 1), generator=rng, dtype=xis_log.dtype, device=xis_log.device)
        self.fill_after_traces(probs, total, step_last, crit_samples, transpowers_log, true_samples)
        return (crit_samples, true_samples)
