#
from __future__ import annotations


#
import argparse
import numpy as onp
import torch
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence, cast, Any, List
from matplotlib.gridspec import GridSpec
from ...framework import Framework
from ....datasets import ViewHMM, DataHMMStream
from ....models import ModelHMM
from ....loaders import LoaderHMM
from ....algorithms import AlgorithmGradientStreamHMM
from ....types import THNUMS, NPNUMS
from ...fit.hmm.stream import FrameworkFitHMMStream


def sample_scores_mse_(sample_througputs: torch.Tensor, true_throughputs: torch.Tensor) -> torch.Tensor:
    R"""
    Compute sample MSE scores.

    Args
    ----
    - sample_throughputs
        Sample througputs.
    - true_througputs
        True througputs.

    Returns
    -------
    - mses
        MSE scores.
    """
    #
    true_throughputs = torch.reshape(true_throughputs, (1, len(true_throughputs)))
    return torch.mean((sample_througputs - true_throughputs) ** 2, dim=1)


def sample_scores_nll_(
    sample_hiddens: torch.Tensor,
    initials_log: torch.Tensor,
    transpowers_log: torch.Tensor,
    emissions_log: torch.Tensor,
) -> torch.Tensor:
    R"""
    Compute sample NLL scores.

    Args
    ----
    - sample_hiddens
        Sample hidden states.
    - initials_log
        Initial log distribution.
    - transpowers_log
        Stack of log transition matrix powers.
    - emissions_log
        Emission log distribution.

    Returns
    -------
    - nlls
        NLL scores.
    """
    #
    (num_samples, num_steps) = sample_hiddens.shape
    steps = torch.arange(num_steps, device=emissions_log.device)
    buf = []
    for i in range(num_samples):
        #
        lls0 = initials_log[sample_hiddens[i, 0]]
        lls1 = transpowers_log[sample_hiddens[i, :-1], sample_hiddens[i, 1:], steps[1:]]
        lls2 = emissions_log[sample_hiddens[i], steps]
        buf.append(-(lls0 + torch.sum(lls1) + torch.sum(lls2)) / num_steps)
    return torch.stack(buf)


#
sample_scores_mse = {False: sample_scores_mse_, True: torch.jit.script(sample_scores_mse_)}
sample_scores_nll = {False: sample_scores_nll_, True: torch.jit.script(sample_scores_nll_)}


class FrameworkTransformHMMStream(Framework):
    R"""
    Framework to transform video streaming HMM dataset.
    """

    def __annotate__(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self._dataset_full: DataHMMStream
        self._model: ModelHMM
        self._algorithm: AlgorithmGradientStreamHMM

        #
        self._directory_dataset: str
        self._capmax: float
        self._capunit: float

        #
        Framework.__annotate__(self)

    def __init__(self: FrameworkTransformHMMStream, /, *, disk: str, clean: bool) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - disk
            Directory for logging space allocation.
        - clean
            If True, destroy logging space after successful exection.

        Returns
        -------
        """
        #
        self._disk = disk
        self._clean = clean

    def arguments(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.arguments(cast(Any, self))
        self._parser1 = self._parser

        #
        self._parser = argparse.ArgumentParser("Transform by HMM Algorithms")
        self._parser.add_argument("--suffix", type=str, required=False, default="", help="Saving title suffix.")
        self._parser.add_argument("--dataset", type=str, required=True, help="Dataset directory.")
        self._parser.add_argument("--transform", type=str, required=True, help="Transformation index definition.")
        self._parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed.")
        self._parser.add_argument("--device", type=str, required=True, help="Computation device.")
        self._parser.add_argument("--jit", action="store_true", help="Enable JIT.")
        self._parser.add_argument("--resume", type=str, required=True, help="Resume from given log.")
        self._parser.add_argument(
            "--num-random-samples",
            type=int,
            required=False,
            default=5,
            help="Number of random samples.",
        )
        self._parser.add_argument(
            "--num-sample-seconds",
            type=float,
            required=True,
            help="Number of seconds (can be float) per sample. This argument assumes training time unit is second.",
        )
        self._parser.add_argument("--disable-step-size", action="store_true", help="Disable step size rendering.")
        self._parser.add_argument("--disable-dense-bar", action="store_true", help="Disable density bar rendering.")
        self._parser.add_argument(
            "--disable-true-capacity",
            action="store_true",
            help="Disable true capacity rendering.",
        )
        self._parser0 = self._parser

    def framein(self: FrameworkTransformHMMStream, cmds: Sequence[str], /) -> None:
        R"""
        Get arguments from framework input interface.

        Args
        ----
        - cmds
            Commands.

        Returns
        -------
        """
        #
        self._parser = self._parser0
        self._args = self._parser.parse_args() if len(cmds) == 0 else self._parser.parse_args(cmds)
        self._args0 = self._args

        #
        self._title_suffix = str(self._args.suffix)
        self._directory_dataset = str(self._args.dataset)
        self._definition_transform = str(self._args.transform)
        self._seed = int(self._args.seed)
        self._device = str(self._args.device)
        self._jit = bool(self._args.jit)
        self._resume = str(self._args.resume)
        self._num_random_samples = int(self._args.num_random_samples)
        self._num_sample_seconds = int(self._args.num_sample_seconds)
        self._disable_step_size = bool(self._args.disable_step_size)
        self._disable_dense_bar = bool(self._args.disable_dense_bar)
        self._disable_true_capacity = bool(self._args.disable_true_capacity)

        #
        with open(os.path.join(self._resume, "arguments.json"), "r") as file:
            #
            args = json.load(file)
        self._args = argparse.Namespace(**args)
        self._args1 = self._args

        # Copy sharing argument names.
        self._title_suffix0 = self._title_suffix
        self._directory_dataset0 = self._directory_dataset
        self._seed0 = self._seed
        self._device0 = self._device
        self._jit0 = self._jit

    def parse(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Parse argument(s) from given command(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.parse(cast(Any, self))

        # Copy sharing argument names.
        self._title_suffix1 = self._title_suffix
        self._directory_dataset1 = self._directory_dataset
        self._seed1 = self._seed
        self._device1 = self._device
        self._jit1 = self._jit

        # Use correct copy for sharing argument names.
        self._title_suffix = self._title_suffix0
        self._directory_dataset = self._directory_dataset0
        self._seed = self._seed0
        self._device = self._device0
        self._jit = self._jit0

        #
        self.sample_scores_mse = sample_scores_mse[self._jit]
        self.sample_scores_nll = sample_scores_nll[self._jit]

        #
        assert (
            float(self._num_sample_seconds) / float(self._args1.transition_unit)
        ).is_integer(), "Number of sample seconds {:.3f} is not a multiplier of training time unit {:.3f}.".format(
            float(self._num_sample_seconds),
            float(self._args1.transition_unit),
        )

    def datasets(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Prepare dataset(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.datasets(cast(Any, self))
        with open(self._definition_transform, "r") as file:
            #
            range_transform = json.load(file)
        self._dataset_transform = ViewHMM.from_indices(self._dataset_full, range_transform)

    def dataset_full(self: FrameworkTransformHMMStream, /) -> DataHMMStream:
        R"""
        Get full HMM dataset.

        Args
        ----

        Returns
        -------
        - dataset
            HMM dataset.
        """
        #
        return FrameworkFitHMMStream.dataset_full(cast(Any, self))

    def dataset_splits_size(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Split full dataset into multiple subsets by size.

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.dataset_splits_size(cast(Any, self))

    def dataset_splits_index(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Split full dataset into multiple subsets by index.

        Args
        ----

        Returns
        -------
        """
        FrameworkFitHMMStream.dataset_splits_index(cast(Any, self))

    def models(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.models(cast(Any, self))

    def loaders(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Prepare loader(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.loaders(cast(Any, self))
        self._loader_transform = LoaderHMM(self._dataset_transform, self._model)

        # Broadcast dataset column names.
        self._columns = {}
        ((self._columns["observation"],), (self._columns["hidden"],)) = self._loader_transform.blocks

    def algorithms(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMMStream.algorithms(cast(Any, self))

    def execute(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Execute.

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm.model.load_state_dict(
            torch.load(os.path.join(self._resume, "parameters.pt"), map_location=self._algorithm.model.device),
        )
        self._algorithm.model.optimizer.load_state_dict(
            torch.load(os.path.join(self._resume, "optimizer.pt"), map_location=self._algorithm.model.device),
        )

        #
        self._supports = getattr(self._algorithm.model.emission, "supports")

        #
        self.render_initial_and_transition()
        self.render_variance()

        #
        self._disk_sample = os.path.join(self._disk_log, "sample")
        if not os.path.isdir(self._disk_sample):
            #
            os.makedirs(self._disk_sample)
        self.traverse(self._loader_transform)

    def traverse(self: FrameworkTransformHMMStream, loader: LoaderHMM, /) -> None:
        R"""
        Traverse given dataset loader.

        Args
        ----
        - loader
            Dataset loader.

        Returns
        -------
        """
        #
        (infos, *memory) = next(iter(loader))
        (_, total, _) = infos.shape
        maxlen = len(str(total - 1))
        with torch.no_grad():
            #
            for (id, (begins, ends)) in enumerate(zip(infos[0], infos[-1])):
                #
                print("- [{:>0{:d}}/{:>{:d}}]".format(id, maxlen, total - 1, maxlen))
                index = loader.dataset.get_index(id)
                sample = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]
                self.transform("{:>0{:d}}".format(id, maxlen), index, sample)

    def transform(self: FrameworkTransformHMMStream, id: str, index: str, memory: Sequence[THNUMS], /) -> None:
        R"""
        Transform a given sample.

        Args
        ----
        - id
            Numeric identifier.
        - index
            String identifier.
        - sample
            Sample memory.

        Returns
        -------
        """
        #
        (observation, hidden) = memory

        #
        (initials, transitions, gaps, emissions_log) = self._algorithm.model.forward(memory)
        (_, length) = emissions_log.shape
        initials_log = torch.log(initials)
        transpowers = self._algorithm.stack_matrix_power(transitions, torch.max(gaps).item())
        transpowers_log = torch.log(transpowers)[:, :, gaps]
        (alphas_log, prob_log) = self._algorithm.forward(initials_log, transpowers_log, emissions_log)
        betas_log = self._algorithm.backward(transpowers_log, emissions_log)
        (gammas_log, xis_log) = self._algorithm.posterior(transpowers_log, emissions_log, alphas_log, betas_log)

        #
        sizes = observation[:, self._columns["observation"].index("size")]
        congestion_windows = observation[:, self._columns["observation"].index("cwnd")].to(self._algorithm.model.dint)
        round_trip_times = observation[:, self._columns["observation"].index("rtt")]
        retransmisson_timeouts = observation[:, self._columns["observation"].index("rto")]
        round_trip_time_mins = observation[:, self._columns["observation"].index("min_rtt")]
        slow_start_thresholds = observation[:, self._columns["observation"].index("ssthresh")]
        last_sends = observation[:, self._columns["observation"].index("last_snd")] * 1000.0
        delivery_rates = observation[:, self._columns["observation"].index("delivery_rate")]
        chunk_times_start = observation[:, self._columns["observation"].index("start_time_elapsed")]
        chunk_times_end = observation[:, self._columns["observation"].index("end_time_elapsed")]
        chunk_sizes = observation[:, self._columns["observation"].index("size")]
        chunk_transition_times = observation[:, self._columns["observation"].index("trans_time")]
        chunk_throughputs = chunk_sizes * 8.0 / chunk_transition_times

        #
        real_times_start = hidden[:, self._columns["hidden"].index("start_time_elapsed")][:-1]
        real_times_end = hidden[:, self._columns["hidden"].index("start_time_elapsed")][1:]
        real_capacities = hidden[:, self._columns["hidden"].index("bandwidth")][:-1]

        # Get hidden state samples.
        emissions = torch.exp(emissions_log)
        (sample_hiddens_crit, sample_hiddens_full) = self._algorithm.sample(
            self._num_random_samples,
            float(self._num_sample_seconds),
            float(self._args1.transition_unit),
            chunk_times_start,
            chunk_times_end,
            gaps,
            torch.log(transpowers),
            emissions_log,
            gammas_log[:, length - 1],
            xis_log,
            self._seed,
        )
        sample_capacities_crit = sample_hiddens_crit * self._capunit
        sample_capacities_full = sample_hiddens_full * self._capunit

        # Get throughputs corresponding to hidden state samples.
        (potential_throughputs, _) = getattr(self._algorithm.model.emission, "capacity_to_throughput",)(
            sizes,
            congestion_windows,
            round_trip_times,
            retransmisson_timeouts,
            round_trip_time_mins,
            slow_start_thresholds,
            last_sends,
            delivery_rates,
            self._capunit
            * torch.arange(self._num_hiddens, device=sample_capacities_crit.device).to(self._algorithm.model.dfloat),
            self._supports,
        )
        buf_sample_throughputs = []
        for i in range(len(sample_hiddens_crit)):
            #
            indices_state = sample_hiddens_crit[i]
            indices_step = torch.arange(len(indices_state), device=indices_state.device)
            buf_sample_throughputs.append(potential_throughputs[indices_state, indices_step])
        sample_throughputs = torch.stack(buf_sample_throughputs)

        # Evaluate scores of samples.
        sample_mses = self.sample_scores_mse(sample_throughputs, chunk_throughputs)
        sample_nlls = self.sample_scores_nll(sample_hiddens_crit, initials_log, transpowers_log, emissions_log)
        sample_scores = torch.stack((sample_mses, sample_nlls))

        #
        self.render_sample(
            id,
            index,
            float(-prob_log.item()) / float(length),
            chunk_times_start.data.cpu().numpy(),
            chunk_times_end.data.cpu().numpy(),
            emissions.data.cpu().numpy(),
            chunk_throughputs.data.cpu().numpy(),
            gaps.data.cpu().numpy(),
            float(self._num_sample_seconds),
            float(self._args1.transition_unit),
            sample_capacities_full.data.cpu().numpy(),
            real_times_start.data.cpu().numpy(),
            real_times_end.data.cpu().numpy(),
            real_capacities.data.cpu().numpy(),
        )

        #
        directory = os.path.join(self._disk_sample, "{:s}.{:s}".format(id, index))
        if not os.path.isdir(directory):
            #
            os.makedirs(directory)

        #
        pd.DataFrame(
            torch.cat((observation, torch.reshape(chunk_throughputs, (len(chunk_throughputs), 1))), dim=1)
            .data.cpu()
            .numpy(),
            columns=[*self._columns["observation"], "throughput"],
        ).to_csv(os.path.join(directory, "observation.csv"), index=False)
        pd.DataFrame(hidden.data.cpu().numpy(), columns=self._columns["hidden"]).to_csv(
            os.path.join(directory, "hidden.csv"),
            index=False,
        )
        with open(os.path.join(directory, "computation.npy"), "wb") as file:
            #
            onp.save(file, emissions_log.data.cpu().numpy())
            onp.save(file, alphas_log.data.cpu().numpy())
            onp.save(file, betas_log.data.cpu().numpy())
            onp.save(file, gammas_log.data.cpu().numpy())
            onp.save(file, xis_log.data.cpu().numpy())
        pd.DataFrame(
            sample_capacities_crit.T.data.cpu().numpy(),
            columns=[
                "{:>0{:d}d}".format(i, len(str(self._num_random_samples - 1))) for i in range(self._num_random_samples)
            ],
        ).to_csv(os.path.join(directory, "sample_crit.csv"), index=False)
        pd.DataFrame(
            sample_capacities_full.T.data.cpu().numpy(),
            columns=[
                "{:>0{:d}d}".format(i, len(str(self._num_random_samples - 1))) for i in range(self._num_random_samples)
            ],
        ).to_csv(os.path.join(directory, "sample_full.csv"), index=False)
        pd.concat(
            (
                pd.DataFrame({"Score": ["MSE", "NLL"]}),
                pd.DataFrame(
                    sample_scores.data.cpu().numpy(),
                    columns=[
                        "{:>0{:d}d}".format(i, len(str(self._num_random_samples - 1)))
                        for i in range(self._num_random_samples)
                    ],
                ),
            ),
            axis=1,
        ).to_csv(os.path.join(directory, "score.csv"), index=False)

    def render_initial_and_transition(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Render initial distribution and transition matrix.

        Args
        ----

        Returns
        -------
        """
        # Achieve sample-independent initial distribution.
        (tensor,) = self._algorithm.model.initial.forward([])
        initials = tensor.data.cpu().numpy()

        # Achieve sample-independent initial distribution.
        (tensor,) = self._algorithm.model.transition.forward([])
        transitions = tensor.data.cpu().numpy()

        # Broadcast information for future rendering.
        self._num_hiddens = len(initials)
        self._precision = 1

        #
        values = (onp.arange(self._num_hiddens) * self._capunit).tolist()
        maxlen = max(len("{:.{:d}f}".format(value, self._precision)) for value in values)
        self._labels = ["{:>{:d}s}".format("{:.{:d}f}".format(value, self._precision), maxlen) for value in values]

        #
        pd.concat(
            [
                pd.DataFrame({"Capacity": self._labels}),
                pd.DataFrame(onp.reshape(initials, (self._num_hiddens, 1)), columns=["Probability"]),
            ],
            axis=1,
        ).to_csv(
            os.path.join(self._disk_log, "initials.csv"),
            index=False,
        )
        pd.concat(
            [
                pd.DataFrame({"Capacity": self._labels}),
                pd.DataFrame(transitions, columns=self._labels),
            ],
            axis=1,
        ).to_csv(
            os.path.join(self._disk_log, "transitions.csv"),
            index=False,
        )

        # Create x and y coordinates for rendering.
        x = onp.arange(self._num_hiddens + 3) * 1.0 - 0.5
        y = onp.arange(self._num_hiddens + 1) * 1.0 - 0.5
        (y, x) = onp.meshgrid(y, x)

        # Fill distribution values as z coordinates for rendering.
        z = onp.zeros((self._num_hiddens + 2, self._num_hiddens))
        z[0, :] = initials
        z[1, :] = float("nan")
        z[2:, :] = transitions.T

        # Render.
        zoom = max(float(self._num_hiddens), 20.0)
        size = 6.0 * zoom / 20.0
        width = size * (8.0 + zoom) / zoom
        height = size * (2.0 + zoom) / zoom
        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(1, 2, width_ratios=[2 + self._num_hiddens, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.set_aspect(1.0)

        #
        lc = (1.0, 1.0, 1.0, 0.35)
        lw = 1.0 / float(self._num_hiddens - 1)
        quadmesh = ax1.pcolormesh(x, y, z, cmap="viridis", edgecolor=lc, linewidth=lw)
        fig.colorbar(quadmesh, cax=ax2)

        #
        ax1.set_xticks(onp.arange(2, self._num_hiddens + 2))
        ax1.set_xticklabels(self._labels, rotation=45)
        ax1.set_xlabel("Next Capacity (Mbps)")

        #
        ax1.set_yticks(onp.arange(self._num_hiddens))
        ax1.set_yticklabels(self._labels, rotation=45)
        ax1.set_ylabel("Previous Capacity (Mbps)")

        #
        for side in ["right", "left", "top", "bottom"]:
            #
            ax1.spines[side].set_visible(False)

        #
        ax2.set_ylabel("Transition Probability")

        #
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        fig.savefig(os.path.join(self._disk_log, "heatmap.png"))
        plt.close(fig)

    def render_variance(self: FrameworkTransformHMMStream, /) -> None:
        R"""
        Render learned emission variance.

        Args
        ----

        Returns
        -------
        """
        #
        variances = getattr(self._algorithm.model.emission, "vars")().data.cpu().numpy()
        deviations = onp.sqrt(variances).T

        #
        pd.concat(
            [
                pd.DataFrame({"Branch": list(range(len(deviations)))}),
                pd.DataFrame(
                    onp.reshape(deviations, (len(deviations), 2)),
                    columns=["Standard Deviation (Head)", "Standard Deviation (Rest)"],
                ),
            ],
            axis=1,
        ).to_csv(
            os.path.join(self._disk_log, "deviations.csv"),
            index=False,
        )

    def render_sample(
        self: FrameworkTransformHMMStream,
        id: str,
        index: str,
        score: float,
        chunk_times_start: NPNUMS,
        chunk_times_end: NPNUMS,
        emissions: NPNUMS,
        chunk_throughputs: NPNUMS,
        gaps: NPNUMS,
        seconds_total: float,
        seconds_unit: float,
        sample_capacities: NPNUMS,
        real_times_start: NPNUMS,
        real_times_end: NPNUMS,
        real_capacities: NPNUMS,
        /,
    ) -> None:
        R"""
        Render a given sample.

        Args
        ----
        - id
            Numeric identifier.
        - index
            String identifier.
        - score
            Score.
        - chunk_times_start
            Start time of each chunk.
        - chunk_times_start
            End time of each chunk.
        - emissions
            Emission probabilities over all states of each chunk.
        - chunk_throughputs
            Observed throughputs of each chunk.
        - gaps
            Number of discretized transition steps between two chunks.
        - seconds_total
            Total seconds.
        - seconds_unit
            Unit seconds.
        - sample_capacities
            Randomly sampled capacities of each chunk.
        - real_times_start
            Start time of real-time capacity evolution.
        - real_times_end
            End time of real-time capacity evolution.
        - real_capacities
            Observed capacities of real-time capacity evolution.

        Returns
        -------
        """
        #
        num_hiddens = self._num_hiddens
        (_, num_chunks) = emissions.shape
        num_evolves = len(real_capacities)

        #
        if self._disable_step_size and self._disable_dense_bar:
            #
            fig = plt.figure(figsize=(16.0, 6.0))
            gs = GridSpec(1, 1)
            ax2 = fig.add_subplot(gs[0, 0])
        elif not self._disable_step_size and self._disable_dense_bar:
            #
            fig = plt.figure(figsize=(16.0, 6.0))
            gs = GridSpec(2, 1, height_ratios=[1, 4])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
        elif self._disable_step_size and not self._disable_dense_bar:
            #
            fig = plt.figure(figsize=(16.0 + 1.0, 6.0))
            gs = GridSpec(1, 2, width_ratios=[32, 1])
            ax2 = fig.add_subplot(gs[0, 0])
            ax3 = fig.add_subplot(gs[0, 1])
        else:
            #
            fig = plt.figure(figsize=(16.0 + 1.0, 6.0))
            gs = GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[32, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[0:2, 1])

        # Render emission probabilities for each chunk.
        x0 = onp.reshape(onp.stack((chunk_times_start, chunk_times_end)).T, (2 * num_chunks,))
        y0 = (onp.arange(num_hiddens + 1) - 0.5) * self._capunit
        z0 = onp.stack((emissions.T, onp.full_like(emissions.T, float("nan"))), axis=1)
        z0 = onp.reshape(z0, (2 * num_chunks, num_hiddens))[:-1]
        (y0, x0) = onp.meshgrid(y0, x0)
        quadmesh = ax2.pcolormesh(
            x0,
            y0,
            z0,
            cmap=sns.dark_palette("lightgray", reverse=True, as_cmap=True),
            alpha=0.75,
        )

        # Render color bar.
        if not self._disable_dense_bar:
            #
            fig.colorbar(quadmesh, cax=ax3)

        #
        xbuf: List[NPNUMS]
        ybuf: List[NPNUMS]

        #
        xbuf = [cast(NPNUMS, x0)]
        ybuf = [cast(NPNUMS, y0)]

        # Render observed throughputs for each chunk.
        x1 = onp.stack((chunk_times_start, chunk_times_end)).T
        x1 = onp.reshape(x1, (2 * num_chunks,))
        y1 = onp.stack((chunk_throughputs, chunk_throughputs)).T
        y1 = onp.reshape(y1, (2 * num_chunks,))
        ax2.plot(x1, y1, color=sns.color_palette()[1], alpha=0.65)
        xbuf.append(cast(NPNUMS, x1))
        ybuf.append(cast(NPNUMS, y1))

        # Render observed real-time capacities.
        if not self._disable_true_capacity:
            #
            x2 = onp.stack((real_times_start, real_times_end)).T
            x2 = onp.reshape(x2, (2 * num_evolves,))
            y2 = onp.stack((real_capacities, real_capacities)).T
            y2 = onp.reshape(y2, (2 * num_evolves,))
            ax2.plot(x2, y2, color=sns.color_palette()[2], alpha=0.65)
            xbuf.append(cast(NPNUMS, x2))
            ybuf.append(cast(NPNUMS, y2))

        # Render randomly sampled capacities for each chunk.
        steps_total = int(seconds_total / seconds_unit)
        sample_times = onp.linspace(0.0, seconds_total, num=steps_total + 1, endpoint=True)
        sample_times_start = sample_times[:-1]
        sample_times_end = sample_times[1:]
        sample_times = onp.reshape(onp.stack((sample_times_start, sample_times_end)).T, (2 * steps_total,))
        for i in range(len(sample_capacities)):
            #
            x_ = sample_times
            y_ = onp.stack((sample_capacities[i], sample_capacities[i])).T
            y_ = onp.reshape(y_, (2 * steps_total,))
            ax2.plot(x_, y_, color=sns.color_palette()[0], alpha=0.85)
            xbuf.append(cast(NPNUMS, x_))
            ybuf.append(cast(NPNUMS, y_))

        # Render discretized HMM transition steps between middle of two chunks.
        if not self._disable_step_size:
            #
            chunk_times_middle = (chunk_times_start + chunk_times_end) / 2
            transit_ranges_middle = (chunk_times_middle[:-1] + chunk_times_middle[1:]) / 2
            transit_ranges_width = chunk_times_middle[1:] - chunk_times_middle[:-1]
            gap_bottom = -0.2
            x3 = transit_ranges_middle
            ax1.bar(
                x3,
                height=gaps[1:] - gap_bottom,
                width=transit_ranges_width,
                bottom=gap_bottom,
                color=sns.color_palette()[3],
                alpha=0.75,
            )
            xbuf.append(cast(NPNUMS, x3))

        # Collect coordinate statistics.
        xmin = min([onp.min(xit).item() for xit in xbuf])
        xmax = max([onp.max(xit).item() for xit in xbuf])
        xptp = xmax - xmin
        xptp = 1.05 if xptp == 0.0 else xptp

        #
        ymin2 = min([onp.min(yit).item() for yit in ybuf])
        ymax2 = max([onp.max(yit).item() for yit in ybuf])
        yptp2 = ymax2 - ymin2
        yptp2 = 1.05 if yptp2 == 0.0 else yptp2

        #
        ax2.set_xlabel("Elapsed Time (sec)")
        ax2.set_ylabel("Capacity/Throught (Mbps)")
        ax2.set_xlim(xmin=xmin - xptp * 0.005, xmax=xmax + xptp * 0.005)
        ax2.set_ylim(ymin=ymin2 - yptp2 * 0.01, ymax=ymax2 + yptp2 * 0.01)

        #
        if not self._disable_step_size:
            # Transition step bottom should ensure 0 is still visible.
            ymin1 = 0.0
            ymax1 = max(onp.max(gaps[1:]).item(), 1.0)
            yptp1 = ymax1 - ymin1
            yptp1 = 1.05 if yptp1 == 0.0 else yptp1

            #
            ax1.xaxis.set_visible(False)
            ax1.set_ylabel("Transition Step(s)")
            ax1.set_xlim(xmin=xmin - xptp * 0.025, xmax=xmax + xptp * 0.025)
            ax1.set_ylim(ymin=min(ymin1 - yptp1 * 0.025, gap_bottom), ymax=ymax1 + yptp1 * 0.025)

        #
        if not self._disable_dense_bar:
            #
            ax3.set_ylabel("Chunk Emission Density Given Capacity")

        #
        fig.suptitle("{:s} ({:.3f})".format(index, score))
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        fig.savefig(os.path.join(self._disk_sample, "{:s}.{:s}.png".format(id, index)))
        plt.close(fig)
