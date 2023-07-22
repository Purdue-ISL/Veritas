#
from __future__ import annotations


#
import argparse
import abc
import numpy as onp
import time
import torch
import os
import pandas as pd
import json
from typing import Sequence, Generic, TypeVar
from ...framework import Framework
from ....datasets import ViewHMM, DataHMM
from ....models import ModelHMM
from ....loaders import LoaderHMM
from ....algorithms import AlgorithmGradientHMM
from ....types import NPNUMS


#
AnyData = TypeVar("AnyData", bound="DataHMM")
AnyAlgorithm = TypeVar("AnyAlgorithm", bound="AlgorithmGradientHMM")


class FrameworkFitHMM(Framework, Generic[AnyData, AnyAlgorithm]):
    R"""
    Framework to fit HMM dataset.
    """

    def __annotate__(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm: AnyAlgorithm
        self._model: ModelHMM

        #
        Framework.__annotate__(self)

    def __init__(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /, *, disk: str, clean: bool, strict: bool) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - disk
            Directory for logging space allocation.
        - clean
            If True, destroy logging space after successful exection.
        - strict
            If True, the training performance should never decrease.

        Returns
        -------
        """
        #
        self._disk = disk
        self._clean = clean

        #
        self.strict = strict

    def arguments(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._parser = argparse.ArgumentParser("Fit by HMM Algorithms")
        self._parser.add_argument("--suffix", type=str, required=False, default="", help="Saving title suffix.")
        self._parser.add_argument("--dataset", type=str, required=True, help="Dataset directory.")
        self._parser.add_argument("--train", type=str, required=True, help="Training index definition.")
        self._parser.add_argument("--valid", type=str, required=True, help="Validation index definition.")
        self._parser.add_argument("--test", type=str, required=True, help="Training index definition.")
        self._parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed.")
        self._parser.add_argument("--device", type=str, required=True, help="Computation device.")
        self._parser.add_argument("--jit", action="store_true", help="Enable JIT.")
        self._parser.add_argument("--initial", type=str, required=True, help="Initial model.")
        self._parser.add_argument("--transition", type=str, required=True, help="Transition model.")
        self._parser.add_argument("--emission", type=str, required=True, help="Emission model.")
        self._parser.add_argument(
            "--smooth",
            type=float,
            required=False,
            default=0.0,
            help="Transition model smoother.",
        )
        self._parser.add_argument("--num-epochs", type=int, required=True, help="Number of epochs.")
        self._parser.add_argument("--eq-eps", type=float, required=False, default=1e-5, help="Equality tolerance.")

    def parse(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Parse argument(s) from given command(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._title_suffix = str(self._args.suffix)
        self._directory_dataset = str(self._args.dataset)
        self._definition_train = str(self._args.train)
        self._definition_valid = str(self._args.valid)
        self._definition_test = str(self._args.test)
        self._seed = int(self._args.seed)
        self._device = str(self._args.device)
        self._jit = bool(self._args.jit)
        self._initial = str(self._args.initial)
        self._transition = str(self._args.transition)
        self._emission = str(self._args.emission)
        self._smooth = float(self._args.smooth)
        self._num_epochs = int(self._args.num_epochs)
        self._eq_eps = float(self._args.eq_eps)

    def datasets(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Prepare dataset(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._dataset_full = self.dataset_full()
        if any(
            (
                self._definition_train.isdecimal(),
                self._definition_valid.isdecimal(),
                self._definition_test.isdecimal(),
            ),
        ):
            #
            self.dataset_splits_size()
        else:
            #
            self.dataset_splits_index()

    @abc.abstractmethod
    def dataset_full(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> AnyData:
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
        pass

    def dataset_splits_size(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Split full dataset into multiple subsets by size.

        Args
        ----

        Returns
        -------
        """
        #
        size_train = int(self._definition_train)
        size_valid = int(self._definition_valid)
        size_test = int(self._definition_test)
        size_tune = size_train + size_valid
        size_full = size_tune + size_test

        #
        range_train = list(range(0, size_train))
        range_valid = list(range(size_train, size_tune))
        range_test = list(range(size_tune, size_full))
        range_tune = range_train + range_valid

        #
        self._dataset_full.set_sections_richness_descent(self._dataset_full.richness, size_full)
        self._dataset_tune = self._dataset_full.copy_sections(range_tune)
        self._dataset_test = self._dataset_full.copy_sections(range_test)
        self._dataset_train = ViewHMM.from_sections(self._dataset_tune, range_train)
        self._dataset_valid = ViewHMM.from_sections(self._dataset_tune, range_valid)

    def dataset_splits_index(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Split full dataset into multiple subsets by index.

        Args
        ----

        Returns
        -------
        """
        #
        with open(self._definition_train, "r") as file:
            #
            range_train = json.load(file)
        with open(self._definition_valid, "r") as file:
            #
            range_valid = json.load(file)
        with open(self._definition_test, "r") as file:
            #
            range_test = json.load(file)
        range_tune = list(set(range_train + range_valid))

        #
        self._dataset_tune = self._dataset_full.copy_indices(range_tune)
        self._dataset_test = self._dataset_full.copy_indices(range_test)
        self._dataset_train = ViewHMM.from_indices(self._dataset_tune, range_train)
        self._dataset_valid = ViewHMM.from_indices(self._dataset_tune, range_valid)

    def loaders(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Prepare loader(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._loader_train = LoaderHMM(self._dataset_train, self._model)
        self._loader_valid = LoaderHMM(self._dataset_valid, self._model)
        self._loader_test = LoaderHMM(self._dataset_test, self._model)

    def execute(self: FrameworkFitHMM[AnyData, AnyAlgorithm], /) -> None:
        R"""
        Execute.

        Args
        ----

        Returns
        -------
        """
        #
        self._losses = onp.zeros((len(self._algorithm.losses), self._num_epochs + 1, 2), dtype=onp.float64)
        self._metrics = onp.zeros((len(self._algorithm.metrics), self._num_epochs + 1, 2), dtype=onp.float64)
        assert tuple(self._algorithm.losses) == ("NLL.Sum", "NLL.Mean")
        assert tuple(self._algorithm.metrics) == ("NLL.Sum", "NLL.Mean")
        report_loss = 1
        report_metric = 1

        #
        if not os.path.isdir(os.path.join(self._disk_log, "checkpoints")):
            #
            os.makedirs(os.path.join(self._disk_log, "checkpoints"))

        #
        maxlen0 = 5
        maxlen1 = 23
        maxlen2 = 8
        maxlen3 = 6
        maxlen4 = 10
        maxlen5 = len(str(self._num_epochs))

        #
        print("+-{:s}-+-{:s}-+-{:s}-+-{:s}-+".format("-" * maxlen0, "-" * maxlen1, "-" * maxlen2, "-" * maxlen3))
        print(
            "| {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} |".format(
                "",
                maxlen0,
                "NLL.Mean",
                maxlen1,
                "",
                maxlen2,
                "",
                maxlen3,
            ),
        )
        print(
            "| {:>{:d}s} +-{:s}-+-{:s}-+ {:>{:d}s} | {:>{:d}s} |".format(
                "Epoch",
                maxlen0,
                "-" * maxlen4,
                "-" * maxlen4,
                "Time.Sec",
                maxlen2,
                "Signal",
                maxlen3,
            ),
        )
        print(
            "| {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} |".format(
                "",
                maxlen0,
                "Train",
                maxlen4,
                "Valid",
                maxlen4,
                "",
                maxlen2,
                "",
                maxlen3,
            ),
        )
        print(
            "+-{:s}-+-{:s}-+-{:s}-+-{:s}-+-{:s}-+".format(
                "-" * maxlen0,
                "-" * maxlen4,
                "-" * maxlen4,
                "-" * maxlen2,
                "-" * maxlen3,
            ),
        )

        #
        time_start = time.time()
        ((self._losses[:, 0, 0], self._losses[:, 0, 1]), (self._metrics[:, 0, 0], self._metrics[:, 0, 1])) = (
            (onp.array([float("inf"), float("inf")]), onp.array([1.0, 1.0])),
            self._algorithm.evaluate(self._algorithm.model, self._loader_valid),
        )
        time_end = time.time()
        report_sum_train = self._losses[report_loss, 0, 0].item()
        report_count_train = self._losses[report_loss, 0, 1].item()
        report_sum_valid = self._metrics[report_metric, 0, 0].item()
        report_count_valid = self._metrics[report_metric, 0, 1].item()
        elapsed = time_end - time_start

        #
        improve = True
        self._criterion = report_sum_valid / report_count_valid
        signal = self.signal(improve)

        #
        print(
            "| {:>{:d}d} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:s} |".format(
                0,
                maxlen0,
                "{:.6f}".format(report_sum_train / report_count_train),
                maxlen4,
                "{:.6f}".format(report_sum_valid / report_count_valid),
                maxlen4,
                "{:.3f}".format(elapsed),
                maxlen2,
                signal,
            )
        )

        #
        if improve:
            #
            torch.save(self._algorithm.model.state_dict(), os.path.join(self._disk_log, "parameters.pt"))
            torch.save(self._algorithm.model.optimizer.state_dict(), os.path.join(self._disk_log, "optimizer.pt"))
        torch.save(
            self._algorithm.model.state_dict(),
            os.path.join(self._disk_log, "checkpoints", "parameters{:>0{:d}d}.pt".format(0, maxlen5)),
        )
        torch.save(
            self._algorithm.model.optimizer.state_dict(),
            os.path.join(self._disk_log, "checkpoints", "optimizer{:>0{:d}d}.pt".format(0, maxlen5)),
        )
        self.save_named_matrix(
            self._algorithm.losses,
            self._losses[:, :, 0].T,
            os.path.join(self._disk_log, "fit.losses.sum.csv"),
        )
        self.save_named_matrix(
            self._algorithm.losses,
            self._losses[:, :, 1].T,
            os.path.join(self._disk_log, "fit.losses.count.csv"),
        )
        self.save_named_matrix(
            self._algorithm.metrics,
            self._metrics[:, :, 0].T,
            os.path.join(self._disk_log, "fit.metrics.sum.csv"),
        )
        self.save_named_matrix(
            self._algorithm.metrics,
            self._metrics[:, :, 1].T,
            os.path.join(self._disk_log, "fit.metrics.count.csv"),
        )

        #
        for i in range(1, self._num_epochs + 1):
            #
            time_start = time.time()
            (self._losses[:, [i], :], self._metrics[:, [i], :]) = self._algorithm.fit(
                self._loader_train,
                self._loader_valid,
                num_epochs=1,
            )
            time_end = time.time()
            report_sum_train = self._losses[report_loss, i, 0].item()
            report_count_train = self._losses[report_loss, i, 1].item()
            report_sum_valid = self._metrics[report_metric, i, 0].item()
            report_count_valid = self._metrics[report_metric, i, 1].item()
            elapsed = time_end - time_start

            #
            criterion = report_sum_valid / report_count_valid
            improve = self._criterion >= criterion
            self._criterion = min(self._criterion, criterion)
            signal = self.signal(improve)

            #
            print(
                "| {:>{:d}d} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:s} |".format(
                    i,
                    maxlen0,
                    "{:.6f}".format(report_sum_train / report_count_train),
                    maxlen4,
                    "{:.6f}".format(report_sum_valid / report_count_valid),
                    maxlen4,
                    "{:.3f}".format(elapsed),
                    maxlen2,
                    signal,
                )
            )

            #
            self.validate_loss(i)

            #
            if improve:
                #
                torch.save(self._algorithm.model.state_dict(), os.path.join(self._disk_log, "parameters.pt"))
                torch.save(self._algorithm.model.optimizer.state_dict(), os.path.join(self._disk_log, "optimizer.pt"))
            torch.save(
                self._algorithm.model.state_dict(),
                os.path.join(self._disk_log, "checkpoints", "parameters{:>0{:d}d}.pt".format(i, maxlen5)),
            )
            torch.save(
                self._algorithm.model.optimizer.state_dict(),
                os.path.join(self._disk_log, "checkpoints", "optimizer{:>0{:d}d}.pt".format(i, maxlen5)),
            )
            self.save_named_matrix(
                self._algorithm.losses,
                self._losses[:, :, 0].T,
                os.path.join(self._disk_log, "fit.losses.sum.csv"),
            )
            self.save_named_matrix(
                self._algorithm.losses,
                self._losses[:, :, 1].T,
                os.path.join(self._disk_log, "fit.losses.count.csv"),
            )
            self.save_named_matrix(
                self._algorithm.metrics,
                self._metrics[:, :, 0].T,
                os.path.join(self._disk_log, "fit.metrics.sum.csv"),
            )
            self.save_named_matrix(
                self._algorithm.metrics,
                self._metrics[:, :, 1].T,
                os.path.join(self._disk_log, "fit.metrics.count.csv"),
            )
        print(
            "+-{:s}-+-{:s}-+-{:s}-+-{:s}-+-{:s}-+".format(
                "-" * maxlen0,
                "-" * maxlen4,
                "-" * maxlen4,
                "-" * maxlen2,
                "-" * maxlen3,
            ),
        )

        #
        (metrics_sum_valid, metrics_count_valid) = self._algorithm.evaluate(self._algorithm.model, self._loader_valid)
        (metrics_sum_test, metrics_count_test) = self._algorithm.evaluate(self._algorithm.model, self._loader_valid)
        self.save_named_matrix(
            self._algorithm.metrics,
            onp.reshape(metrics_sum_valid, (1, len(metrics_sum_valid))),
            os.path.join(self._disk_log, "final.valid.sum.csv"),
        )
        self.save_named_matrix(
            self._algorithm.metrics,
            onp.reshape(metrics_count_valid, (1, len(metrics_count_valid))),
            os.path.join(self._disk_log, "final.valid.count.csv"),
        )
        self.save_named_matrix(
            self._algorithm.metrics,
            onp.reshape(metrics_sum_test, (1, len(metrics_sum_test))),
            os.path.join(self._disk_log, "final.test.sum.csv"),
        )
        self.save_named_matrix(
            self._algorithm.metrics,
            onp.reshape(metrics_count_test, (1, len(metrics_count_test))),
            os.path.join(self._disk_log, "final.test.count.csv"),
        )

    def signal(self: FrameworkFitHMM[AnyData, AnyAlgorithm], improve: bool, /) -> str:
        R"""
        Signal string.

        Args
        ----
        - improve
            If fitting criterion is improved.

        Returns
        -------
        - msg
            Signal message string
        """
        #
        return "     {:s}".format("\033[92m↓\033[0m" if improve else "\033[91m↑\033[0m")

    def save_named_matrix(
        self: FrameworkFitHMM[AnyData, AnyAlgorithm],
        columns: Sequence[str],
        matrix: NPNUMS,
        path: str,
        /,
    ) -> None:
        R"""
        Save named matrix as csv file.

        Args
        ----
        - columns
            Column names.
        - matrix
            Matrix data.
        - path
            Saving path.

        Returns
        -------
        """
        #
        pd.DataFrame(matrix, columns=columns).to_csv(path, index=False)

    def validate_loss(self: FrameworkFitHMM[AnyData, AnyAlgorithm], epoch: int, /) -> None:
        R"""
        Compare losses of every epoch.

        Args
        ----
        - epoch
            Epoch ID.

        Returns
        -------
        """
        #
        assert not self.strict or float(self._losses[0, epoch, 0]) < float(self._losses[0, epoch - 1, 0])
