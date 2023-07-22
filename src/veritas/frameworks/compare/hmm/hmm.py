#
from __future__ import annotations


#
import argparse
import abc
import time
from typing import Generic, TypeVar, Union
from ...framework import Framework
from ....datasets import ViewHMM, DataHMM
from ....models import ModelHMM
from ....loaders import LoaderHMM
from ....algorithms import AlgorithmGradientHMM, AlgorithmConventionHMM


#
AnyData = TypeVar("AnyData", bound="DataHMM")
AnyAlgorithmTest = TypeVar("AnyAlgorithmTest", bound="Union[AlgorithmGradientHMM, AlgorithmConventionHMM]")
AnyAlgorithmBase = TypeVar("AnyAlgorithmBase", bound="Union[AlgorithmGradientHMM, AlgorithmConventionHMM]")


class FrameworkCompareHMM(Framework, Generic[AnyData, AnyAlgorithmTest, AnyAlgorithmBase]):
    R"""
    Framework to compare two algorithms on HMM dataset.
    """

    def __annotate__(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm_test: AnyAlgorithmTest
        self._algorithm_base: AnyAlgorithmBase

        #
        self._model_test: ModelHMM

        #
        Framework.__annotate__(self)

    def __init__(
        self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase],
        /,
        *,
        disk: str,
        clean: bool,
    ) -> None:
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

    def arguments(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._parser = argparse.ArgumentParser("Comparing 2 HMM Algorithms")
        self._parser.add_argument("--suffix", type=str, required=False, default="", help="Saving title suffix.")
        self._parser.add_argument("--dataset", type=str, required=True, help="Dataset directory.")
        self._parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed.")
        self._parser.add_argument("--device", type=str, required=True, help="Computation device.")
        self._parser.add_argument("--jit", action="store_true", help="Enable JIT.")
        self._parser.add_argument("--num-epochs", type=int, required=True, help="Number of epochs.")
        self._parser.add_argument("--eq-eps", type=float, required=False, default=1e-5, help="Equality tolerance.")

    def parse(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
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
        self._seed = int(self._args.seed)
        self._device = str(self._args.device)
        self._jit = bool(self._args.jit)
        self._num_epochs = int(self._args.num_epochs)
        self._eq_eps = float(self._args.eq_eps)

    def datasets(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Prepare dataset(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._dataset_full = self.dataset_full()
        self._dataset_full.set_sections_richness_descent(self._dataset_full.richness, 10)
        self._dataset_tune = self._dataset_full.copy_sections([0, 1, 2, 3, 4, 5, 6, 7])
        self._dataset_test = self._dataset_full.copy_sections([8, 9])
        self._dataset_train = ViewHMM.from_sections(self._dataset_tune, [0, 1, 2, 3, 4, 5, 6])
        self._dataset_valid = ViewHMM.from_sections(self._dataset_tune, [7])

        #
        assert len(self._dataset_train) + len(self._dataset_valid) == len(self._dataset_tune)
        assert len(self._dataset_tune) + len(self._dataset_test) == len(self._dataset_full)

    @abc.abstractmethod
    def dataset_full(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> AnyData:
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

    def loaders(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Prepare loader(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._loader_train = LoaderHMM(self._dataset_train, self._model_test)
        self._loader_valid = LoaderHMM(self._dataset_valid, self._model_test)
        self._loader_test = LoaderHMM(self._dataset_test, self._model_test)

    def execute(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Execute.

        Args
        ----

        Returns
        -------
        """
        #
        maxlen0 = 5
        maxlen1 = 23
        maxlen2 = 23
        maxlen3 = 10

        # Ensure our implementation performs close to library implementation.
        print("+-{:s}-+-{:s}-+-{:s}-+".format("-" * maxlen0, "-" * maxlen1, "-" * maxlen2))
        print("| {:>{:d}s} | {:>{:d}s} | {:>{:d}s} |".format("", maxlen0, "NLL.Sum", maxlen1, "Time.Sec", maxlen2))
        print(
            "| {:>{:d}s} +-{:s}-+-{:s}-+-{:s}-+-{:s}-+".format(
                "Epoch",
                maxlen0,
                "-" * maxlen3,
                "-" * maxlen3,
                "-" * maxlen3,
                "-" * maxlen3,
            ),
        )
        print(
            "| {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} |".format(
                "",
                maxlen0,
                "Testing",
                maxlen3,
                "Baseline",
                maxlen3,
                "Testing",
                maxlen3,
                "Baseline",
                maxlen3,
            ),
        )
        print(
            "+-{:s}-+-{:s}-+-{:s}-+-{:s}-+-{:s}-+".format(
                "-" * maxlen0,
                "-" * maxlen3,
                "-" * maxlen3,
                "-" * maxlen3,
                "-" * maxlen3,
            ),
        )
        for i in range(1, self._num_epochs + 1):
            #
            self.communicate()

            #
            time_start = time.time()
            (self._losses_test, self._metrics_test) = self._algorithm_test.fit(
                self._loader_train,
                self._loader_valid,
                num_epochs=1,
            )
            time_end = time.time()
            nllsum_test = self._losses_test[0, -1, 0].item()
            elapsed_test = time_end - time_start

            #
            time_start = time.time()
            (self._losses_base, self._metrics_base) = self._algorithm_base.fit(
                self._loader_train,
                self._loader_valid,
                num_epochs=1,
            )
            time_end = time.time()
            nllsum_base = self._losses_base[0, -1, 0].item()
            elapsed_base = time_end - time_start

            #
            print(
                "| {:>{:d}d} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} | {:>{:d}s} |".format(
                    i,
                    maxlen0,
                    "{:.2f}".format(nllsum_test),
                    maxlen3,
                    "{:.2f}".format(nllsum_base),
                    maxlen3,
                    "{:.2f}".format(elapsed_test),
                    maxlen3,
                    "{:.2f}".format(elapsed_base),
                    maxlen3,
                )
            )

            #
            self.validate_loss()
            self.validate_metric()
            self.validate_parameter()
        print(
            "+-{:s}-+-{:s}-+-{:s}-+-{:s}-+-{:s}-+".format(
                "-" * maxlen0,
                "-" * maxlen3,
                "-" * maxlen3,
                "-" * maxlen3,
                "-" * maxlen3,
            ),
        )

        #
        self.validate_final()

    @abc.abstractmethod
    def communicate(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Communicate between two algorithms.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def validate_loss(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Compare losses of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def validate_metric(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Compare metrics of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def validate_parameter(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Compare parameters of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def validate_final(self: FrameworkCompareHMM[AnyData, AnyAlgorithmTest, AnyAlgorithmBase], /) -> None:
        R"""
        Compare final operation(s) after tuning.

        Args
        ----

        Returns
        -------
        """
