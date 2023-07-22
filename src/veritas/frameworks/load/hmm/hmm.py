#
from __future__ import annotations


#
import argparse
import abc
import numpy as onp
import more_itertools as xitertools
from typing import Sequence, Generic, TypeVar, Tuple
from ...framework import Framework
from ....datasets import ViewHMM, DataHMM, MetaHMM


#
AnyData = TypeVar("AnyData", bound="DataHMM")


class FrameworkLoadHMM(Framework, Generic[AnyData]):
    R"""
    Framework to load HMM dataset.
    """

    def __init__(self: FrameworkLoadHMM[AnyData], /, *, disk: str, clean: bool) -> None:
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

    def arguments(self: FrameworkLoadHMM[AnyData], /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._parser = argparse.ArgumentParser("Loading HMM Datasets")
        self._parser.add_argument("--suffix", type=str, required=False, default="", help="Saving title suffix.")
        self._parser.add_argument("--dataset", type=str, required=True, help="Dataset directory.")
        self._parser.add_argument("--eq-eps", type=float, required=False, default=1e-5, help="Equality tolerance.")

    def parse(self: FrameworkLoadHMM[AnyData], /) -> None:
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
        self._eq_eps = float(self._args.eq_eps)

    def datasets(self: FrameworkLoadHMM[AnyData], /) -> None:
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
        self._dataset_train_half0 = ViewHMM.from_sections(self._dataset_train, [0, 1, 2, 3])
        self._dataset_train_half1 = ViewHMM.from_sections(self._dataset_train, [4, 5, 6])
        self._dataset_null = ViewHMM.from_sections(self._dataset_full, [])

        #
        assert len(self._dataset_train) + len(self._dataset_valid) == len(self._dataset_tune)
        assert len(self._dataset_tune) + len(self._dataset_test) == len(self._dataset_full)
        assert len(self._dataset_train_half0) + len(self._dataset_train_half1) == len(self._dataset_train)
        assert len(self._dataset_null) == 0

        #
        self._datasets: Sequence[Sequence[Sequence[Tuple[str, MetaHMM]]]]

        #
        self._datasets = [
            [
                [("Train0", self._dataset_train_half0), ("Train1", self._dataset_train_half1)],
                [("Train", self._dataset_train), ("Valid", self._dataset_valid)],
                [("Tune", self._dataset_tune)],
            ],
            [[("Test", self._dataset_test)]],
        ]

    @abc.abstractmethod
    def dataset_full(self: FrameworkLoadHMM[AnyData], /) -> AnyData:
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

    def models(self: FrameworkLoadHMM[AnyData], /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def loaders(self: FrameworkLoadHMM[AnyData], /) -> None:
        R"""
        Prepare loader(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def algorithms(self: FrameworkLoadHMM[AnyData], /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def execute(self: FrameworkLoadHMM[AnyData], /) -> None:
        R"""
        Execute.

        Args
        ----

        Returns
        -------
        """
        # Ensure column definition is safe.
        print("Data Columns:")
        print(self._dataset_full.repr_columns(indent=0))

        # Ensure string and integer indices definition is safe.
        for i in range(len(self._dataset_full)):
            #
            index = self._dataset_full.get_index(i)
            id = self._dataset_full.get_id(index)
            assert self._dataset_full.get_index(id) == index
            assert id == i

        #
        maxlen0 = max(
            xitertools.collapse([[[len(name) for (name, _) in chunk] for chunk in block] for block in self._datasets]),
        )
        maxlen2 = len(str(len(self._dataset_full)))
        maxlen3 = max(maxlen2, 2)
        maxlen1 = maxlen2 + 1 + maxlen3

        #
        print("Multiple Views:")
        for (i, block) in enumerate(self._datasets):
            #
            print("+-{:s}-+-{:s}-+".format("-" * maxlen0, "-" * maxlen1))
            for (j, chunk) in enumerate(block):
                #
                if j > 0:
                    #
                    print("| {:<{:d}s} +-{:s}-+".format("", maxlen0, "-" * maxlen1))
                for (name, meta) in chunk:
                    #
                    if isinstance(meta, ViewHMM):
                        #
                        print(
                            "| {:<{:d}s} | {:>{:d}d}/{:>{:d}d} |".format(
                                name,
                                maxlen0,
                                len(meta),
                                maxlen2,
                                len(meta.data),
                                maxlen3,
                            ),
                        )
                    else:
                        #
                        print(
                            "| {:<{:d}s} | {:>{:d}d}/{:>{:d}s} |".format(
                                name,
                                maxlen0,
                                len(meta),
                                maxlen2,
                                "NA",
                                maxlen3,
                            ),
                        )
        print("+-{:s}-+-{:s}-+".format("-" * maxlen0, "-" * maxlen1))

        # Ensure mapping is correct.
        for block in self._datasets:
            #
            for chunk in block:
                #
                for (_, meta0) in chunk:
                    #
                    if isinstance(meta0, ViewHMM):
                        #
                        meta1 = meta0.data
                    else:
                        #
                        meta1 = self._dataset_full

                    # Ensure later operation to be valid.
                    assert len(meta0) <= len(meta1)

                    # Ensure samples of the same name to be the same.
                    assert all(
                        all(
                            onp.all(ele0 == ele1).item()
                            for (ele0, ele1) in zip(meta0[meta0.get_id(index)], meta1[meta1.get_id(index)])
                        )
                        for index in meta0.indices
                    )
