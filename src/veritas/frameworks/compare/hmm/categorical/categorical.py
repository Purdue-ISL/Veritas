#
from __future__ import annotations


#
from typing import TypeVar, Union
from .....datasets import DataHMMCategorical
from .....algorithms import AlgorithmGradientHMM, AlgorithmConventionHMM
from ..hmm import FrameworkCompareHMM
from ....load.hmm.categorical import load


#
AnyAlgorithmTest = TypeVar("AnyAlgorithmTest", bound="Union[AlgorithmGradientHMM, AlgorithmConventionHMM]")
AnyAlgorithmBase = TypeVar("AnyAlgorithmBase", bound="Union[AlgorithmGradientHMM, AlgorithmConventionHMM]")


class FrameworkCompareHMMCategorical(FrameworkCompareHMM[DataHMMCategorical, AnyAlgorithmTest, AnyAlgorithmBase]):
    R"""
    Framework to compare two algorithms on categorical HMM dataset.
    """

    def dataset_full(
        self: FrameworkCompareHMMCategorical[AnyAlgorithmTest, AnyAlgorithmBase],
        /,
    ) -> DataHMMCategorical:
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
        return load(self._directory_dataset)
