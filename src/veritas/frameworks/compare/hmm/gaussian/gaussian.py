#
from __future__ import annotations


#
from typing import TypeVar, Union
from .....datasets import DataHMMGaussian
from .....algorithms import AlgorithmGradientHMM, AlgorithmConventionHMM
from ..hmm import FrameworkCompareHMM
from ....load.hmm.gaussian import load


#
AnyAlgorithmTest = TypeVar("AnyAlgorithmTest", bound="Union[AlgorithmGradientHMM, AlgorithmConventionHMM]")
AnyAlgorithmBase = TypeVar("AnyAlgorithmBase", bound="Union[AlgorithmGradientHMM, AlgorithmConventionHMM]")


class FrameworkCompareHMMGaussian(FrameworkCompareHMM[DataHMMGaussian, AnyAlgorithmTest, AnyAlgorithmBase]):
    R"""
    Framework to compare two algorithms on Gaussian HMM dataset.
    """

    def dataset_full(
        self: FrameworkCompareHMMGaussian[AnyAlgorithmTest, AnyAlgorithmBase],
        /,
    ) -> DataHMMGaussian:
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
