#
from __future__ import annotations


#
import torch
from typing import Tuple
from ....datasets import DataHMMGaussian
from ....models import (
    ModelHMM,
    ModelInitial,
    ModelInitialGeneric,
    ModelTransitionGeneric,
    ModelEmissionGaussian,
)
from ....algorithms import AlgorithmGradientHMM
from .hmm import FrameworkFitHMM
from ....types import THNUMS
from ...load.hmm.gaussian import load


class FrameworkFitHMMGaussian(FrameworkFitHMM[DataHMMGaussian, AlgorithmGradientHMM]):
    R"""
    Framework to fit categorical HMM dataset.
    """

    def dataset_full(self: FrameworkFitHMMGaussian, /) -> DataHMMGaussian:
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

    def models(self: FrameworkFitHMMGaussian, /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._num_hiddens = self._dataset_tune.num_hiddens
        self._num_features = self._dataset_tune.num_features

        #
        initial: ModelInitial[Tuple[()]]

        #
        if self._initial == "generic":
            #
            initial = ModelInitialGeneric(self._num_hiddens, dint=None, dfloat=torch.float64, initeta=1.0)
        else:
            #
            raise RuntimeError('Unknown Gaussian initial model "{:s}".'.format(self._initial))

        #
        if self._transition == "generic":
            #
            transition = ModelTransitionGeneric(
                self._num_hiddens,
                self._smooth,
                dint=None,
                dfloat=torch.float64,
                transeta=1.0,
            )
        else:
            #
            raise RuntimeError('Unknown Gaussian transition model "{:s}".'.format(self._transition))

        #
        if self._emission == "gaussian":
            #
            emission = ModelEmissionGaussian(
                self._num_hiddens,
                self._num_features,
                dint=None,
                dfloat=torch.float64,
                emiteta=1.0,
            )
        else:
            #
            raise RuntimeError('Unknown Gaussian emission model "{:s}".'.format(self._emission))

        #
        thrng = torch.Generator("cpu").manual_seed(self._seed)
        self._model = ModelHMM(initial, transition, emission).reset(thrng).to(self._device).sgd(dict(lr=1.0))

    def algorithms(self: FrameworkFitHMMGaussian, /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm = AlgorithmGradientHMM(self._model, jit=self._jit)
