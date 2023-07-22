#
from __future__ import annotations


#
import abc
import torch
import numpy as onp
from typing import TypeVar, Any, Tuple, Sequence
from ..models.model import Model
from ..loaders.loader import LoaderFinite
from .algorithm import Algorithm
from ..types import NPFLOATS


#
AnyModel = TypeVar("AnyModel", bound="Model[Any, Any]")
AnyLoaderFinite = TypeVar("AnyLoaderFinite", bound="LoaderFinite")


class AlgorithmGradient(Algorithm[AnyModel, AnyLoaderFinite]):
    R"""
    Gradient-based algorithm.
    """

    def __annotate__(self: AlgorithmGradient[AnyModel, AnyLoaderFinite], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self.losses: Sequence[str]
        self.metrics: Sequence[str]
        self.model: AnyModel

    def fit(
        self: AlgorithmGradient[AnyModel, AnyLoaderFinite],
        loader_train: AnyLoaderFinite,
        loader_valid: AnyLoaderFinite,
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
            (losses[:, epc, 0], losses[:, epc, 1]) = self.train(self.model, loader_train)
            (metrics[:, epc, 0], metrics[:, epc, 1]) = self.evaluate(self.model, loader_valid)
        return (losses, metrics)

    @abc.abstractmethod
    def train(
        self: AlgorithmGradient[AnyModel, AnyLoaderFinite],
        model: AnyModel,
        loader: AnyLoaderFinite,
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
        pass
