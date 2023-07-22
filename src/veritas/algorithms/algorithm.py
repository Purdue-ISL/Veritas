#
from __future__ import annotations


#
import abc
import torch
from typing import TypeVar, Any, Generic, Sequence, Tuple
from ..models.model import Model
from ..loaders.loader import LoaderFinite
from ..types import NPNUMS, NPFLOATS


#
AnyModel = TypeVar("AnyModel", bound="Model[Any, Any]")
AnyLoaderFinite = TypeVar("AnyLoaderFinite", bound="LoaderFinite")


class Algorithm(abc.ABC, Generic[AnyModel, AnyLoaderFinite]):
    R"""
    Algorithm.
    """

    @abc.abstractmethod
    def fit(
        self: Algorithm[AnyModel, AnyLoaderFinite],
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
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def evaluate(
        self: Algorithm[AnyModel, AnyLoaderFinite],
        model: AnyModel,
        loader: AnyLoaderFinite,
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
        pass

    @abc.abstractmethod
    def details(
        self: Algorithm[AnyModel, AnyLoaderFinite],
        model: AnyModel,
        loader: AnyLoaderFinite,
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
        pass
