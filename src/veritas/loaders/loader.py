#
from __future__ import annotations


#
import abc
from typing import TypeVar, Sequence
from ..types import THNUMS


#
SelfLoaderFinite = TypeVar("SelfLoaderFinite", bound="LoaderFinite")


class LoaderFinite(abc.ABC):
    R"""
    Loading from finite dataset to computation device.
    """

    @abc.abstractmethod
    def transfer(self: LoaderFinite, ids: Sequence[int], /) -> Sequence[THNUMS]:
        R"""
        Transfer samples of given IDs.

        Args
        ----
        - ids
            Sample IDs.

        Returns
        -------
        - memory
            Memory on device(s).
        """
        #
        pass

    @abc.abstractmethod
    def __iter__(self: SelfLoaderFinite, /) -> SelfLoaderFinite:
        R"""
        Get iterator of the class.
        For loader class, the iterator is itself.

        Args
        ----

        Returns
        -------
        - iterator
            Iterator.
        """
        #
        pass

    @abc.abstractmethod
    def __next__(self: LoaderFinite, /) -> Sequence[THNUMS]:
        R"""
        Get next element of an iteration on the class.

        Args
        ----

        Returns
        -------
        - batch
            Data of an iteration.
            Data that is shared among iterations should be included.
        """
        #
        pass
