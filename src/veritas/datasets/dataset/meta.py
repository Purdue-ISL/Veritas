#
from __future__ import annotations


#
import abc
import numpy as onp
from typing import TypeVar, Sequence, Dict, cast, Generic, List
from ...types import NPINTS, NPSTRS, NPRECS


#
AnySample = TypeVar("AnySample")


class MetaFinite(abc.ABC, Generic[AnySample]):
    R"""
    Data form of finite samples.
    """

    def __annotate__(self: MetaFinite[AnySample], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self.columns: Sequence[Dict[str, List[str]]]
        self.types: Dict[str, List[str]]
        self.indices: NPSTRS
        self.sections: NPINTS
        self.memory: Sequence[NPRECS]

    @abc.abstractmethod
    def __getitem__(self: MetaFinite[AnySample], id: int, /) -> AnySample:
        R"""
        Get item.

        Args
        ----
        - id
            Index.

        Returns
        -------
        - sample
            A sample.
        """
        #
        pass

    def get_index(self: MetaFinite[AnySample], id: int, /) -> str:
        R"""
        Get raw data index of item .

        Args
        ----
        - id
            Numeric index.

        Returns
        -------
        - index
            Raw data index.
        """
        #
        return cast(str, self.indices[id].item())

    def get_id(self: MetaFinite[AnySample], index: str, /) -> int:
        R"""
        Get item ID of raw data index.

        Args
        ----
        - index
            Raw data index.

        Returns
        -------
        - id
            Item ID.
        """
        #
        return onp.where(self.indices == index)[0].item()

    @abc.abstractmethod
    def __len__(self: MetaFinite[AnySample], /) -> int:
        R"""
        Get length.

        Args
        ----

        Returns
        -------
        - length
            Length.
        """
        #
        pass
