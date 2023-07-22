#
from __future__ import annotations


#
import numpy as onp
from typing import TypeVar, Sequence, Type, Any
from ...types import NPINTS, NPBOOLS
from .meta import MetaFinite


#
AnySample = TypeVar("AnySample")
SelfViewFinite = TypeVar("SelfViewFinite", bound="ViewFinite[Any]")


class ViewFinite(MetaFinite[AnySample]):
    R"""
    Virtual view of finite samples.
    """

    def __init__(
        self: ViewFinite[AnySample],
        data: MetaFinite[AnySample],
        mappings: NPINTS,
        sections: NPINTS,
        /,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - data
            Real data.
            It can also be a virtual view.
        - mappings
            ID Mappings from virtual ID to real data ID.
        - sections
            Numeric section assignments of all samples.

        Returns
        -------
        """
        #
        self.data = data
        self.mappings = mappings
        self.indices = data.indices[mappings]
        self.sections = sections

        #
        self.ground = self.data
        while isinstance(self.ground, ViewFinite):
            #
            self.ground = self.ground.ground

        #
        self.columns = self.data.columns
        self.types = self.data.types

    @classmethod
    def from_sections(
        cls: Type[SelfViewFinite],
        data: MetaFinite[AnySample],
        takes: Sequence[int],
        /,
    ) -> SelfViewFinite:
        R"""
        Initialize the class from given section IDs.

        Args
        ----
        - data
            Real data.
            It can also be a virtual view.
        - takes
            Section IDs to be taken.

        Returns
        -------
        - self
            Instance.
        """
        #
        return cls.from_masks(data, onp.isin(data.sections, takes))

    @classmethod
    def from_indices(
        cls: Type[SelfViewFinite],
        data: MetaFinite[AnySample],
        takes: Sequence[str],
        /,
    ) -> SelfViewFinite:
        R"""
        Initialize the class from given indices.

        Args
        ----
        - data
            Real data.
            It can also be a virtual view.
        - takes
            Sample indices to be taken.

        Returns
        -------
        - self
            Instance.
        """
        #
        return cls.from_masks(data, onp.isin(data.indices, takes))

    @classmethod
    def from_masks(
        cls: Type[SelfViewFinite],
        data: MetaFinite[AnySample],
        masks: NPBOOLS,
        /,
    ) -> SelfViewFinite:
        R"""
        Initialize the class from sample masks.

        Args
        ----
        - data
            Real data.
            It can also be a virtual view.
        - masks
            Sample masks to be taken.

        Returns
        -------
        - self
            Instance.
        """
        #
        mappings = onp.arange(len(data))[masks]

        #
        if onp.any(masks):
            #
            inverses = {v: i for (i, v) in enumerate(onp.unique(data.sections[masks]).tolist())}
            sections = onp.vectorize(inverses.get)(data.sections[masks])
        else:
            #
            sections = data.sections[masks]
        return cls(data, mappings, sections)

    def __len__(self: ViewFinite[AnySample], /) -> int:
        R"""
        Get length,

        Args
        ----

        Returns
        -------
        - length
            Length.
        """
        #
        return len(self.mappings)
