#
from __future__ import annotations


#
import more_itertools as xitertools
from typing import Tuple, Callable
from ...types import NPRECS
from ..dataset import MetaFinite


#
Sample = Tuple[NPRECS, NPRECS]


class MetaHMM(MetaFinite[Sample]):
    R"""
    Data form of hidden Markov processes.
    """

    def set_sections_richness_descent(self: MetaHMM, f: Callable[[Sample], float], num: int, /) -> None:
        R"""
        Allocate section assignments by descending sample information richness.

        Args
        ----
        - f
            Sample information richness evaluation function.
        - num
            Number of sections.

        Returns
        -------
        - sections
            Section assignments.
        """
        #
        scores_with_ids = sorted(
            enumerate([(f(self[i]), -i) for i in range(len(self))]),
            key=lambda it: it[-1],
            reverse=True,
        )
        ids = [id for (id, _) in scores_with_ids]
        for (i, chunk) in enumerate(xitertools.distribute(num, ids)):
            #
            self.sections[list(chunk)] = i
            #
            self.sections[list(chunk)] = i
