#
from __future__ import annotations


#
from typing import Tuple
from ...types import NPRECS
from ..dataset import ViewFinite
from .meta import MetaHMM


#
Sample = Tuple[NPRECS, NPRECS]


class ViewHMM(ViewFinite[Sample], MetaHMM):
    R"""
    Virtual view of hidden Markov processes.
    """

    def __getitem__(self: ViewHMM, id: int, /) -> Sample:
        R"""
        Get item.

        Args
        ----
        - id
            Index.

        Returns
        -------
        - observation
            Observation trace of given index.
        - hidden
            Hidden state trace of given index.
        """
        #
        return self.data[self.mappings[id].item()]
        # \\ id = self.mappings[id].item()
        # \\ if isinstance(self.data, DataHMM):
        # \\     #
        # \\     islice = slice(self.data.starts[id].item(), self.data.ends[id].item())
        # \\     return (self.data.observations[islice], self.data.hiddens[islice])
        # \\ else:
        # \\     #
        # \\     return self.data[id]
