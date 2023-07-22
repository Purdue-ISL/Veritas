#
from __future__ import annotations

#
import torch
import more_itertools as xitertools
import numpy as onp
from typing import Sequence, List
from .loader import LoaderFinite
from ..datasets import MetaHMM
from ..models import ModelHMM
from ..types import THNUMS


class LoaderHMM(LoaderFinite):
    R"""
    Loading from dataset to computation device for HMM.
    """

    def __init__(self: LoaderHMM, dataset: MetaHMM, model: ModelHMM, /) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - dataset
            Dataset to load from.
        - model
            Model to load for.

        Returns
        -------
        """
        #
        self.dataset = dataset
        self.model = model
        self.device = self.model.device

        #
        self.mappings = {"int64": self.model.dint, "float64": self.model.dfloat}
        self.blocks = [
            [cols for (dtype, cols) in chunk.items() if dtype in self.mappings and len(cols) > 0]
            for chunk in self.dataset.columns
        ]
        self.columns = list(xitertools.flatten(xitertools.flatten(self.blocks)))
        self.nblocks = xitertools.ilen(xitertools.flatten(self.blocks))
        self.ncols = len(self.columns)

        # Transfer full dataset to computation device memory.
        self.memory = self.transfer(list(range(len(self.dataset))))

    def transfer(self: LoaderHMM, ids: Sequence[int], /) -> Sequence[THNUMS]:
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
        blocks: Sequence[List[THNUMS]]

        # Collect essential data for all samples.
        length = len(ids)
        blocks = [[] for _ in range(self.nblocks)]
        for i in ids:
            #
            (records_observation, records_hidden) = self.dataset[i]
            j = 0
            for records in (records_observation, records_hidden):
                #
                if records.dtype.names is not None:
                    #
                    for name in records.dtype.names:
                        #
                        array = records[name]
                        dtype = str(array.dtype)
                        if array.size > 0 and dtype in self.mappings:
                            #
                            tensor = torch.from_numpy(array).to(self.mappings[dtype])
                            blocks[j].append(tensor)
                            j += 1

        #
        sizes = onp.array([[len(blocks[j][i]) for j in range(self.nblocks)] for i in range(length)]).T
        bounds = onp.zeros((self.nblocks, 1 + length), dtype=sizes.dtype)
        onp.cumsum(sizes, axis=1, out=bounds[:, 1:])
        ends = bounds[:, 1:]
        begins = bounds[:, :-1]
        pointers = onp.stack((begins.T, sizes.T, ends.T))

        #
        assert all(
            len(block) == length for block in blocks
        ), "One loaded memory block does not contain the same number of samples as dataset."

        #
        memory = [torch.cat(block).to(self.device) for block in blocks]
        memory = [torch.from_numpy(pointers).to(self.model.dint).to(self.device)] + memory
        return memory

    def __iter__(self: LoaderHMM, /) -> LoaderHMM:
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
        self._iter_ptr = 0
        self._iter_max = 1
        return self

    def __next__(self: LoaderHMM, /) -> Sequence[THNUMS]:
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
        if self._iter_ptr < self._iter_max:
            #
            self._iter_ptr += 1
            return self.memory
        raise StopIteration
