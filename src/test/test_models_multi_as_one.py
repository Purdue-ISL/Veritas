#
from __future__ import annotations


#
import pytest
import torch
from typing import Sequence, Tuple, Type
from veritas.models import Model
from veritas.models.model import Memory


class ModelMultiDevices(Model[Tuple[()], Tuple[()]]):
    R"""
    A model with multiple devices.
    """

    def __init__(self: ModelMultiDevices, devices: Sequence[torch.device], /) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - devices
            Multiple devices.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        buf = []
        for device in devices:
            #
            buf.append(torch.nn.Parameter(torch.zeros(3, 3, device=device)))
        self.tensors = torch.nn.ParameterList(buf)

    def reset(self: ModelMultiDevices, rng: torch.Generator, /) -> ModelMultiDevices:
        R"""
        Reset parameters.

        Args
        ----
        - rng
            Random state.

        Returns
        -------
        - self
            Instance itself.
        """
        #
        raise NotImplementedError

    def forward(self: ModelMultiDevices, inputs: Memory, /) -> Memory:
        R"""
        Forward.

        Args
        ----
        - inputs
            Input memory.

        Returns
        -------
        - outputs
            Output memory.
        """
        #
        raise NotImplementedError

    @classmethod
    def inputs(cls: Type[ModelMultiDevices], memory: Memory, /) -> Tuple[()]:
        R"""
        Decoder memory into exact input form.

        Args
        ----
        - memory
            Decoding memory.

        Returns
        -------
        - inputs
            Exact input form.
        """
        #
        raise NotImplementedError

    @classmethod
    def outputs(cls: Type[ModelMultiDevices], memory: Memory, /) -> Tuple[()]:
        R"""
        Decoder memory into exact output form.

        Args
        ----
        - memory
            Decoding memory.

        Returns
        -------
        - outputs
            Exact output form.
        """
        #
        raise NotImplementedError


@pytest.mark.skipif(not torch.cuda.is_available(), reason="at least one GPU is required.")
@pytest.mark.xfail
def test_device() -> None:
    R"""
    Test corner case forcing multiple devices into one device.

    Args
    ----

    Returns
    -------
    """
    #
    model = ModelMultiDevices([torch.device("cpu"), torch.device("cuda")])
    model.device


@pytest.mark.xfail
def test_dtype() -> None:
    R"""
    Test corner case forcing multiple dtypes into one dtype.

    Args
    ----

    Returns
    -------
    """
    #
    Model.dunique([torch.float32, torch.float64])


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for test_corner in (test_device, test_dtype):
        #
        try:
            #
            test_corner()
        except RuntimeError:
            #
            pass


#
if __name__ == "__main__":
    #
    main()
