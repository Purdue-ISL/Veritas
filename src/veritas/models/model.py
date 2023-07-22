#
from __future__ import annotations

#
import torch
import abc
from typing import TypeVar, Sequence, Generic, Type, Any, Tuple, Dict
from ..types import THNUMS


#
Memory = Sequence[THNUMS]
AnyInputs = TypeVar("AnyInputs", bound="Tuple[THNUMS, ...]")
AnyOutputs = TypeVar("AnyOutputs", bound="Tuple[THNUMS, ...]")
SelfModel = TypeVar("SelfModel", bound="Model[Any, Any]")


class Model(abc.ABC, torch.nn.Module, Generic[AnyInputs, AnyOutputs]):
    R"""
    Model.
    """
    # Default precisions.
    DINT = torch.int64
    DFLOAT = torch.float32

    def __annotate__(self: Model[AnyInputs, AnyOutputs], /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self._dint: torch.dtype
        self._dfloat: torch.dtype

        #
        self.optimizer: torch.optim.Optimizer

    def sgd(self: SelfModel, kwargs: Dict[str, Any]) -> SelfModel:
        R"""
        Optimize model by SGD.

        Args
        ----
        - args
            Keyword arguments for SGD.

        Returns
        -------
        """
        #
        self.optimizer = torch.optim.SGD(self.parameters(), **kwargs)
        return self

    @property
    def devices(self: Model[AnyInputs, AnyOutputs], /) -> Sequence[torch.device]:
        R"""
        Get computation device(s) of the model.

        Args
        ----

        Returns
        -------
        - devices
            All computation device.
        """
        #
        return [torch.device(name) for name in sorted({str(param.device) for param in self.parameters()})]

    @property
    def device(self: Model[AnyInputs, AnyOutputs], /) -> torch.device:
        R"""
        Get the only computation device of the model.

        Args
        ----

        Returns
        -------
        - adevice
            The only computation device.
        """
        #
        devices = self.devices
        if len(devices) == 1:
            #
            (device,) = devices
            return device
        else:
            #
            raise RuntimeError("Expect exactly one device for the model, but get zero or more than one device(s).")

    @property
    def dint(self: Model[AnyInputs, AnyOutputs], /) -> torch.dtype:
        R"""
        Get default integer precision of the model.

        Args
        ----

        Returns
        -------
        - dtype
            Default integer precision.
        """
        #
        return self._dint

    @property
    def dfloat(self: Model[AnyInputs, AnyOutputs], /) -> torch.dtype:
        R"""
        Get default floating precision of the model.

        Args
        ----

        Returns
        -------
        - dtype
            Default floating precision.
        """
        #
        return self._dfloat

    @staticmethod
    def dunique(dtypes: Sequence[torch.dtype], /) -> torch.dtype:
        R"""
        Force dtypes to be unique.

        Args
        ----
        - dtypes
            All dtypes.

        Returns
        -------
        - dtype
            The unique dtype.
        """
        #
        buf = list(set(dtypes))
        if len(buf) == 1:
            #
            (dtype,) = buf
            return dtype
        else:
            #
            raise RuntimeError("Expect only an unique dtype, but get zero or more than one dtype(s).")

    @abc.abstractmethod
    def reset(self: SelfModel, rng: torch.Generator, /) -> SelfModel:
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
        pass

    @abc.abstractmethod
    def forward(self: Model[AnyInputs, AnyOutputs], inputs: Memory, /) -> Memory:
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
        pass

    @classmethod
    @abc.abstractmethod
    def inputs(cls: Type[Model[AnyInputs, AnyOutputs]], memory: Memory, /) -> AnyInputs:
        R"""
        Decode memory into exact input form.

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
        pass

    @classmethod
    @abc.abstractmethod
    def outputs(cls: Type[Model[AnyInputs, AnyOutputs]], memory: Memory, /) -> AnyOutputs:
        R"""
        Decode memory into exact output form.

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
        pass
