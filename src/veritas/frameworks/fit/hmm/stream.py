#
from __future__ import annotations


#
import torch
import json
from typing import Tuple
from ....datasets import DataHMMStream
from ....models import (
    ModelHMM,
    ModelInitial,
    ModelInitialGeneric,
    ModelTransition,
    ModelTransitionGeneric,
    ModelTransitionDiagSym,
    ModelTransitionDiagAsym,
    ModelTransitionGaussianSym,
    ModelTransitionGaussianAsym,
    ModelEmission,
    ModelEmissionCategorical,
    ModelEmissionStreamCategoricalPseudo,
    ModelEmissionGaussian,
    ModelEmissionStreamGaussianPseudo,
    ModelEmissionStreamEstimate,
)
from ....models.hmm.emission._jit.estimate import capacity_to_throughput
from ....algorithms import AlgorithmGradientStreamHMM
from .hmm import FrameworkFitHMM
from ....types import THNUMS
from ...load.hmm.stream import load


class FrameworkFitHMMStream(FrameworkFitHMM[DataHMMStream, AlgorithmGradientStreamHMM]):
    R"""
    Framework to fit video streaming HMM dataset.
    """

    def arguments(self: FrameworkFitHMMStream, /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMM.arguments(self)
        self._parser.add_argument("--capacity-max", type=float, required=True, help="Maximum capacity.")
        self._parser.add_argument("--filter-capmax", action="store_true", help="Filter maximum capacity in dataset.")
        self._parser.add_argument("--capacity-unit", type=float, required=True, help="Capacity unit.")
        self._parser.add_argument("--capacity-min", type=float, required=False, default=0.1, help="Minimum capacity.")
        self._parser.add_argument("--transition-unit", type=float, required=True, help="Transition unit time.")
        self._parser.add_argument(
            "--initeta",
            type=float,
            required=True,
            help="Weight for emission initial distribution update.",
        )
        self._parser.add_argument(
            "--transeta",
            type=float,
            required=True,
            help="Weight for emission transition matrix update.",
        )
        self._parser.add_argument("--vareta", type=float, required=True, help="Weight for emission variance update.")
        self._parser.add_argument(
            "--varinit",
            type=float,
            required=True,
            help="Constant initialization for emission variance update.",
        )
        self._parser.add_argument(
            "--varmax-head",
            type=float,
            required=True,
            help="Maximum for heading emission variance update.",
        )
        self._parser.add_argument(
            "--varmax-rest",
            type=float,
            required=True,
            help="Maximum for heading emission variance update.",
        )
        self._parser.add_argument(
            "--head-by-time",
            type=float,
            required=True,
            help="Heading variance time (second).",
        )
        self._parser.add_argument(
            "--head-by-chunk",
            type=int,
            required=True,
            help="Heading variance chunks.",
        )
        self._parser.add_argument(
            "--transextra",
            type=int,
            required=False,
            default=0,
            help="Estimation transition extra arguments.",
        )
        self._parser.add_argument(
            "--include-beyond",
            action="store_true",
            help="Include target states which is invalid in estimation transition construction.",
        )
        self._parser.add_argument(
            "--support",
            type=str,
            required=False,
            default="",
            help="Path to external supporting data for estimation.",
        )

    def parse(self: FrameworkFitHMMStream, /) -> None:
        R"""
        Parse argument(s) from given command(s).

        Args
        ----

        Returns
        -------
        """
        #
        FrameworkFitHMM.parse(self)
        self._capmax = float(self._args.capacity_max)
        self._filter_capmax = bool(self._args.filter_capmax)
        self._capunit = float(self._args.capacity_unit)
        self._capmin = float(self._args.capacity_min)
        self._transunit = float(self._args.transition_unit)
        self._initeta = float(self._args.initeta)
        self._transeta = float(self._args.transeta)
        self._vareta = float(self._args.vareta)
        self._varinit = float(self._args.varinit)
        self._varmax_head = float(self._args.varmax_head)
        self._varmax_rest = float(self._args.varmax_rest)
        self._head_by_time = float(self._args.head_by_time)
        self._head_by_chunk = int(self._args.head_by_chunk)
        self._transextra = int(self._args.transextra)
        self._include_beyond = bool(self._args.include_beyond)
        self._support = str(self._args.support)

        #
        assert self._capmin < self._capunit, "Minimum capacity should be strictly smaller than capacity unit."

    def dataset_full(self: FrameworkFitHMMStream, /) -> DataHMMStream:
        R"""
        Get full HMM dataset.

        Args
        ----

        Returns
        -------
        - dataset
            HMM dataset.
        """
        #
        return load(self._directory_dataset, self._capmax if self._filter_capmax else float("inf"))

    def models(self: FrameworkFitHMMStream, /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._num_hiddens = 1 + int(round(self._capmax / self._capunit))

        #
        initial: ModelInitial[Tuple[()]]
        transition: ModelTransition[Tuple[()]]
        emission: ModelEmission[Tuple[THNUMS], Tuple[THNUMS, THNUMS]]

        #
        if self._initial == "generic":
            #
            initial = ModelInitialGeneric(self._num_hiddens, dint=None, dfloat=torch.float64, initeta=self._initeta)
        else:
            #
            raise RuntimeError('Unknown video streaming initial model "{:s}".'.format(self._initial))

        #
        if self._transition == "generic":
            #
            transition = ModelTransitionGeneric(
                self._num_hiddens,
                self._smooth,
                dint=None,
                dfloat=torch.float64,
                transeta=self._transeta,
            )
        elif self._transition == "diag.sym":
            #
            transition = ModelTransitionDiagSym(
                self._num_hiddens,
                self._transextra,
                self._include_beyond,
                self._smooth,
                dint=None,
                dfloat=torch.float64,
                transeta=self._transeta,
            )
        elif self._transition == "diag.asym":
            #
            transition = ModelTransitionDiagAsym(
                self._num_hiddens,
                self._transextra,
                self._include_beyond,
                self._smooth,
                dint=None,
                dfloat=torch.float64,
                transeta=self._transeta,
            )
        elif self._transition == "gaussian.sym":
            #
            transition = ModelTransitionGaussianSym(
                self._num_hiddens,
                self._transextra,
                self._include_beyond,
                self._smooth,
                dint=None,
                dfloat=torch.float64,
                transeta=self._transeta,
            )
        elif self._transition == "gaussian.asym":
            #
            transition = ModelTransitionGaussianAsym(
                self._num_hiddens,
                self._transextra,
                self._include_beyond,
                self._smooth,
                dint=None,
                dfloat=torch.float64,
                transeta=self._transeta,
            )
        else:
            #
            raise RuntimeError('Unknown video streaming transition model "{:s}".'.format(self._transition))

        #
        if self._emission == "categorical":
            #
            emission = ModelEmissionStreamCategoricalPseudo(
                ModelEmissionCategorical(
                    self._num_hiddens,
                    self._num_hiddens,
                    dint=None,
                    dfloat=torch.float64,
                    emiteta=1.0,
                ),
            )
        elif self._emission == "gaussian":
            #
            emission = ModelEmissionStreamGaussianPseudo(
                ModelEmissionGaussian(self._num_hiddens, 1, dint=None, dfloat=torch.float64, emiteta=1.0),
            )
        elif self._emission in capacity_to_throughput.keys():
            # Load external supporting data.
            if len(self._support) > 0:
                #
                with open(self._support, "r") as file:
                    #
                    supports = torch.tensor(json.load(file))
            else:
                #
                supports = torch.tensor(0.0)

            #
            emission = ModelEmissionStreamEstimate(
                self._num_hiddens,
                self._capunit,
                self._capmin,
                self._transunit,
                self._emission,
                supports,
                columns=self._dataset_tune.columns,
                dint=None,
                dfloat=torch.float64,
                jit=self._jit,
                vareta=self._vareta,
                varinit=self._varinit,
                varmax_head=self._varmax_head,
                varmax_rest=self._varmax_rest,
                head_by_time=self._head_by_time,
                head_by_chunk=self._head_by_chunk,
            )
        else:
            #
            raise RuntimeError('Unknown video streaming emission model "{:s}".'.format(self._emission))

        #
        thrng = torch.Generator("cpu").manual_seed(self._seed)
        self._model = ModelHMM(initial, transition, emission).reset(thrng).to(self._device).sgd(dict(lr=1.0))

    def algorithms(self: FrameworkFitHMMStream, /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm = AlgorithmGradientStreamHMM(self._model, jit=self._jit)
