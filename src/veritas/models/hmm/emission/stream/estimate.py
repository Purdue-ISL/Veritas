#
from __future__ import annotations


#
import torch
import math
from typing import Tuple, Type, Optional, List, Dict, Sequence
from ....model import Memory
from .....types import THNUMS, THNUMS
from ..emission import ModelEmission
from .._jit.estimate import capacity_to_throughput, num_branches
from ..gaussian import log_prob


#
Inputs = Tuple[THNUMS]
Outputs = Tuple[THNUMS, THNUMS]


class ModelEmissionStreamEstimate(ModelEmission[Inputs, Outputs]):
    R"""
    Video streaming estimation emission distribution model.
    """

    def __init__(
        self: ModelEmissionStreamEstimate,
        num_hiddens: int,
        capunit: float,
        capmin: float,
        transunit: float,
        version: str,
        supports: torch.Tensor,
        /,
        *,
        columns: Sequence[Dict[str, List[str]]],
        dint: Optional[torch.dtype],
        dfloat: Optional[torch.dtype],
        jit: bool,
        vareta: float,
        varinit: float,
        varmax_head: float,
        varmax_rest: float,
        head_by_time: float,
        head_by_chunk: int,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - num_hiddens
            Number of hidden states.
        - capunit
            Capacity unit per hidden state increment.
        - capmin
            Capacity minimum.
            Used to rise up capacity of state zero.
        - transunit
            Transition time unit.
        - version
            Appled estimation function version.
        - supports
            External supporting data for applied estimation function.
        - columns
            Column names for full data.
        - dint
            Integer precision.
        - dfloat
            Floating precision.
        - jit
            If True, use JIT.
        - vareta
            Variance update scale.
        - varinit
            Variance initialzation value.
        - varmax_head
            Variance maximum for heading chunks.
        - varmax_head
            Variance maximum for the rest of chunks.
        - head_by_time
            Heading duration where variance is a special case.
            The union with later argument defines the real heading part.
        - head_by_int
            Heading number of chunks where variance is a special case.
            The union with former argument defines the real heading part.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_hiddens = num_hiddens
        self.capunit = capunit
        self.capmin = capmin
        self.transunit = transunit

        #
        self.varmax: torch.Tensor
        self.varinit: torch.Tensor

        #
        self.vareta = vareta
        self.register_buffer("varmax", torch.tensor([varmax_head, varmax_rest]))
        self.register_buffer(
            "varinit",
            sigmoidrev(
                torch.tensor([float(varinit) * varmax_head / varmax_rest, float(varinit)]) / self.varmax,
            ),
        )

        #
        self.head_by_time = head_by_time
        self.head_by_chunk = head_by_chunk

        #
        observation = columns[0]["float64"]
        self._size = observation.index("size")
        self._trans_time = observation.index("trans_time")
        self._cwnd = observation.index("cwnd")
        self._rtt = observation.index("rtt")
        self._rto = observation.index("rto")
        self._ssthresh = observation.index("ssthresh")
        self._last_snd = observation.index("last_snd")
        self._min_rtt = observation.index("min_rtt")
        self._delivery_rate = observation.index("delivery_rate")
        self._start_time_elapsed = observation.index("start_time_elapsed")
        self._end_time_elapsed = observation.index("end_time_elapsed")

        #
        self._dint = self.DINT if dint is None else dint
        self._dfloat = self.DFLOAT if dfloat is None else dfloat

        #
        self.num_branches = num_branches[version]
        self._vars = torch.nn.Parameter(torch.zeros(2, self.num_branches, dtype=self.dfloat))

        #
        self.counts_: torch.Tensor
        self.weights_: torch.Tensor
        self.sums_: torch.Tensor
        self.squares_: torch.Tensor
        self.means_: torch.Tensor
        self.vars_: torch.Tensor

        #
        self.register_buffer("counts_", torch.zeros(2 * self.num_branches, dtype=torch.long))
        self.register_buffer("weights_", torch.zeros(2 * self.num_branches, dtype=self.dfloat))
        self.register_buffer("sums_", torch.zeros(2 * self.num_branches, dtype=self.dfloat))
        self.register_buffer("squares_", torch.zeros(2 * self.num_branches, dtype=self.dfloat))
        self.register_buffer("means_", torch.zeros(2 * self.num_branches, dtype=self.dfloat))
        self.register_buffer("vars_", torch.zeros(2 * self.num_branches, dtype=self.dfloat))

        #
        self.capacity_to_throughput = capacity_to_throughput[version][jit]

        #
        self.supports: torch.Tensor

        #
        self.register_buffer("supports", torch.zeros(2, dtype=self.dfloat))

        #
        if supports.ndim == 0:
            #
            self.supports[0].fill_(0.0)
            self.supports[1].fill_(1.0)
        else:
            #
            (support_false, support_true) = torch.softmax(supports, dim=0)
            self.supports[0].fill_(support_false / (self.num_branches - 1))
            self.supports[1].fill_(support_true)

    def vars(self: ModelEmissionStreamEstimate, /) -> THNUMS:
        R"""
        Real variance(s).

        Args
        ----

        Returns
        -------
        - vars
            Real variance(s).
        """
        #
        return torch.sigmoid(self._vars) * torch.reshape(self.varmax, (2, 1))

    def reset(self: ModelEmissionStreamEstimate, rng: torch.Generator, /) -> ModelEmissionStreamEstimate:
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
        # Initialize by random distribution.
        self._vars.data[0].fill_(self.varinit[0])
        self._vars.data[1].fill_(self.varinit[1])
        return self

    def forward(self: ModelEmissionStreamEstimate, inputs: Memory, /) -> Memory:
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
        # Parse input memory.
        (observation, _) = inputs

        #
        sizes = observation[:, self._size]
        transition_times = observation[:, self._trans_time]
        congestion_windows = observation[:, self._cwnd].to(self.dint)
        round_trip_times = observation[:, self._rtt]
        retransmisson_timeouts = observation[:, self._rto]
        round_trip_time_mins = observation[:, self._min_rtt]
        slow_start_thresholds = observation[:, self._ssthresh]
        last_sends = observation[:, self._last_snd] * 1000.0
        delivery_rates = observation[:, self._delivery_rate]
        elapsed_times_start = observation[:, self._start_time_elapsed]
        elapsed_times_end = observation[:, self._start_time_elapsed]
        capacities = self.capunit * torch.arange(self.num_hiddens, device=observation.device).to(self.dfloat)
        capacities[0].fill_(self.capmin)

        #
        gaps = chunk_times_to_gaps(elapsed_times_start, elapsed_times_end, self.transunit).to(self.dint)

        # Compute expecting throughput.
        (throughputs_pred, branches) = self.capacity_to_throughput(
            sizes,
            congestion_windows,
            round_trip_times,
            retransmisson_timeouts,
            round_trip_time_mins,
            slow_start_thresholds,
            last_sends,
            delivery_rates,
            capacities,
            self.supports,
        )
        masks_head = torch.reshape(
            torch.logical_or(
                elapsed_times_start < self.head_by_time,
                torch.arange(len(observation), device=observation.device) < self.head_by_chunk,
            ),
            (1, len(observation)),
        )
        branches = (branches + 1) * (2 - masks_head.to(torch.int64)) - 1
        variances = torch.reshape(self.vars(), (2 * self.num_branches,))[branches]

        # Compute observed throughput.
        throughputs_true = sizes * 8.0 / transition_times

        # Compute the observed value distribution by Gaussian with predicted value as mean.
        data = torch.reshape(throughputs_true, (1, len(throughputs_true), 1))
        mean = torch.reshape(throughputs_pred, (*throughputs_pred.shape, 1))
        var = torch.reshape(variances, (*variances.shape, 1))
        probs_log = log_prob(data, mean, var)

        # Force capacities less than true throughput to be impossible.
        filter_capacities = torch.reshape(capacities, (len(capacities), 1))
        filter_throughputs = torch.reshape(throughputs_true, (1, len(observation)))
        filter_throughputs = torch.minimum(filter_throughputs, torch.max(filter_capacities))
        masks_lt_throughput = filter_capacities < filter_throughputs
        assert torch.all(torch.logical_not(torch.all(masks_lt_throughput, dim=0))).item()
        probs_log[masks_lt_throughput] = float("-inf")
        assert torch.all(torch.any(probs_log > float("-inf"), dim=0)).item()
        return [gaps, probs_log]

    @classmethod
    def inputs(cls: Type[ModelEmissionStreamEstimate], memory: Memory, /) -> Inputs:
        R"""
        Decode memory into exact input form.

        Args
        ----
        - memory
            Decoding memory.

        Returns
        -------
        - outputs
            Exact input form.
        """
        #
        (observation,) = memory
        return (observation,)

    @classmethod
    def outputs(cls: Type[ModelEmissionStreamEstimate], memory: Memory, /) -> Outputs:
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
        (gaps, emissions) = memory
        return (gaps, emissions)

    def estimate(self: ModelEmissionStreamEstimate, /) -> None:
        R"""
        Estimation initialization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.counts_.data.zero_()
        self.weights_.data.zero_()
        self.sums_.data.zero_()
        self.squares_.data.zero_()

    def accumulate(self: ModelEmissionStreamEstimate, inputs: Memory, posterior: THNUMS, /) -> None:
        R"""
        Suffcient statistics accumulation for HMM EM algorithm.

        Args
        ----
        - inputs
            Inputs of a sample.
        - posterior
            Posterior estimation of a sample.

        Returns
        -------
        """
        # Parse input memory.
        (observation, _) = inputs

        #
        sizes = observation[:, self._size]
        transition_times = observation[:, self._trans_time]
        congestion_windows = observation[:, self._cwnd].to(self.dint)
        round_trip_times = observation[:, self._rtt]
        retransmisson_timeouts = observation[:, self._rto]
        round_trip_time_mins = observation[:, self._min_rtt]
        slow_start_thresholds = observation[:, self._ssthresh]
        last_sends = observation[:, self._last_snd] * 1000.0
        delivery_rates = observation[:, self._delivery_rate]
        elapsed_times_start = observation[:, self._start_time_elapsed]
        elapsed_times_end = observation[:, self._start_time_elapsed]
        capacities = self.capunit * torch.arange(self.num_hiddens, device=observation.device)
        capacities[0].fill_(self.capmin)

        # Compute expecting throughput.
        (_, branches) = self.capacity_to_throughput(
            sizes,
            congestion_windows,
            round_trip_times,
            retransmisson_timeouts,
            round_trip_time_mins,
            slow_start_thresholds,
            last_sends,
            delivery_rates,
            capacities,
            self.supports,
        )
        masks_head = torch.reshape(
            torch.logical_or(
                elapsed_times_start < self.head_by_time,
                torch.arange(len(observation), device=observation.device) < self.head_by_chunk,
            ),
            (1, len(observation)),
        )
        branches = (branches + 1) * (2 - masks_head.to(torch.int64)) - 1

        # Compute observed throughput.
        branches = torch.reshape(branches, (self.num_hiddens, len(observation)))
        data = torch.reshape(sizes * 8.0 / transition_times, (1, len(observation)))
        weight = torch.reshape(posterior, (self.num_hiddens, len(observation)))

        # Accumulate statistics for corresponding branches.
        length = self.num_hiddens * len(observation)
        indices_ = torch.reshape(branches, (length,))
        weights_ = torch.reshape(weight, (length,))
        sums_ = torch.reshape(weight * data, (length,))
        squares_ = torch.reshape(weight * data**2, (length,))
        self.counts_.index_add_(
            0,
            indices_,
            torch.ones(len(indices_), dtype=self.counts_.dtype, device=self.counts_.device),
        )
        self.weights_.index_add_(0, indices_, weights_)
        self.sums_.index_add_(0, indices_, sums_)
        self.squares_.index_add_(0, indices_, squares_)

    def maximize(self: ModelEmissionStreamEstimate, /) -> None:
        R"""
        Maximization for HMM EM algorithm.

        Args
        ----

        Returns
        -------
        """
        #
        self.means_.copy_(self.sums_ / self.weights_)
        self.vars_.copy_(
            (self.squares_ - 2 * self.sums_ * self.means_ + self.weights_ * self.means_**2) / self.weights_,
        )

    def backward(self: ModelEmissionStreamEstimate, /) -> None:
        R"""
        Translate maximization results for HMM EM algorithm into gradients.

        Args
        ----

        Returns
        -------
        """
        # Zero-out invalid update which is very like caused by no observations.
        masks = torch.isnan(self.vars_)
        self.vars_[masks] = 0.0
        vars = torch.reshape(self._vars, (2, self.num_branches))

        # Gradient should be the difference.
        loss = self.vareta * 0.5 * torch.sum((self.vars() - vars) ** 2)
        loss.backward()


def sigmoidrev(val: torch.Tensor, /) -> torch.Tensor:
    R"""
    Reverse sigmoid.

    Args
    ----
    - val
        Value after sigmoid.

    Returns
    -------
    - val
        Value before sigmoid.
    """
    #
    return torch.log(val / (1.0 - val))


def discretize_chunk_times_start(times_start: THNUMS, unit: float, /) -> THNUMS:
    R"""
    Translate chunk start times to discretized steps.

    Args
    ----
    - times_start
        Start time of each chunk.
    - unit
        Discretized unit.

    Returns
    -------
    - steps_start
        Discretized steps.
    """
    #
    return torch.floor(times_start / unit).long()


def chunk_times_to_gaps(times_start: THNUMS, times_end: THNUMS, unit: float, /) -> THNUMS:
    R"""
    Translate chunk time ranges to discretized gaps between previous and current chunks.

    Args
    ----
    - times_start
        Start time of each chunk.
    - times_end
        End time of each chunk.
    - unit
        Discretized unit.

    Returns
    -------
    - gaps
        Discretized gaps between previous and current chunks.
        The first gap should always be 0, since it has no previous chunk.
    """
    #
    steps_start = discretize_chunk_times_start(times_start, unit)
    gaps = torch.zeros_like(steps_start)
    gaps[1:] = steps_start[1:] - steps_start[:-1]
    return gaps
