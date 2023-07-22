#
import os
import torch
from typing import Tuple
import sys
from src.veritas.frameworks import FrameworkTransformHMMStream
from src.veritas.models.hmm.emission import register


def capacity_to_throughput_v3_(
    sizes: torch.Tensor,
    congestion_windows: torch.Tensor,
    round_trip_times: torch.Tensor,
    retransmisson_timeouts: torch.Tensor,
    round_trip_time_mins: torch.Tensor,
    slow_start_thresholds: torch.Tensor,
    last_sends: torch.Tensor,
    delivery_rates: torch.Tensor,
    capacities: torch.Tensor,
    supports: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Get expected throughput given capacity (version 0).

    Args
    ----
    - sizes
        Size.
        Not multiplied by 8 (unit is still byte).
    - congestion_windows
        Congestion window.
    - retransmission_timeouts
        Retransmission timeout.
    - round_trip_time_mins
        Round trip time minimum.
    - slow_start_thresholds
        Slow start threshold.
    - last_sends
        Last send.
        Multiplied by 1000 (unit changes from second to millisecond).
    - delivery_rates
        Delivery rates.
    - capacities
        Given capacity (bandwidth).
    - supports
        External supporting data.

    Returns
    -------
    - througputs
        Throughput.
    - branches
        Computation branch.
    """
    #
    num_states = len(capacities)
    num_steps = max(
        len(sizes),
        len(congestion_windows),
        len(round_trip_times),
        len(retransmisson_timeouts),
        len(round_trip_time_mins),
        len(slow_start_thresholds),
        len(last_sends),
        len(delivery_rates),
    )
    throughputs = torch.zeros(num_states, num_steps, dtype=capacities.dtype, device=capacities.device)
    branches = torch.zeros(num_states, num_steps, dtype=torch.long, device=capacities.device)

    # Scalar buffers.
    bdp = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)
    bdp_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    data_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    sent = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    round = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    cwnd = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    ssh_thresh = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)

    #
    sizes_bit = sizes * 8.0
    for state in range(num_states):
        #
        for step in range(num_steps):
            #
            bdp.fill_(capacities[state] * round_trip_times[step] / 8.0)
            bdp_segments.fill_(torch.round(bdp * 1000.0 / 1448.0))
            data_segments.fill_(torch.round(sizes[step] * 1000.0 / 1448.0))

            #
            if congestion_windows[step] > bdp_segments:
                #
                if data_segments > bdp_segments:
                    #
                    if capacities[state] == 0.0:
                        #
                        throughputs[state, step] = 0.0
                        branches[state, step] = 0
                    else:
                        #
                        throughputs[state, step] = sizes_bit[step] / (
                            sizes_bit[step] / capacities[state] + round_trip_times[step]
                        )
                        branches[state, step] = 1
                else:
                    #
                    throughputs[state, step] = sizes_bit[step] / (round_trip_times[step])
                    branches[state, step] = 2
            else:
                #
                sent.fill_(0)
                round.fill_(0)
                cwnd.copy_(congestion_windows[step].long())
                ssh_thresh.copy_(slow_start_thresholds[step])
                while sent < data_segments:
                    #
                    sent.add_(torch.minimum(cwnd, bdp_segments))
                    cwnd.add_(1)
                    round.add_(1)
                throughputs[state, step] = sizes_bit[step] / (round * round_trip_times[step])
                branches[state, step] = 3
    return (throughputs, branches)

def capacity_to_throughput_v7_(
    sizes: torch.Tensor,
    congestion_windows: torch.Tensor,
    round_trip_times: torch.Tensor,
    retransmisson_timeouts: torch.Tensor,
    round_trip_time_mins: torch.Tensor,
    slow_start_thresholds: torch.Tensor,
    last_sends: torch.Tensor,
    delivery_rates: torch.Tensor,
    capacities: torch.Tensor,
    supports: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Get expected throughput given capacity (version 0).

    Args
    ----
    - sizes
        Size.
        Not multiplied by 8 (unit is still byte).
    - congestion_windows
        Congestion window.
    - retransmission_timeouts
        Retransmission timeout.
    - round_trip_time_mins
        Round trip time minimum.
    - slow_start_thresholds
        Slow start threshold.
    - last_sends
        Last send.
        Multiplied by 1000 (unit changes from second to millisecond).
    - delivery_rates
        Delivery rates.
    - capacities
        Given capacity (bandwidth).
    - supports
        External supporting data.

    Returns
    -------
    - througputs
        Throughput.
    - branches
        Computation branch.
    """
    #
    num_states = len(capacities)
    num_steps = max(
        len(sizes),
        len(congestion_windows),
        len(round_trip_times),
        len(retransmisson_timeouts),
        len(round_trip_time_mins),
        len(slow_start_thresholds),
        len(last_sends),
        len(delivery_rates),
    )
    throughputs = torch.zeros(num_states, num_steps, dtype=capacities.dtype, device=capacities.device)
    branches = torch.zeros(num_states, num_steps, dtype=torch.long, device=capacities.device)

    # Scalar buffers.
    bdp = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)
    bdp_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    data_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    sent = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    round = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    cwnd = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    ssh_thresh = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)

    #
    sizes_bit = sizes * 8.0
    for state in range(num_states):
        #
        for step in range(num_steps):
            #
            bdp.fill_(capacities[state] * round_trip_times[step] / 8.0)
            bdp_segments.fill_(torch.round(bdp * 1000.0 / 1448.0))
            data_segments.fill_(torch.round(sizes[step] * 1000.0 / 1448.0))

            if data_segments > bdp_segments:
                #
                if capacities[state] == 0.0:
                    #
                    throughputs[state, step] = 0.0
                    branches[state, step] = 0
                else:
                    #
                    throughputs[state, step] = sizes_bit[step] / (
                        sizes_bit[step] / capacities[state] + round_trip_times[step]
                    )
                    branches[state, step] = 1
            else:
                #
                throughputs[state, step] = sizes_bit[step] / (round_trip_times[step])
                branches[state, step] = 2

    return (throughputs, branches)

def capacity_to_throughput_v10_(
    sizes: torch.Tensor,
    congestion_windows: torch.Tensor,
    round_trip_times: torch.Tensor,
    retransmisson_timeouts: torch.Tensor,
    round_trip_time_mins: torch.Tensor,
    slow_start_thresholds: torch.Tensor,
    last_sends: torch.Tensor,
    delivery_rates: torch.Tensor,
    capacities: torch.Tensor,
    supports: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Get expected throughput given capacity (version 0).

    Args
    ----
    - sizes
        Size.
        Not multiplied by 8 (unit is still byte).
    - congestion_windows
        Congestion window.
    - retransmission_timeouts
        Retransmission timeout.
    - round_trip_time_mins
        Round trip time minimum.
    - slow_start_thresholds
        Slow start threshold.
    - last_sends
        Last send.
        Multiplied by 1000 (unit changes from second to millisecond).
    - delivery_rates
        Delivery rates.
    - capacities
        Given capacity (bandwidth).
    - supports
        External supporting data.

    Returns
    -------
    - througputs
        Throughput.
    - branches
        Computation branch.
    """
    #
    num_states = len(capacities)
    num_steps = max(
        len(sizes),
        len(congestion_windows),
        len(round_trip_times),
        len(retransmisson_timeouts),
        len(round_trip_time_mins),
        len(slow_start_thresholds),
        len(last_sends),
        len(delivery_rates),
    )
    throughputs = torch.zeros(num_states, num_steps, dtype=capacities.dtype, device=capacities.device)
    branches = torch.zeros(num_states, num_steps, dtype=torch.long, device=capacities.device)

    # Scalar buffers.
    bdp = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)
    bdp_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    data_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    sent = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    round = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    cwnd = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    ssh_thresh = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)

    #
    sizes_bit = sizes * 8.0
    for state in range(num_states):
        #
        for step in range(num_steps):
            #
            bdp.fill_(capacities[state] * round_trip_time_mins[step] / 8.0)
            bdp_segments.fill_(torch.round(bdp * 1000.0 / 1448.0))
            data_segments.fill_(torch.round(sizes[step] * 1000.0 / 1448.0))

            #
            if congestion_windows[step] > bdp_segments:
                #
                if data_segments > bdp_segments:
                    #
                    if capacities[state] == 0.0:
                        #
                        throughputs[state, step] = 0.0
                        branches[state, step] = 0
                    else:
                        #
                        throughputs[state, step] = sizes_bit[step] / (
                            sizes_bit[step] / capacities[state] + round_trip_time_mins[step]
                        )
                        branches[state, step] = 1
                else:
                    #
                    throughputs[state, step] = sizes_bit[step] / (round_trip_time_mins[step])
                    branches[state, step] = 2
            else:
                #
                sent.fill_(0)
                round.fill_(0)
                cwnd.copy_(congestion_windows[step].long())
                ssh_thresh.copy_(slow_start_thresholds[step])
                while sent < data_segments:
                    #
                    sent.add_(torch.minimum(cwnd, bdp_segments))
                    cwnd.add_(1)
                    round.add_(1)
                throughputs[state, step] = sizes_bit[step] / (round * round_trip_time_mins[step])
                branches[state, step] = 3
    return (throughputs, branches)

def capacity_to_throughput_v17_(
    sizes: torch.Tensor,
    congestion_windows: torch.Tensor,
    round_trip_times: torch.Tensor,
    retransmisson_timeouts: torch.Tensor,
    round_trip_time_mins: torch.Tensor,
    slow_start_thresholds: torch.Tensor,
    last_sends: torch.Tensor,
    delivery_rates: torch.Tensor,
    capacities: torch.Tensor,
    supports: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Get expected throughput given capacity (version 0).

    Args
    ----
    - sizes
        Size.
        Not multiplied by 8 (unit is still byte).
    - congestion_windows
        Congestion window.
    - retransmission_timeouts
        Retransmission timeout.
    - round_trip_time_mins
        Round trip time minimum.
    - slow_start_thresholds
        Slow start threshold.
    - last_sends
        Last send.
        Multiplied by 1000 (unit changes from second to millisecond).
    - delivery_rates
        Delivery rates.
    - capacities
        Given capacity (bandwidth).
    - supports
        External supporting data.

    Returns
    -------
    - througputs
        Throughput.
    - branches
        Computation branch.
    """
    #
    num_states = len(capacities)
    num_steps = max(
        len(sizes),
        len(congestion_windows),
        len(round_trip_times),
        len(retransmisson_timeouts),
        len(round_trip_time_mins),
        len(slow_start_thresholds),
        len(last_sends),
        len(delivery_rates),
    )
    throughputs = torch.zeros(num_states, num_steps, dtype=capacities.dtype, device=capacities.device)
    branches = torch.zeros(num_states, num_steps, dtype=torch.long, device=capacities.device)

    # Scalar buffers.
    bdp = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)
    bdp_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    data_segments = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    sent = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    round = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    cwnd = torch.tensor(0, dtype=torch.long, device=throughputs.device)
    ssh_thresh = torch.tensor(0.0, dtype=throughputs.dtype, device=throughputs.device)

    #
    sizes_bit = sizes * 8.0
    for state in range(num_states):
        #
        for step in range(num_steps):
            #
            bdp.fill_(capacities[state] * round_trip_time_mins[step] / 8.0)
            bdp_segments.fill_(torch.round(bdp * 1000.0 / 1448.0))
            data_segments.fill_(torch.round(sizes[step] * 1000.0 / 1448.0))

            if data_segments > bdp_segments:
                #
                if capacities[state] == 0.0:
                    #
                    throughputs[state, step] = 0.0
                    branches[state, step] = 0
                else:
                    #
                    throughputs[state, step] = sizes_bit[step] / (
                        sizes_bit[step] / capacities[state] + round_trip_time_mins[step]
                    )
                    branches[state, step] = 1
            else:
                #
                throughputs[state, step] = sizes_bit[step] / (round_trip_time_mins[step])
                branches[state, step] = 2

    return (throughputs, branches)

# Register customized throughput esitmation function tightly below this comment.
# Here is registration example given function `fcustom` with 3 output branches.
# ```
# register("fcustom", fcustom, 3)
# ```

register("v3", capacity_to_throughput_v3_, 4, jit=True)
register("v7", capacity_to_throughput_v7_, 3, jit=True)
register("v10", capacity_to_throughput_v10_, 4, jit=True)
register("v17", capacity_to_throughput_v17_, 4, jit=True)

def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    framework = FrameworkTransformHMMStream(disk=os.path.join("logs", "transform"), clean=False)
    framework([])


#
if __name__ == "__main__":
    #
    main()
