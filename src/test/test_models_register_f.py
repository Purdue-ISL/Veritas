#
import torch
from typing import Tuple
from veritas.models.hmm.emission import register


def capacity_to_throughput_x_(
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
    Get expected throughput given capacity (no name).

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
    throughputs = torch.tile(torch.reshape(capacities, (num_states, 1)), (1, num_steps))
    branches = torch.zeros(num_states, num_steps, dtype=torch.long, device=capacities.device)
    return (throughputs, branches)


def test_register() -> None:
    R"""
    Test estimation function registration.

    Args
    ----

    Returns
    -------
    """
    #
    register("x", capacity_to_throughput_x_, 1, jit=True)


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    test_register()


#
if __name__ == "__main__":
    #
    main()
