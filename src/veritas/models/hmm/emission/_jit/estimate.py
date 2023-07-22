#
import torch
from typing import Tuple, Callable


#
capacity_to_throughput = {}
num_branches = {}


def register(
    name: str,
    func: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    num: int,
    /,
    *,
    jit: bool,
) -> None:
    R"""
    Register an estimation function.

    Args
    ----
    - name
        Registration name.
    - func
        Estimation function.
    - num
        Number of conditional return branches.
    - jit
        Register jitted function.

    Returns
    -------
    """
    #
    capacity_to_throughput[name] = {False: func}
    if jit:
        #
        capacity_to_throughput[name][True] = torch.jit.script(func)
    num_branches[name] = num
