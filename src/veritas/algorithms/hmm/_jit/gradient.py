#
import torch
from typing import Tuple


def baum_welch_forward_log_(
    initials_log: torch.Tensor,
    transitions_log: torch.Tensor,
    emissions_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Baum-Welch forward pass in log version.

    Args
    ----
    - initials_log
        Initial log distribution.
    - transitions_log
        Log transition matrix.
    - emissions_log
        Emission log distribution.

    Returns
    -------
    - alphas_log
        Hidden state log distribution given forward observations.
    - prob_log
        Log probability.
    """
    #
    (num_states, num_times) = emissions_log.shape
    alphas_log = torch.zeros(num_states, num_times, dtype=emissions_log.dtype, device=emissions_log.device)

    #
    alphas_log[:, 0] = initials_log + emissions_log[:, 0]
    for t in range(1, num_times):
        #
        alphas_log[:, t] = torch.logsumexp(alphas_log[:, [t - 1]] + transitions_log, dim=0) + emissions_log[:, t]

    #
    prob_log = torch.logsumexp(alphas_log[:, -1], dim=0)
    return (alphas_log, prob_log)


def baum_welch_backward_log_(transitions_log: torch.Tensor, emissions_log: torch.Tensor) -> torch.Tensor:
    R"""
    Baum-Welch backward pass in log version.

    Args
    ----
    - transitions_log
        Log transition matrix.
    - emissions_log
        Emission log distribution.

    Returns
    -------
    - betas_log
        Hidden state log distribution given backward observations.
    """
    #
    (num_states, num_times) = emissions_log.shape
    betas_log = torch.zeros(num_states, num_times, dtype=emissions_log.dtype, device=emissions_log.device)
    transitions_t_log = transitions_log.T

    #
    betas_log[:, num_times - 1] = 0.0
    for t in range(num_times - 2, -1, -1):
        #
        betas_log[:, t] = torch.logsumexp(betas_log[:, [t + 1]] + emissions_log[:, [t + 1]] + transitions_t_log, dim=0)
    return betas_log


def baum_welch_posterior_log_(
    transitions_log: torch.Tensor,
    emissions_log: torch.Tensor,
    alphas_log: torch.Tensor,
    betas_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Baum-Welch posterior pass in log version.

    Args
    ----
    - transitions_log
        Log transition matrix.
    - emissions_log
        Emission log distribution.
    - alphas_log
        Hidden state log distribution given forward observations.
    - betas_log
        Hidden state log distribution given backward observations.

    Returns
    -------
    - gammas_log
        Hidden state log distribution.
    - xis_log
        Hidden state transition pair log distribution.
    """
    #
    (num_states, num_times) = emissions_log.shape

    #
    gammas_log = alphas_log + betas_log
    gammas_log -= torch.logsumexp(gammas_log, dim=0, keepdim=True)

    #
    alphas_log_ = torch.reshape(alphas_log[:, : num_times - 1], (num_states, 1, num_times - 1))
    transitions_log_ = torch.reshape(transitions_log, (num_states, num_states, 1))
    emissions_log_ = torch.reshape(emissions_log[:, 1:], (1, num_states, num_times - 1))
    betas_log_ = torch.reshape(betas_log[:, 1:], (1, num_states, num_times - 1))
    xis_log = alphas_log_ + transitions_log_ + emissions_log_ + betas_log_
    xis_log -= torch.logsumexp(xis_log, dim=(0, 1), keepdim=True)
    return (gammas_log, xis_log)


#
baum_welch_forward_log = {False: baum_welch_forward_log_, True: torch.jit.script(baum_welch_forward_log_)}
baum_welch_backward_log = {False: baum_welch_backward_log_, True: torch.jit.script(baum_welch_backward_log_)}
baum_welch_posterior_log = {False: baum_welch_posterior_log_, True: torch.jit.script(baum_welch_posterior_log_)}
