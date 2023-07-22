#
import torch
from typing import Tuple


def stack_matrix_power_(mat: torch.Tensor, n: int) -> torch.Tensor:
    R"""
    Get stack of consecutive matrix powers starting from 0.

    Args
    ----
    - mat
        Matrix.
    - n
        Maixmum (inclusive) power.

    Returns
    -------
    - pows
        Stack of powers.
    """
    #
    buf = [torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)]
    for _ in range(1, n + 1):
        # stable stochastic matrix power.
        prod = torch.relu(torch.mm(buf[-1], mat))
        prod /= torch.sum(prod, dim=1)
        buf.append(prod)
    return torch.stack(buf, dim=2)


def baum_welch_forward_log_(
    initials_log: torch.Tensor,
    transpowers_log: torch.Tensor,
    emissions_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Baum-Welch forward pass in log version.

    Args
    ----
    - initials_log
        Initial log distribution.
    - transpowers_log
        Stack of log transition matrix powers.
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
        alphas_log[:, t] = (
            torch.logsumexp(alphas_log[:, [t - 1]] + transpowers_log[:, :, t], dim=0) + emissions_log[:, t]
        )

    #
    prob_log = torch.logsumexp(alphas_log[:, -1], dim=0)
    return (alphas_log, prob_log)


def baum_welch_backward_log_(transpowers_log: torch.Tensor, emissions_log: torch.Tensor) -> torch.Tensor:
    R"""
    Baum-Welch backward pass in log version.

    Args
    ----
    - transpowers_log
        Stack of log transition matrix powers.
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
    transitions_t_log = torch.transpose(transpowers_log, 0, 1)

    #
    betas_log[:, num_times - 1] = 0.0
    for t in range(num_times - 2, -1, -1):
        #
        betas_log[:, t] = torch.logsumexp(
            betas_log[:, [t + 1]] + emissions_log[:, [t + 1]] + transitions_t_log[:, :, t + 1],
            dim=0,
        )
    return betas_log


def baum_welch_posterior_log_(
    transpowers_log: torch.Tensor,
    emissions_log: torch.Tensor,
    alphas_log: torch.Tensor,
    betas_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R"""
    Baum-Welch posterior pass in log version.

    Args
    ----
    - transpowers_log
        Stack of log transition matrix powers.
    - gaps
        Number of hidden state transition steps between consecutive observations.
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
    transpowers_log_ = transpowers_log[:, :, 1:]
    emissions_log_ = torch.reshape(emissions_log[:, 1:], (1, num_states, num_times - 1))
    betas_log_ = torch.reshape(betas_log[:, 1:], (1, num_states, num_times - 1))
    xis_log = alphas_log_ + transpowers_log_ + emissions_log_ + betas_log_
    xis_log -= torch.logsumexp(xis_log, dim=(0, 1), keepdim=True)
    return (gammas_log, xis_log)


def sample_hidden_traces_(probs: torch.Tensor, gammas_log: torch.Tensor, xis_log: torch.Tensor) -> torch.Tensor:
    R"""
    Baum-Welch posterior pass in log version.

    Args
    ----
    - probs
        Allocated sampling probabilities.
    - gammas_log
        Posterior hidden state log distribution.
    - xis_log
        Posterior hidden state transition pair log distribution.

    Returns
    -------
    - traces
        Sampled hidden traces.
    """
    #
    (num_samples, num_steps) = probs.shape

    # Get CDF from normalized distribution.
    gammas_cumsum = torch.cumsum(torch.exp(gammas_log - torch.logsumexp(gammas_log, dim=0)), dim=0)
    xis_cumsum = torch.cumsum(torch.exp(xis_log - torch.logsumexp(xis_log, dim=0)), dim=0)

    #
    traces = torch.zeros(num_samples, num_steps, dtype=torch.long, device=probs.device)
    for i in range(num_samples):
        #
        greaters = torch.nonzero(gammas_cumsum > probs[i, num_steps - 1])
        traces[i, num_steps - 1] = torch.min(greaters)
        for t in range(num_steps - 2, -1, -1):
            #
            greaters = torch.nonzero(xis_cumsum[:, traces[i, t + 1], t] > probs[i, t])
            traces[i, t] = torch.min(greaters)
    return traces


def fill_between_traces_(
    probs: torch.Tensor,
    locs: torch.Tensor,
    steps_start: torch.Tensor,
    crit_samples: torch.Tensor,
    transpowers_log: torch.Tensor,
    emissions_log: torch.Tensor,
    full_samples: torch.Tensor,
) -> None:
    R"""
    Baum-Welch posterior pass in log version.

    Args
    ----
    - probs
        Allocated sampling probabilities.
    - locs
        Location of steps to fill samples.
    - steps_start
        Critical sample start steps.
    - crit_samples
        Critical samples.
    - transpowers_log
        Stack of log transition matrix powers.
    - emissions_log
        Log distribution of emission.
    - full_samples
        Full samples.

    Returns
    -------
    """
    #
    (num, _) = crit_samples.shape

    #
    ptr = 0
    for i in range(len(locs)):
        #
        step_lower = int(steps_start[locs[i] - 1].item())
        step_upper = int(steps_start[locs[i]].item())
        state_lower = crit_samples[:, locs[i] - 1]
        state_upper = crit_samples[:, locs[i]]
        for t in range(step_upper - 1, step_lower, -1):
            #
            for n in range(num):
                # Sample a state between from the last observation.
                pdf_log_lower = transpowers_log[state_lower[n], :, t - step_lower]
                pdf_log_upper = transpowers_log[:, state_upper[n], 1]
                pdf_log = pdf_log_lower + pdf_log_upper + emissions_log[:, t]
                cdf = torch.cumsum(torch.exp(pdf_log - torch.logsumexp(pdf_log, dim=0)), dim=0)
                state = torch.min(torch.nonzero(cdf > probs[n, ptr]))
                state_upper[n] = state

                #
                full_samples[n, t] = state
            ptr += 1


def fill_after_traces_(
    probs: torch.Tensor,
    step_total: int,
    step_last: int,
    crit_samples: torch.Tensor,
    transpowers_log: torch.Tensor,
    full_samples: torch.Tensor,
) -> None:
    R"""
    Baum-Welch posterior pass in log version.

    Args
    ----
    - probs
        Allocated sampling probabilities.
    - step_total
        Total steps filled in full samples.
    - steps_last
        Last step given by critical.
    - crit_samples
        Critical samples.
    - transpowers_log
        Stack of log transition matrix powers.
    - full_samples
        Full samples.

    Returns
    -------
    """
    #
    (num, _) = crit_samples.shape

    #
    state_lower = crit_samples[:, -1]
    for i in range(step_total - step_last - 1):
        #
        for n in range(num):
            #
            pdf_log = transpowers_log[state_lower[n], :, 0]
            cdf = torch.cumsum(torch.exp(pdf_log - torch.logsumexp(pdf_log, dim=0)), dim=0)
            state = torch.min(torch.nonzero(cdf > probs[n, i]))
            full_samples[n, step_last + i + 1] = state


#
stack_matrix_power = {False: stack_matrix_power_, True: torch.jit.script(stack_matrix_power_)}
baum_welch_forward_log = {False: baum_welch_forward_log_, True: torch.jit.script(baum_welch_forward_log_)}
baum_welch_backward_log = {False: baum_welch_backward_log_, True: torch.jit.script(baum_welch_backward_log_)}
baum_welch_posterior_log = {False: baum_welch_posterior_log_, True: torch.jit.script(baum_welch_posterior_log_)}
sample_hidden_traces = {False: sample_hidden_traces_, True: torch.jit.script(sample_hidden_traces_)}
fill_between_traces = {False: fill_between_traces_, True: torch.jit.script(fill_between_traces_)}
fill_after_traces = {False: fill_after_traces_, True: torch.jit.script(fill_after_traces_)}
