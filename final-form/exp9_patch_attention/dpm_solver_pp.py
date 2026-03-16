"""DPM-Solver++ for mr-Diff: High-Quality Sampling in ~15 Steps.

Replaces the 100-step DDPM reverse process with DPM-Solver++ (Lu et al.,
NeurIPS 2022), a high-order ODE solver for diffusion probabilistic models.

Key idea: The reverse diffusion can be written as an ODE in log-SNR space.
DPM-Solver++ exploits the semi-linear structure of this ODE, analytically
solving the linear part and using exponential integrators for the nonlinear
(model-prediction) part. This yields high-quality samples in 15-20 steps
versus DDPM's 100.

Our model uses x0-prediction (clean data prediction), which aligns directly
with DPM-Solver++'s data-prediction formulation.

Reference:
    Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models," NeurIPS 2022.
"""

import torch
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────
# Schedule Utilities
# ──────────────────────────────────────────────────────────────────────

def compute_log_snr(alpha_bars: torch.Tensor) -> torch.Tensor:
    """Compute log signal-to-noise ratio: λ(t) = log(ᾱ_t / (1 - ᾱ_t)).

    The log-SNR is the natural coordinate for DPM-Solver. It decreases
    monotonically from +inf (clean data) to -inf (pure noise), making
    it a universal parameterization across different noise schedules.
    """
    return torch.log(alpha_bars / (1.0 - alpha_bars))


def select_timesteps(num_diffusion_steps: int, num_solver_steps: int) -> list[int]:
    """Select uniformly spaced timesteps for DPM-Solver++.

    Maps M solver steps back onto the K-step diffusion schedule using
    uniform spacing in the timestep index. The solver visits these
    checkpoints, skipping intermediate steps entirely.

    Args:
        num_diffusion_steps: Total diffusion steps K (e.g., 100).
        num_solver_steps: Desired solver steps M (e.g., 15-20).

    Returns:
        List of M+1 timestep indices from K-1 down to 0.
    """
    step_size = num_diffusion_steps / num_solver_steps
    timesteps = [int(round(i * step_size)) for i in range(num_solver_steps, -1, -1)]
    timesteps = [min(t, num_diffusion_steps - 1) for t in timesteps]
    return timesteps


# ──────────────────────────────────────────────────────────────────────
# DPM-Solver++ (Second-Order, Multistep)
# ──────────────────────────────────────────────────────────────────────

class DPMSolverPP:
    """DPM-Solver++ second-order multistep sampler.

    Uses the data-prediction (x0) parameterization with a multistep
    update that caches the previous model output to achieve second-order
    accuracy without extra network evaluations per step.

    Attributes:
        schedule: Diffusion schedule with precomputed alpha_bars.
        num_solver_steps: Number of DPM-Solver++ steps (default 20).
    """

    def __init__(self, schedule, num_solver_steps: int = 20):
        self.schedule = schedule
        self.num_solver_steps = num_solver_steps

        # Precompute log-SNR for all diffusion timesteps
        self.log_snr = compute_log_snr(self.schedule.alpha_bars)

        # Select the timestep sub-sequence
        self.timesteps = select_timesteps(
            self.schedule.num_steps, self.num_solver_steps
        )

    def _sigma_and_alpha(self, k: int):
        """Extract noise level (σ) and signal level (α) at timestep k."""
        alpha_bar = self.schedule.alpha_bars[k]
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1.0 - alpha_bar)
        return alpha, sigma

    def _x0_to_noise(self, yk, x0_pred, alpha, sigma):
        """Convert x0-prediction to noise-prediction: ε = (y_k - α·x0) / σ."""
        return (yk - alpha * x0_pred) / sigma.clamp(min=1e-8)

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable,
        shape: tuple,
        device: torch.device,
        conditioning_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Run DPM-Solver++ reverse sampling.

        Algorithm outline (second-order multistep):
            1. Start from y_K ~ N(0, I)
            2. For each pair of adjacent timesteps (t_i, t_{i-1}):
               a. Predict x0 from the current noisy sample
               b. If we have a cached prediction from the prior step,
                  use second-order correction; otherwise use first-order
               c. Compute the updated sample via the exponential integrator

        Args:
            model_fn: Denoising model that predicts x0 given (y_noisy, k).
                      Signature: model_fn(y_noisy, k_tensor) -> x0_pred
            shape: Shape of the sample to generate [B, T, D].
            device: Device to run on.
            conditioning_fn: Optional function returning conditioning kwargs.

        Returns:
            Clean sample x0 of the given shape.
        """
        # Step 1: Initialize from pure noise
        yk = torch.randn(shape, device=device)

        # Cache for multistep: stores previous x0 prediction
        prev_x0 = None

        # Step 2: Iterate through selected timesteps
        for i in range(len(self.timesteps) - 1):
            k_curr = self.timesteps[i]       # Current (noisier)
            k_next = self.timesteps[i + 1]   # Next (cleaner)

            # Get schedule values
            alpha_curr, sigma_curr = self._sigma_and_alpha(k_curr)
            alpha_next, sigma_next = self._sigma_and_alpha(k_next)
            lambda_curr = self.log_snr[k_curr]
            lambda_next = self.log_snr[k_next]
            h = lambda_next - lambda_curr  # Step size in log-SNR space

            # Predict clean data x0 from current noisy sample
            batch_size = shape[0]
            k_tensor = torch.full(
                (batch_size,), k_curr, device=device, dtype=torch.long
            )
            x0_pred = model_fn(yk, k_tensor)

            if prev_x0 is None or i == 0:
                # ── First-Order Update (Euler) ──
                # x_{t-1} = (α_{t-1}/α_t) · x_t + α_{t-1}·(e^h - 1) · x0
                yk = (
                    (sigma_next / sigma_curr) * yk
                    + alpha_next * (torch.exp(-h) - 1.0) * x0_pred
                )
            else:
                # ── Second-Order Update (Multistep) ──
                # Uses cached x0 from previous step for correction.
                #
                # The second-order term adds a curvature correction:
                #   D = (x0_curr - x0_prev) / (λ_curr - λ_prev)
                #   correction = α_{t-1} · ((e^h - 1)/h - 1) · D
                lambda_prev = self.log_snr[self.timesteps[i - 1]]
                r = (lambda_curr - lambda_prev) / h

                # Second-order correction coefficient
                D = x0_pred - prev_x0
                coeff_1 = alpha_next * (torch.exp(-h) - 1.0)
                coeff_2 = alpha_next * ((torch.exp(-h) - 1.0) / h + 1.0)

                yk = (
                    (sigma_next / sigma_curr) * yk
                    + coeff_1 * x0_pred
                    + (coeff_2 / (2.0 * r)) * D
                )

            # Cache current prediction for next multistep update
            prev_x0 = x0_pred

        return yk
