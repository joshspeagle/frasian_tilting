"""Probe autograd vs finite-difference gradients of tilted_pvalue kernels.

Diagnostic tool for the zero-gradient-through-eta investigation initiated
after the FR kernel fix (commit 83a3c0e). For each of PL / OT / MX, this
script:

1. Computes p_jax = kernel(theta, D, w, mu0, sigma, eta, "waldo").
2. Computes g_autograd = d p_jax / d eta via jax.grad.
3. Computes g_fd = (kernel(eta + h) - kernel(eta - h)) / (2h) with h=1e-4.
4. Prints |g_autograd|, |g_fd|, ratio at each tested eta.

Run as `python tools/probe_grad_pl_mx_ot.py`.
"""

from __future__ import annotations

# JAX x64 MUST be enabled before any jax.numpy import.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from frasian.learned.training.pvalue_jax import (
    power_law_tilted_pvalue_jax,
    ot_tilted_pvalue_jax,
    mixture_tilted_pvalue_jax,
)


# Canonical inputs (matching the framework's σ₀-anchored training distribution).
# Note: the original spec said theta=0.0, but with theta=mu0=0 the (mu0 - theta)
# prior-conflict term vanishes in the WALDO closed form (b=0 in all three kernels),
# making p_jax constant in eta and the gradient genuinely zero. To exercise the
# eta-dependent path we offset theta off mu0. theta=0.3 keeps us inside the
# sigma0-anchored support (sigma0 = sigma*sqrt(w/(1-w)) = 1 here) and gives a
# meaningful eta gradient.
THETA = 0.3
D = 0.5
W = 0.5
MU0 = 0.0
SIGMA = 1.0
H_FD = 1e-4


def _kernel_eta_only(kernel, eta_scalar):
    """Curry a kernel to a function of eta only, with other args fixed."""
    return kernel(
        jnp.asarray(THETA, dtype=jnp.float64),
        jnp.asarray(D, dtype=jnp.float64),
        jnp.asarray(W, dtype=jnp.float64),
        jnp.asarray(MU0, dtype=jnp.float64),
        jnp.asarray(SIGMA, dtype=jnp.float64),
        eta_scalar,
        "waldo",
    )


def probe_kernel(name: str, kernel, eta_values: list[float]) -> list[tuple]:
    """Run autograd vs finite-difference probe at each eta.

    Returns list of (eta, p_jax, g_autograd, g_fd, ratio) tuples.
    """

    def f(eta_scalar):
        return _kernel_eta_only(kernel, eta_scalar)

    grad_f = jax.grad(f)

    rows = []
    for eta_val in eta_values:
        eta_arr = jnp.asarray(eta_val, dtype=jnp.float64)

        p = float(f(eta_arr))
        g_ad = float(grad_f(eta_arr))
        p_plus = float(f(jnp.asarray(eta_val + H_FD, dtype=jnp.float64)))
        p_minus = float(f(jnp.asarray(eta_val - H_FD, dtype=jnp.float64)))
        g_fd = (p_plus - p_minus) / (2.0 * H_FD)

        if abs(g_fd) < 1e-12:
            ratio = float("nan") if abs(g_ad) < 1e-12 else float("inf")
        else:
            ratio = g_ad / g_fd

        rows.append((eta_val, p, g_ad, g_fd, ratio))
    return rows


def verdict(rows: list[tuple]) -> str:
    """Classify the probe results into CLEAN / BUGGY / SUSPICIOUS."""
    suspicious = False
    buggy = False
    for _eta, _p, g_ad, g_fd, ratio in rows:
        # If FD is essentially zero too, the kernel is locally flat — skip.
        if abs(g_fd) < 1e-10 and abs(g_ad) < 1e-10:
            continue
        # BUGGY: autograd ~ 0 while FD clearly non-zero.
        if abs(g_fd) > 1e-6 and abs(g_ad) < 1e-8:
            buggy = True
            continue
        # CLEAN: ratio close to 1.
        if abs(g_fd) > 1e-10:
            if 0.95 <= ratio <= 1.05:
                continue
            # SUSPICIOUS if discrepancy is large but not zero.
            suspicious = True
    if buggy:
        return "BUGGY"
    if suspicious:
        return "SUSPICIOUS"
    return "CLEAN"


def print_table(name: str, rows: list[tuple]) -> None:
    print(f"\n=== {name} ===")
    print(
        f"{'eta':>8}  {'p_jax':>14}  {'|g_autograd|':>14}  "
        f"{'|g_fd|':>14}  {'ratio':>14}"
    )
    for eta, p, g_ad, g_fd, ratio in rows:
        print(
            f"{eta:>8.4f}  {p:>14.6e}  {abs(g_ad):>14.6e}  "
            f"{abs(g_fd):>14.6e}  {ratio:>14.6e}"
        )
    print(f"VERDICT: {verdict(rows)}")


def main() -> None:
    global THETA
    print(f"Inputs: theta={THETA}, D={D}, w={W}, mu0={MU0}, sigma={SIGMA}, h_fd={H_FD}")
    print(f"JAX x64 enabled: {jax.config.read('jax_enable_x64')}")

    pl_etas = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5]
    ot_etas = [0.0, 0.2, 0.5, 0.8, 1.0]
    mx_etas = [0.05, 0.2, 0.5, 0.8, 0.95]

    # First pass: theta=0.3 (mild conflict, σ₀-anchored training-like).
    print_table("power_law", probe_kernel("power_law", power_law_tilted_pvalue_jax, pl_etas))
    print_table("ot", probe_kernel("ot", ot_tilted_pvalue_jax, ot_etas))
    print_table("mixture", probe_kernel("mixture", mixture_tilted_pvalue_jax, mx_etas))

    # Second pass: theta=1.0 (strong conflict, p-values not saturated near 1).
    # The theta=0.3 case showed p_jax ≈ 1.0 at eta=0.2 for OT and MX, which
    # masks the FD signal (p is clipped near the upper boundary).
    THETA = 1.0
    print(f"\n\n>>> Second pass: theta={THETA} (strong conflict, non-saturated p)")
    print_table("power_law (theta=1.0)", probe_kernel("power_law", power_law_tilted_pvalue_jax, pl_etas))
    print_table("ot (theta=1.0)", probe_kernel("ot", ot_tilted_pvalue_jax, ot_etas))
    print_table("mixture (theta=1.0)", probe_kernel("mixture", mixture_tilted_pvalue_jax, mx_etas))


if __name__ == "__main__":
    main()
