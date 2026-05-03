"""End-to-end training of `MonotonicEtaArtifact`.

Single-stage training: sample `(w, θ_true)` from `TrainingDistribution`,
draw `D | θ_true, w`, compute the dynamic p-value across a θ-grid via
the chosen scheme's torch tilted_pvalue, evaluate the loss, backprop
through the MLP. The trained checkpoint is self-describing (records
scheme + loss + α mode + training distribution).

The loss landscape is differentiable end-to-end:
- `tilted_pvalue` is a Φ-of-affine-in-η formula (smooth).
- `θ-grid` is fixed per sample; `torch.trapezoid` is differentiable.
- The MLP output is bounded `0.01 + 0.98·sigmoid(...)` so denominators
  in `tilted_pvalue` (`denom = 1 - η(1-w)` for power_law) stay safely
  positive.

A small post-training calibration check is run on a 5×5 (θ_true, w)
grid and recorded in the checkpoint metadata as
`calibration_report`. The checkpoint is written even if calibration
fails (with a flag) — downstream regression tests can refuse to load
it; the trainer's job is to produce evidence, not gate.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..transforms import eta_inverse_torch
from .architecture import MonotonicEtaNet
from .losses import (cd_variance_loss, integrated_pvalue_loss,
                       static_width_loss)
from .pvalue_torch import get_torch_tilted_pvalue
from .sampling import TrainingDistribution, draw_data_batch, lhs_sample


_LOSS_FNS = {
    "integrated_p": integrated_pvalue_loss,
    "cd_variance": cd_variance_loss,
}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _theta_grid_for_D(
    D: torch.Tensor,
    sigma: torch.Tensor,
    n_grid: int,
    search_mult: float,
) -> torch.Tensor:
    """Build a per-sample θ-grid `D ± search_mult·σ`.

    Returns a `(B, n_grid)` tensor where each row is a linspace
    centred on its sample's `D`.
    """
    half = search_mult * sigma
    # Build a (n_grid,) base [0, 1] and broadcast.
    u = torch.linspace(0.0, 1.0, n_grid, device=D.device, dtype=D.dtype)
    lo = (D - half).unsqueeze(-1)
    hi = (D + half).unsqueeze(-1)
    return lo + u.unsqueeze(0) * (hi - lo)


def _eta_from_mlp(
    model: MonotonicEtaNet,
    abs_delta_theta: torch.Tensor,
    w: torch.Tensor,
    scheme_name: str,
) -> torch.Tensor:
    """Run the MLP on `(B, N)` `(w, |Δ_θ|)` and return η on the original
    scale via the per-scheme inverse transform."""
    from ..transforms import delta_transform_torch

    delta_prime = delta_transform_torch(abs_delta_theta)
    B, N = abs_delta_theta.shape
    w_b = w.unsqueeze(-1).expand_as(delta_prime)
    x_flat = torch.stack([w_b.reshape(-1), delta_prime.reshape(-1)], dim=-1)
    eta_prime = model(x_flat).reshape(B, N)
    return eta_inverse_torch(scheme_name, eta_prime, w_b)


def _compute_pvalue_grid(
    *,
    model: MonotonicEtaNet,
    batch: dict,
    scheme_name: str,
    statistic_name: str,
    n_grid: int,
    search_mult: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(p_theta, theta_per_sample)` for a batch.

    Returns a `(B, N)` p-value tensor and a `(B, N)` θ-grid tensor.
    """
    tilted_pvalue = get_torch_tilted_pvalue(scheme_name)

    D = batch["D"]            # (B,)
    w = batch["w"]            # (B,)
    mu0 = batch["mu0"]        # scalar
    sigma = batch["sigma"]    # scalar

    theta_per_sample = _theta_grid_for_D(D, sigma, n_grid, search_mult)
    abs_delta_theta = torch.abs(
        (1.0 - w.unsqueeze(-1)) * (mu0 - theta_per_sample) / sigma
    )
    eta = _eta_from_mlp(model, abs_delta_theta, w, scheme_name)

    p_theta = tilted_pvalue(
        theta_per_sample,
        D.unsqueeze(-1),
        w.unsqueeze(-1),
        mu0,
        sigma,
        eta,
        statistic_name,
    )

    # Numerical sanity: under torch grad, p_theta should already lie in
    # [0, 1] mathematically. Guard against float32 drift only after
    # gradient logic has captured it; we *don't* clamp under autograd
    # (clamp at the boundary kills the gradient). Instead, log a warning
    # if drift exceeds tol.
    if not torch.is_grad_enabled():
        out_of_range = (
            (p_theta < -1e-5) | (p_theta > 1 + 1e-5)
        ).any().item()
        if out_of_range:
            import warnings
            warnings.warn(
                f"p_theta drifted outside [0, 1] by > 1e-5 in {scheme_name}/"
                f"{statistic_name} forward; consider float64 or tighter "
                f"η bounds.",
                RuntimeWarning,
            )

    return p_theta, theta_per_sample


def _compute_loss(
    *,
    model: MonotonicEtaNet,
    batch: dict,
    scheme_name: str,
    statistic_name: str,
    loss_kind: str,
    alpha: Optional[float],
    n_grid: int,
    search_mult: float,
) -> torch.Tensor:
    """One forward pass returning a scalar loss for the batch."""
    p_theta, theta_per_sample = _compute_pvalue_grid(
        model=model, batch=batch,
        scheme_name=scheme_name, statistic_name=statistic_name,
        n_grid=n_grid, search_mult=search_mult,
    )

    if loss_kind == "integrated_p":
        return integrated_pvalue_loss(p_theta, theta_per_sample)
    if loss_kind == "cd_variance":
        return cd_variance_loss(p_theta, theta_per_sample)
    if loss_kind == "static_width":
        if alpha is None:
            raise ValueError("static_width loss requires alpha not None")
        return static_width_loss(p_theta, theta_per_sample, alpha=alpha)
    raise ValueError(f"Unknown loss_kind={loss_kind!r}")


@dataclass
class TrainResult:
    artifact_path: Path
    train_losses: list[float]
    val_losses: list[float]
    final_val_loss: float
    metadata: dict[str, Any]


def _compute_calibration_report(
    *,
    model: MonotonicEtaNet,
    scheme_name: str,
    statistic_name: str,
    train_dist: TrainingDistribution,
    alphas: tuple = (0.05, 0.10, 0.20),
    theta_true_grid: tuple = (-3.0, -1.0, 0.0, 1.0, 3.0),
    w_grid: tuple = (0.2, 0.35, 0.5, 0.65, 0.8),
    n_reps: int = 1000,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Empirical coverage of the trained MLP on a (θ_true, w) × α grid.

    Calibration is automatic for any η(|Δ|; w) (Phase C deriver #5),
    but this empirically verifies it on the trained model. For each
    cell, we draw `n_reps` `D ~ N(θ_true, σ²)`, compute
    `p_dyn(θ_true; D, η)` (one evaluation per replicate, at the true
    parameter), and report `P(p ≥ α)` as the coverage. Under H0:
    `p` is `U[0, 1]`, so `P(p ≥ α) = 1 - α`.

    Returns a nested dict serialisable into the checkpoint:
        {"alphas": [...], "theta_true_grid": [...], "w_grid": [...],
         "coverage": [[[float per α] per w] per θ_true]}
    """
    rng = torch.Generator(device=device).manual_seed(seed)
    sigma = train_dist.sigma
    mu0 = train_dist.mu0

    # Pre-cast scalars.
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    mu0_t = torch.tensor(mu0, dtype=torch.float32, device=device)

    coverage = [[[0.0 for _ in alphas] for _ in w_grid]
                  for _ in theta_true_grid]

    model.eval()
    with torch.no_grad():
        for i, theta_true in enumerate(theta_true_grid):
            theta_true_t = torch.tensor(
                theta_true, dtype=torch.float32, device=device
            )
            for j, w_val in enumerate(w_grid):
                w_t = torch.full((n_reps,), float(w_val),
                                  dtype=torch.float32, device=device)
                D = theta_true_t + sigma_t * torch.randn(
                    n_reps, generator=rng, device=device, dtype=torch.float32,
                )
                # Evaluate p_dyn at theta = theta_true for each D.
                # |Δ_{θ_true}| = (1 - w)|μ₀ - θ_true|/σ — same for all D.
                abs_delta = torch.abs(
                    (1.0 - w_t) * (mu0_t - theta_true_t) / sigma_t
                )
                # MLP at scalar (w, |Δ|) returns scalar η_prime.
                from ..transforms import (delta_transform_torch,
                                            eta_inverse_torch)
                delta_prime = delta_transform_torch(abs_delta)
                x = torch.stack([w_t, delta_prime], dim=-1)
                eta_prime = model(x).squeeze(-1)
                eta = eta_inverse_torch(scheme_name, eta_prime, w_t)

                tilted_pvalue = get_torch_tilted_pvalue(scheme_name)
                # Compute p at theta = theta_true for each D.
                theta_in = theta_true_t.expand(n_reps)
                p = tilted_pvalue(
                    theta_in, D, w_t, mu0_t, sigma_t, eta, statistic_name,
                )
                p = p.clamp(0.0, 1.0)
                for k, alpha in enumerate(alphas):
                    cov = float((p >= alpha).float().mean().item())
                    coverage[i][j][k] = cov

    return {
        "alphas": list(alphas),
        "theta_true_grid": list(theta_true_grid),
        "w_grid": list(w_grid),
        "coverage": coverage,
        "n_reps": n_reps,
    }


def fit_monotonic_eta_artifact(
    *,
    scheme_name: str,
    statistic_name: str = "waldo",
    loss_kind: str = "integrated_p",
    alpha: Optional[float] = None,
    train_dist: Optional[TrainingDistribution] = None,
    n_lhs: int = 10000,
    n_mc: int = 8,
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    theta_grid_n: int = 401,
    search_mult: float = 8.0,
    patience: int = 15,
    min_delta: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    out_path: Path,
    architecture_kwargs: Optional[dict] = None,
    version: str = "v0",
    verbose: bool = True,
) -> TrainResult:
    """Train a `MonotonicEtaNet` end-to-end on the dynamic-procedure loss.

    Constant-LR AdamW with **early stopping**: every epoch, validation
    loss is checked; if it hasn't improved by ≥ `min_delta` for
    `patience` consecutive epochs, training stops and the model is
    rolled back to the best (lowest val-loss) state seen. The checkpoint
    records both the configured `n_epochs` and the actual `epochs_run`.

    Writes a checkpoint at `out_path` with all metadata required by
    `MonotonicEtaArtifact.load()`. Returns a `TrainResult`.

    See `scripts/train_learned_eta.py` for the CLI entry point.
    """
    if loss_kind not in _LOSS_FNS and loss_kind != "static_width":
        raise ValueError(
            f"loss_kind must be one of {{integrated_p, cd_variance, static_width}}; "
            f"got {loss_kind!r}"
        )
    if loss_kind == "static_width":
        if alpha is None:
            raise ValueError("loss_kind=static_width requires --alpha")
        alpha_mode = f"fixed_{alpha:.6g}"
    else:
        if alpha is not None:
            raise ValueError(
                f"alpha={alpha} given but loss_kind={loss_kind!r} is "
                f"α-marginalised; alpha must be None"
            )
        alpha_mode = "marginalised"

    if train_dist is None:
        train_dist = TrainingDistribution.normal_normal_default()
    if architecture_kwargs is None:
        architecture_kwargs = {"shared_sizes": (64, 64), "mono_sizes": (64, 64)}

    device_resolved = _resolve_device(device)
    if verbose:
        print(f"[fit_monotonic_eta_artifact] device={device_resolved}")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Generate LHS samples and split train/val.
    w_all, theta_true_all = lhs_sample(train_dist, n_lhs, seed=seed)
    n_val = max(int(0.1 * n_lhs), 1)
    perm = rng.permutation(n_lhs)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    w_train, theta_train = w_all[train_idx], theta_true_all[train_idx]
    w_val, theta_val = w_all[val_idx], theta_true_all[val_idx]

    if verbose:
        print(f"[fit_monotonic_eta_artifact] n_train={len(train_idx)} "
              f"n_val={len(val_idx)} n_mc={n_mc} "
              f"early_stop=patience={patience}, min_delta={min_delta}")

    model = MonotonicEtaNet(**architecture_kwargs).to(device_resolved)
    # Constant LR: AdamW with no scheduler. The dynamic-η problem is
    # well-conditioned (smooth, bounded losses on a low-dim input) so
    # OneCycleLR's annealing wasn't buying anything; constant LR with
    # early stopping is simpler and matches user expectation.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    n_train = len(train_idx)
    steps_per_epoch = max(n_train // batch_size, 1)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_since_best = 0
    epochs_run = 0
    stopped_early = False

    for epoch in range(n_epochs):
        epochs_run = epoch + 1
        model.train()
        epoch_train = 0.0
        ep_perm = rng.permutation(n_train)
        for step in range(steps_per_epoch):
            idx = ep_perm[step * batch_size:(step + 1) * batch_size]
            if idx.size == 0:
                continue
            batch = draw_data_batch(
                train_dist, w_train[idx], theta_train[idx],
                n_mc=n_mc, rng=rng, device=device_resolved,
            )
            optimizer.zero_grad()
            loss = _compute_loss(
                model=model, batch=batch,
                scheme_name=scheme_name, statistic_name=statistic_name,
                loss_kind=loss_kind, alpha=alpha,
                n_grid=theta_grid_n, search_mult=search_mult,
            )
            loss.backward()
            optimizer.step()
            epoch_train += float(loss.item())
        train_losses.append(epoch_train / max(steps_per_epoch, 1))

        # Validation
        model.eval()
        with torch.no_grad():
            batch = draw_data_batch(
                train_dist, w_val, theta_val,
                n_mc=n_mc, rng=rng, device=device_resolved,
            )
            val_loss_t = _compute_loss(
                model=model, batch=batch,
                scheme_name=scheme_name, statistic_name=statistic_name,
                loss_kind=loss_kind, alpha=alpha,
                n_grid=theta_grid_n, search_mult=search_mult,
            )
            val_loss = float(val_loss_t.item())
            val_losses.append(val_loss)

        # Early-stopping bookkeeping: improvement = val_loss < best_val - min_delta.
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone()
                            for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if verbose and ((epoch + 1) % max(n_epochs // 10, 1) == 0 or epoch == 0):
            print(f"[epoch {epoch + 1}/{n_epochs}] "
                  f"train={train_losses[-1]:.4f} "
                  f"val={val_loss:.4f} "
                  f"best={best_val:.4f} (epoch {best_epoch + 1}, "
                  f"no-improve {epochs_since_best})")

        if epochs_since_best >= patience:
            stopped_early = True
            if verbose:
                print(f"[early stop] no val improvement for {patience} "
                      f"epochs at epoch {epoch + 1}; best={best_val:.4f} "
                      f"(epoch {best_epoch + 1}).")
            break

    # Roll back to best checkpoint.
    model.load_state_dict(best_state)

    # Post-training calibration report (Phase-C skeptic block #1).
    if verbose:
        print(f"[fit_monotonic_eta_artifact] computing calibration report...")
    calibration_report = _compute_calibration_report(
        model=model, scheme_name=scheme_name,
        statistic_name=statistic_name, train_dist=train_dist,
        device=device_resolved, seed=seed + 1,
    )
    if verbose:
        # Summarise: max |coverage - (1-α)| across the grid for each α.
        import numpy as _np
        cov = _np.array(calibration_report["coverage"])
        for k, a in enumerate(calibration_report["alphas"]):
            err = _np.abs(cov[..., k] - (1.0 - a))
            print(f"[calibration α={a}] "
                  f"mean cov={float(cov[..., k].mean()):.4f}, "
                  f"max |err|={float(err.max()):.4f}")

    # Save checkpoint
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "checkpoint_format_version": 1,
        "architecture": "MonotonicEtaNet",
        "architecture_kwargs": architecture_kwargs,
        "model_state_dict": model.state_dict(),
        "scheme": scheme_name,
        "loss": loss_kind,
        "alpha_mode": alpha_mode,
        "training_distribution": train_dist.to_dict(),
        "calibration_report": calibration_report,
        "n_lhs": n_lhs,
        "n_mc": n_mc,
        "n_epochs": n_epochs,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch + 1,
        "patience": patience,
        "min_delta": min_delta,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "theta_grid_n": theta_grid_n,
        "search_mult": search_mult,
        "seed": seed,
        "version": version,
        "training_finished_at": _dt.datetime.utcnow().isoformat(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_val_loss": best_val,
    }
    torch.save(state, str(out_path))
    if verbose:
        print(f"[fit_monotonic_eta_artifact] wrote {out_path}")

    return TrainResult(
        artifact_path=out_path,
        train_losses=train_losses,
        val_losses=val_losses,
        final_val_loss=best_val,
        metadata={k: v for k, v in state.items() if k != "model_state_dict"},
    )
