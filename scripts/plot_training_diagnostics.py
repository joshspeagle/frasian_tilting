"""Multi-panel comparison of training diagnostics across configs.

Reads N diagnostic JSON sidecars (path or autodetected from
artifacts/probe_v4_*.diagnostics.json). Produces a 4-row figure:

  Row 1: D1 metrics (eta_mean, eta_std, corr_with_argmin) over epochs.
  Row 2: D2 gradient norms (input_w, output_b, lowW vs highW).
  Row 3: D3 activation stats (penult_std_mean, n_dead_neurons).
  Row 4: D4 loss decomposition (loss_lowW vs loss_highW).

One color per config (5 configs total).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIG_NAMES = [
    "baseline", "no_boundary", "no_norm", "anti_wald_10", "stratified",
]
CONFIG_COLORS = {
    "baseline":     "tab:blue",
    "no_boundary":  "tab:orange",
    "no_norm":      "tab:green",
    "anti_wald_10": "tab:red",
    "stratified":   "tab:purple",
}


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("output/illustrations/training_diagnostics.png"))
    args = parser.parse_args()

    data: dict[str, dict] = {}
    for name in CONFIG_NAMES:
        path = Path(f"artifacts/probe_v4_{name}.diagnostics.json")
        if path.exists():
            data[name] = _load(path)
        else:
            print(f"missing: {path}")

    if not data:
        raise SystemExit("No diagnostic JSONs found.")

    # Build a 4x3 grid of panels.
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))

    # Row 1: D1 metrics
    panels_d1 = [
        ("d1_eta_mean", "trained η mean"),
        ("d1_eta_range", "trained η range (max-min)"),
        ("d1_corr_with_argmin", "corr(trained, argmin)"),
    ]
    for c, (key, label) in enumerate(panels_d1):
        ax = axes[0, c]
        for name, d in data.items():
            xs = [e["epoch"] for e in d["epochs"]]
            ys = [e[key] for e in d["epochs"]]
            ax.plot(xs, ys, label=name, color=CONFIG_COLORS[name], lw=1.5)
        ax.set_xlabel("epoch"); ax.set_ylabel(label)
        if c == 0:
            ax.legend(fontsize=8, loc="best")
        if key == "d1_corr_with_argmin":
            ax.axhline(0.0, color="gray", lw=0.5, ls="--")

    # Row 2: D2 gradient norms
    panels_d2 = [
        (["d2_grad_norm_input_w", "d2_grad_norm_output_b"],
         "grad: input_w (solid) vs output_b (dashed)"),
        (["d2_grad_norm_lowW", "d2_grad_norm_highW"],
         "grad: lowW (solid) vs highW (dashed)"),
        (["d2_grad_norm_output_w"], "grad: output layer weight"),
    ]
    for c, (keys, label) in enumerate(panels_d2):
        ax = axes[1, c]
        for name, d in data.items():
            xs = [e["epoch"] for e in d["epochs"]]
            for k_idx, k in enumerate(keys):
                ys = [e[k] for e in d["epochs"]]
                ls = "-" if k_idx == 0 else "--"
                ax.plot(xs, ys, color=CONFIG_COLORS[name], lw=1.3, ls=ls)
        ax.set_xlabel("epoch"); ax.set_ylabel(label)
        ax.set_yscale("log")

    # Row 3: D3 activation stats
    for c, (key, label, ylabel) in enumerate([
        ("d3_penult_std_mean", "penult layer mean(std)", "std"),
        ("d3_penult_std_min", "penult layer min(std)", "std"),
        ("d3_n_dead_neurons", "# dead neurons", "count"),
    ]):
        ax = axes[2, c]
        for name, d in data.items():
            xs = [e["epoch"] for e in d["epochs"]]
            ys = [e[key] for e in d["epochs"]]
            ax.plot(xs, ys, label=name, color=CONFIG_COLORS[name], lw=1.3)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10)

    # Row 4: D4 loss decomposition
    for c, (keys, label) in enumerate([
        (["d4_loss_lowW"],  "loss on lowW slices"),
        (["d4_loss_midW"],  "loss on midW slices"),
        (["d4_loss_highW"], "loss on highW slices"),
    ]):
        ax = axes[3, c]
        for name, d in data.items():
            xs = [e["epoch"] for e in d["epochs"]]
            for k in keys:
                ys = [e[k] for e in d["epochs"]]
                ax.plot(xs, ys, color=CONFIG_COLORS[name], lw=1.3)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title(label, fontsize=10)

    fig.suptitle(
        "Learned-η training diagnostics: 5 configs side-by-side\n"
        "(see docs/notes/2026-05-09-mixture-smoothness-and-learned-eta-tails.md)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}")

    # Numerical summary.
    print("\nFinal-epoch metrics:")
    print(f"{'config':<14} {'eta_mean':>10} {'corr':>10} {'penult_std':>12} "
          f"{'dead':>5} {'gN(in_w)':>10} {'gN(out_b)':>11}")
    for name, d in data.items():
        ep = d["epochs"][-1]
        print(f"{name:<14} {ep['d1_eta_mean']:+10.3f} "
              f"{ep['d1_corr_with_argmin']:+10.3f} "
              f"{ep['d3_penult_std_mean']:12.4f} "
              f"{ep['d3_n_dead_neurons']:5d} "
              f"{ep['d2_grad_norm_input_w']:10.4f} "
              f"{ep['d2_grad_norm_output_b']:11.4f}")


if __name__ == "__main__":
    main()
