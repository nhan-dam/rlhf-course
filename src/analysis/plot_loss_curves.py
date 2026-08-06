"""
Training and Evaluation Loss Curves
===================================
Plot the training and held-out evaluation loss of a Hugging Face Trainer run on
one chart, read from the run's trainer_state.json. The training loss is shown
raw and exponential-moving-average (EMA) smoothed, the evaluation loss is
overlaid with markers, the best checkpoint is starred, and dashed vertical lines
optionally mark epoch boundaries so overfitting in a later epoch is visible.

This is the helper used to produce the loss figures in the stage reports, so the
figures are regenerable rather than one-off.

Inputs
------
--state / --run-dir : a trainer_state.json path, or a checkpoints_<label>
                      directory whose latest checkpoint is used.
--out               : output PNG path.
--title             : chart title (default derived from the run label).
--smoothing         : EMA weight in [0, 1); higher is smoother (default 0.9).
--epoch-lines       : 'auto' (default, on when the run has more than one epoch),
                      'on', or 'off'.

Outputs
-------
A PNG written to --out; nothing is returned.

Public API
----------
plot_loss_curves(state_path, out_path, title, smoothing, epoch_lines) - render and save.
load_loss_history(state_path)                                         - parse trainer_state.json.
"""

# stdlib
import argparse
import glob
import json
import os
import re

# third-party
import matplotlib
matplotlib.use("Agg")  # headless: write files without a display server
import matplotlib.pyplot as plt


def main() -> None:
    args = parse_args()
    state_path = _resolve_state_path(args.state, args.run_dir)
    epoch_lines = {"auto": None, "on": True, "off": False}[args.epoch_lines]
    plot_loss_curves(state_path, args.out, args.title, args.smoothing, epoch_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Path to a trainer_state.json file.")
    source.add_argument("--run-dir", help="A checkpoints_<label> dir; its latest checkpoint is used.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Chart title (default derived from the run label).")
    parser.add_argument("--smoothing", type=float, default=0.9, help="EMA weight in [0, 1) (default 0.9).")
    parser.add_argument("--epoch-lines", choices=["auto", "on", "off"], default="auto",
                        help="Draw epoch-boundary lines (default: auto, on when >1 epoch).")
    return parser.parse_args()


def _resolve_state_path(state: str | None, run_dir: str | None) -> str:
    """Return a trainer_state.json path from either an explicit file or a run directory."""
    if state:
        return state
    direct = os.path.join(run_dir, "trainer_state.json")
    if os.path.isfile(direct):
        return direct
    checkpoints = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No trainer_state.json or checkpoint-* under {run_dir!r}.")
    latest = max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[-1]))
    return os.path.join(latest, "trainer_state.json")


def load_loss_history(state_path: str) -> dict:
    """Parse trainer_state.json into the loss series plus the best checkpoint and epoch count.

    Returns a dict with train_steps/train_loss, eval_steps/eval_loss, best_step,
    best_loss, total_epochs, and final_step. Steps and epochs come straight from
    the Trainer's log_history, so the result matches exactly what training logged.
    """
    state = json.load(open(state_path))
    log = state["log_history"]
    train = [(e["step"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
    evals = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]
    if not train or not evals:
        raise ValueError(f"{state_path} has no training and/or evaluation loss entries.")
    best_step, best_loss = min(evals, key=lambda pair: pair[1])
    return {
        "train_steps": [s for s, _ in train],
        "train_loss":  [v for _, v in train],
        "eval_steps":  [s for s, _ in evals],
        "eval_loss":   [v for _, v in evals],
        "best_step":   best_step,
        "best_loss":   best_loss,
        "total_epochs": round(state.get("epoch", 1)),
        "final_step":  train[-1][0],
    }


def _ema(values: list[float], weight: float) -> list[float]:
    """Return the bias-corrected exponential moving average, matching TensorBoard smoothing."""
    smoothed, last = [], 0.0
    for i, value in enumerate(values):
        last = weight * last + (1 - weight) * value
        smoothed.append(last / (1 - weight ** (i + 1)))
    return smoothed


def _run_label(state_path: str) -> str:
    """Best-effort run label from a path like .../checkpoints_<label>/checkpoint-N/..."""
    match = re.search(r"(?:checkpoints|adapter)_([0-9a-f]+)", state_path)
    return match.group(1) if match else os.path.basename(os.path.dirname(state_path))


def plot_loss_curves(
    state_path: str,
    out_path: str,
    title: str | None = None,
    smoothing: float = 0.9,
    epoch_lines: bool | None = None,
) -> None:
    """Render the training/evaluation loss chart for one run and save it to out_path.

    Args:
        state_path:  trainer_state.json to read.
        out_path:    output PNG path (parent directories are created).
        title:       chart title; defaults to a title naming the run label.
        smoothing:   EMA weight for the training-loss curve.
        epoch_lines: draw epoch-boundary lines; None auto-enables them for multi-epoch runs.
    """
    history = load_loss_history(state_path)
    label = _run_label(state_path)
    title = title or f"Training and evaluation loss (run {label})"
    if epoch_lines is None:
        epoch_lines = history["total_epochs"] > 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_steps"], history["train_loss"],
            color="tab:blue", alpha=0.25, lw=1.0, label="training loss (raw)")
    ax.plot(history["train_steps"], _ema(history["train_loss"], smoothing),
            color="tab:blue", lw=2.0, label=f"training loss (EMA, weight {smoothing:g})")
    ax.plot(history["eval_steps"], history["eval_loss"],
            color="tab:orange", marker="o", ms=3.5, lw=1.5, label="held-out eval loss")
    ax.scatter([history["best_step"]], [history["best_loss"]], color="tab:red", marker="*",
               s=70, zorder=5,
               label=f"best checkpoint (step {history['best_step']}, {history['best_loss']:.4f})")

    if epoch_lines and history["total_epochs"] > 1:
        for k in range(1, history["total_epochs"]):
            boundary = history["final_step"] * k / history["total_epochs"]
            ax.axvline(boundary, color="gray", ls="--", lw=1.2)
        ax.text(history["final_step"] / history["total_epochs"], ax.get_ylim()[1],
                "  epoch boundary", va="top", ha="left", color="gray", fontsize=9)

    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (best eval loss {history['best_loss']:.4f} at step {history['best_step']})")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Input: a Trainer run is fully described for plotting by its
#   trainer_state.json log_history, so the script reads losses straight from
#   there and never needs the model or the dataset.
# - Series: training loss is logged frequently and is noisy, so it is drawn raw
#   (faint) plus an EMA-smoothed line; the sparser evaluation loss is overlaid
#   with markers, and the best (lowest eval-loss) checkpoint is starred.
# - Smoothing: _ema applies bias-corrected exponential smoothing, the same
#   scheme TensorBoard uses, so the smoothed curve matches the dashboard.
# - Epoch lines: the epoch count comes from the final log entry; for multi-epoch
#   runs, dashed verticals at the boundaries make a rising eval loss after an
#   epoch (overfitting) easy to read against the still-falling training loss.
# - Reuse: plot_loss_curves and load_loss_history are importable, so other
#   tooling can reuse the parsing and rendering without the command-line layer.
# =============================================================================
