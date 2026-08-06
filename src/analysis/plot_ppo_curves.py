"""
PPO Training-Trajectory Curves
==============================
Plot the monitored objectives of a TRL PPO run on update-step axes, read from
the run's trainer_state.json log_history. Every scalar shown is exactly what the
trainer logged to TensorBoard, so the figures are regenerable rather than
one-off. Noisy per-update series are drawn raw (faint) with an exponential
-moving-average (EMA) overlay, matching the SFT/RM loss figures produced by
plot_loss_curves.py.

Three figures are produced, one per signal tier of the PPO report's Section 5:

- primary   : objective/scores and objective/rlhf_reward over objective/kl,
              stacked and sharing the x-axis.
- stability  : policy/entropy_avg, loss/value_avg, policy/clipfrac_avg, and
              policy/approxkl_avg as a 2x2 panel.
- context   : val/num_eos_tokens on its own axis.

Inputs
------
--state / --run-dir : a trainer_state.json path, or a checkpoints_<label>
                      directory whose latest checkpoint is used.
--out-dir           : directory for the PNGs (default reports/assets/images).
--figure            : which figure(s) to render (default all).
--smoothing         : EMA weight in [0, 1); higher is smoother (default 0.9).

Outputs
-------
One PNG per selected figure, named ppo_<figure>_<label>.png; nothing is returned.

Public API
----------
plot_ppo_primary(history, out_path, label, smoothing)    - reward and KL.
plot_ppo_stability(history, out_path, label, smoothing)  - entropy and trust-region health.
plot_ppo_context(history, out_path, label, smoothing)    - EOS termination.
load_ppo_history(state_path)                             - parse trainer_state.json.
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

# The scalars this module reads from the PPO trainer's log_history.
SERIES_KEYS = (
    "objective/scores", "objective/rlhf_reward", "objective/non_score_reward",
    "objective/kl", "policy/entropy_avg",
    "loss/value_avg", "policy/clipfrac_avg", "policy/approxkl_avg", "val/num_eos_tokens",
)


def main() -> None:
    args = parse_args()
    state_path = _resolve_state_path(args.state, args.run_dir)
    history = load_ppo_history(state_path)
    label = _run_label(state_path)
    os.makedirs(args.out_dir, exist_ok=True)

    figures = {
        "primary": plot_ppo_primary,
        "stability": plot_ppo_stability,
        "context": plot_ppo_context,
    }
    selected = figures if args.figure == "all" else {args.figure: figures[args.figure]}
    for name, fn in selected.items():
        out_path = os.path.join(args.out_dir, f"ppo_{name}_{label}.png")
        fn(history, out_path, label, args.smoothing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Path to a trainer_state.json file.")
    source.add_argument("--run-dir", help="A checkpoints_<label> dir; its latest checkpoint is used.")
    parser.add_argument("--out-dir", default="reports/assets/images", help="Output directory (default reports/assets/images).")
    parser.add_argument("--figure", choices=["all", "primary", "stability", "context"],
                        default="all", help="Which figure to render (default all).")
    parser.add_argument("--smoothing", type=float, default=0.9, help="EMA weight in [0, 1) (default 0.9).")
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


def load_ppo_history(state_path: str) -> dict:
    """Parse trainer_state.json into the PPO scalar series keyed by update step.

    Returns a dict with a 'steps' list and one list per key in SERIES_KEYS, taken
    straight from the Trainer's log_history so the result matches what training
    logged. Entries missing a key are skipped for that key only.
    """
    log = json.load(open(state_path))["log_history"]
    steps = [e["step"] for e in log if "step" in e]
    history = {"steps": steps}
    for key in SERIES_KEYS:
        history[key] = [e.get(key) for e in log if "step" in e]
    if not steps:
        raise ValueError(f"{state_path} has no stepped log_history entries.")
    return history


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


def _raw_and_ema(ax, steps, values, colour, label, smoothing):
    """Draw a faint raw series with a bold EMA overlay, the house convention."""
    ax.plot(steps, values, color=colour, alpha=0.22, lw=1.0)
    ax.plot(steps, _ema(values, smoothing), color=colour, lw=2.0,
            label=f"{label} (EMA {smoothing:g})")


def plot_ppo_primary(history: dict, out_path: str, label: str, smoothing: float = 0.9) -> None:
    """Primary tier: reward (raw score and net RLHF reward) over KL divergence.

    The gap between the two reward series is exactly beta times the KL plotted
    below, so it is not shaded: it would restate the lower panel rescaled.
    """
    steps = history["steps"]
    kl = history["objective/kl"]
    kl_max_i = max(range(len(kl)), key=lambda i: kl[i])

    fig, (ax_r, ax_kl) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    _raw_and_ema(ax_r, steps, history["objective/scores"], "tab:blue", "raw RM score", smoothing)
    _raw_and_ema(ax_r, steps, history["objective/rlhf_reward"], "tab:green", "net RLHF reward", smoothing)
    ax_r.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax_r.set_ylabel("reward")
    ax_r.set_title(f"Primary signals (run {label})")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(frameon=False, fontsize=9, loc="lower right")

    _raw_and_ema(ax_kl, steps, kl, "tab:purple", "KL to SFT reference", smoothing)
    ax_kl.scatter([steps[kl_max_i]], [kl[kl_max_i]], color="tab:red", marker="v", s=45, zorder=5,
                  label=f"maximum {kl[kl_max_i]:.2f} (update {steps[kl_max_i]})")
    ax_kl.set_ylabel("objective/kl (nats)")
    ax_kl.set_xlabel("update step")
    ax_kl.grid(True, alpha=0.3)
    ax_kl.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, out_path)


def plot_ppo_stability(history: dict, out_path: str, label: str, smoothing: float = 0.9) -> None:
    """Stability tier: entropy, critic value loss, policy clip fraction, and approximate KL.

    val/ratio is deliberately not a panel. It is a mean of per-microbatch mean
    ratios, so signed deviations cancel and it sits at 1.000 throughout, whereas
    policy/approxkl_avg squares the same log-ratio and does carry the movement.
    """
    steps = history["steps"]
    entropy = history["policy/entropy_avg"]
    ent_ema = _ema(entropy, smoothing)
    ent_min_i = min(range(len(ent_ema)), key=lambda i: ent_ema[i])

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    (ax_ent, ax_vl), (ax_cf, ax_kl) = axes

    _raw_and_ema(ax_ent, steps, entropy, "tab:orange", "per-token entropy", smoothing)
    ax_ent.scatter([steps[ent_min_i]], [ent_ema[ent_min_i]], color="tab:red", marker="^", s=45, zorder=5,
                   label=f"smoothed min {ent_ema[ent_min_i]:.3f} (update {steps[ent_min_i]})")
    ax_ent.set_ylim(bottom=0.0)
    ax_ent.set_ylabel("policy/entropy_avg (nats per token)")
    ax_ent.set_title("Per-token policy entropy")
    ax_ent.legend(frameon=False, fontsize=8, loc="lower right")

    _raw_and_ema(ax_vl, steps, history["loss/value_avg"], "tab:blue", "value loss", smoothing)
    ax_vl.set_ylabel("loss/value_avg")
    ax_vl.set_title("Critic value loss")

    _raw_and_ema(ax_cf, steps, history["policy/clipfrac_avg"], "tab:green", "clip fraction", smoothing)
    ax_cf.set_ylabel("policy/clipfrac_avg")
    ax_cf.set_xlabel("update step")
    ax_cf.set_title("Policy clip fraction")

    _raw_and_ema(ax_kl, steps, history["policy/approxkl_avg"], "tab:purple", "approximate KL", smoothing)
    ax_kl.set_ylabel("policy/approxkl_avg")
    ax_kl.set_xlabel("update step")
    ax_kl.set_title("Approximate KL to the rollout policy")

    for ax in (ax_ent, ax_vl, ax_cf, ax_kl):
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Stability signals (run {label})", fontsize=12)
    _save(fig, out_path)


def plot_ppo_context(history: dict, out_path: str, label: str, smoothing: float = 0.9) -> None:
    """Context tier: EOS tokens per batch, showing the missing-EOS penalty taking effect."""
    steps = history["steps"]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    _raw_and_ema(ax, steps, history["val/num_eos_tokens"], "tab:orange", "EOS tokens per batch", smoothing)
    ax.set_xlabel("update step")
    ax.set_ylabel("val/num_eos_tokens")
    ax.set_title(f"Context signal: EOS terminations (run {label})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, out_path)


def _save(fig, out_path: str) -> None:
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Input: a TRL PPO run is fully described for plotting by its
#   trainer_state.json log_history, so the script reads the scalars straight from
#   there and never needs the models or the dataset. The same log backs the
#   run's TensorBoard event file.
# - Series: per-update PPO scalars are noisy, so each is drawn raw (faint) plus an
#   EMA overlay, including policy/approxkl_avg, whose raw series is noisy enough
#   that the opening decline is only legible once smoothed.
# - Smoothing: _ema applies bias-corrected exponential smoothing, the same scheme
#   TensorBoard and plot_loss_curves.py use, so the curves match the dashboard.
# - Entropy choice: the stability panel plots policy/entropy_avg, the closed-form
#   per-token entropy of the policy's output distribution, not objective/entropy.
#   The latter is a summed sampled-token surprisal whose post-EOS positions TRL
#   fills with a sentinel log-probability of 1.0, so it subtracts a nat per padded
#   slot and tracks response length rather than entropy, going negative once
#   padding dominates. See the PPO report, Section 5.1.
# - Reuse: the three plot_* functions and load_ppo_history are importable, so
#   other tooling can reuse the parsing and rendering without the CLI layer.
# =============================================================================
