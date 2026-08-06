"""Aggregate per-run metrics into ranked comparison tables, by pipeline stage.

Every stage writes `metrics_<label>.json` (the run summary) and `config_<label>.json`
(the resolved hyperparameters) to its results directory. This script collects
each such pair, joins them by label, and prints one table per stage with
stage-appropriate columns, sorted by the metric that matters for that stage:

- SFT  : best held-out loss (ascending).
- RM   : held-out pairwise accuracy (descending).
- PPO  : final RLHF reward (descending).
- DPO  : held-out implicit-reward accuracy (descending).

Besides printing, each rendered table is also written to its stage's results
directory as both a tab-separated file (`summary_<stage>.tsv`, for spreadsheets
and further analysis) and a Markdown table (`summary_<stage>.md`, ready to paste
into a report). Rich colour markup is stripped from the on-disk versions.

Usage:
    python -m src.analysis.aggregate_metrics                 # all stages
    python -m src.analysis.aggregate_metrics --stage rm      # one stage
    python -m src.analysis.aggregate_metrics --results-root /path/to/results
    python -m src.analysis.aggregate_metrics --no-files      # print only, do not write
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Callable

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..common.config import PROJECT_ROOT

console = Console()


# --- column helpers --------------------------------------------------------
# Each column is (header, justify, getter). The getter maps a merged run record
# to a display string; the record carries the metrics keys plus a '_config' dict.

def num(key: str, prec: int = 3) -> Callable[[dict], str]:
    def get(rec: dict) -> str:
        v = rec.get(key)
        return "-" if v is None else f"{v:.{prec}f}"
    return get


def cfg(key: str) -> Callable[[dict], str]:
    def get(rec: dict) -> str:
        v = rec.get("_config", {}).get(key)
        return "-" if v is None else str(v)
    return get


def cfg_lr(rec: dict) -> str:
    v = rec.get("_config", {}).get("learning_rate")
    return "-" if v is None else f"{v:.0e}"


def cfg_targets(rec: dict) -> str:
    v = rec.get("_config", {}).get("lora_target_modules") or rec.get("_config", {}).get("target_modules")
    return str(len(v)) if isinstance(v, list) else "-"


def accuracy_cell(rec: dict) -> str:
    v = rec.get("held_out_accuracy")
    if v is None:
        return "-"
    colour = "green" if rec.get("passed") else "red"
    return f"[{colour}]{v:.3f}[/]"


def timestamp(rec: dict) -> str:
    return (rec.get("timestamp_utc", "-") or "-")[:19].replace("T", " ")


# --- per-stage specification ----------------------------------------------

STAGE_SPECS: dict[str, dict] = {
    "sft": {
        "name": "SFT runs (ranked by best held-out loss)",
        "subdir": "sft_lora_dolly",
        "sort": ("best_eval_loss", False),  # ascending: lower loss is better
        "columns": [
            ("label", "left", lambda r: str(r.get("label", "?"))),
            ("best loss", "right", num("best_eval_loss")),
            ("perplexity", "right", num("best_eval_perplexity", 2)),
            ("final train", "right", num("final_train_loss")),
            ("r", "right", cfg("lora_r")),
            ("epochs", "right", cfg("n_epochs")),
            ("lr", "right", cfg_lr),
            ("steps", "right", lambda r: str(r.get("global_step", "-"))),
            ("timestamp", "left", timestamp),
        ],
    },
    "rm": {
        "name": "Reward-model runs (ranked by held-out pairwise accuracy)",
        "subdir": "reward_model_hh",
        "sort": ("held_out_accuracy", True),  # descending
        "columns": [
            ("label", "left", lambda r: str(r.get("label", "?"))),
            ("accuracy", "right", accuracy_cell),
            ("margin", "right", num("held_out_margin")),
            ("r", "right", cfg("lora_r")),
            ("targets", "right", cfg_targets),
            ("epochs", "right", cfg("n_epochs")),
            ("lr", "right", cfg_lr),
            ("pairs", "right", lambda r: str(r.get("n_eval_pairs", "-"))),
            ("timestamp", "left", timestamp),
        ],
    },
    "ppo": {
        "name": "PPO runs (ranked by final RLHF reward)",
        "subdir": "ppo_rlhf_loop",
        "sort": ("rlhf_reward", True),  # descending
        "columns": [
            ("label", "left", lambda r: str(r.get("label", "?"))),
            ("rlhf reward", "right", num("rlhf_reward")),
            ("rm score", "right", num("reward_model_score")),
            ("kl", "right", num("kl")),
            ("entropy", "right", num("entropy")),
            ("beta", "right", cfg("kl_coef")),
            ("episodes", "right", cfg("total_episodes")),
            ("lr", "right", cfg_lr),
            ("timestamp", "left", timestamp),
        ],
    },
    "dpo": {
        "name": "DPO runs (ranked by held-out implicit-reward accuracy)",
        "subdir": "dpo_lora_hh",
        "sort": ("gate_accuracy", True),  # descending
        "columns": [
            ("label", "left", lambda r: str(r.get("label", "?"))),
            ("accuracy", "right", num("gate_accuracy")),
            ("margin", "right", num("gate_margin")),
            ("logp chosen", "right", num("gate_logps_chosen", 1)),
            ("logp rejected", "right", num("gate_logps_rejected", 1)),
            ("beta", "right", cfg("beta")),
            ("loss", "right", cfg("loss_type")),
            ("lr", "right", cfg_lr),
            ("pairs", "right", lambda r: str(r.get("n_eval_pairs", "-"))),
            ("timestamp", "left", timestamp),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", default="all", choices=["all", *STAGE_SPECS],
        help="Pipeline stage to report (default: all).",
    )
    parser.add_argument(
        "--results-root", default=f"{PROJECT_ROOT}/results",
        help="Root directory containing each stage's results subdirectory.",
    )
    parser.add_argument(
        "--no-files", action="store_true",
        help="Only print the tables; do not write the TSV/Markdown summaries to disk.",
    )
    return parser.parse_args()


def load_runs(results_dir: str) -> list[dict]:
    """Return one merged record per run, joining metrics with its config."""
    runs = []
    for metrics_path in glob.glob(os.path.join(results_dir, "metrics_*.json")):
        with open(metrics_path) as f:
            record = json.load(f)
        label = record.get("label", os.path.basename(metrics_path)[len("metrics_"):-len(".json")])
        config_path = os.path.join(results_dir, f"config_{label}.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                record["_config"] = json.load(f)
        else:
            record["_config"] = {}
        runs.append(record)
    return runs


def render_stage(stage: str, results_root: str, write_files: bool = True) -> None:
    spec = STAGE_SPECS[stage]
    results_dir = os.path.join(results_root, spec["subdir"])
    runs = load_runs(results_dir)
    if not runs:
        console.print(f"[yellow]No runs found for '{stage}' in[/yellow] {results_dir}")
        return

    sort_key, reverse = spec["sort"]
    inf = float("-inf") if reverse else float("inf")
    runs.sort(key=lambda r: r.get(sort_key, inf) if r.get(sort_key) is not None else inf, reverse=reverse)

    headers = [header for header, _, _ in spec["columns"]]
    # Cells carry rich colour markup for the console; strip it for on-disk files.
    rows_markup = [[getter(run) for _, _, getter in spec["columns"]] for run in runs]
    rows_plain = [[Text.from_markup(cell).plain for cell in row] for row in rows_markup]

    table = Table(title=spec["name"])
    for header, justify, _ in spec["columns"]:
        table.add_column(header, justify=justify)
    for row in rows_markup:
        table.add_row(*row)
    console.print(table)

    if write_files:
        _write_summary_files(results_dir, stage, spec["name"], headers, rows_plain)


def _write_summary_files(
    results_dir: str, stage: str, title: str, headers: list[str], rows: list[list[str]]
) -> None:
    """Write the table to a TSV file (analysis) and a Markdown file (paste into a report)."""
    frame = pd.DataFrame(rows, columns=headers)

    tsv_path = os.path.join(results_dir, f"summary_{stage}.tsv")
    frame.to_csv(tsv_path, sep="\t", index=False)

    md_path = os.path.join(results_dir, f"summary_{stage}.md")
    md_lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    console.print(f"[green]Wrote[/green] {tsv_path} and {md_path}")


def main() -> None:
    args = parse_args()
    stages = list(STAGE_SPECS) if args.stage == "all" else [args.stage]
    for stage in stages:
        render_stage(stage, args.results_root, write_files=not args.no_files)


if __name__ == "__main__":
    main()
