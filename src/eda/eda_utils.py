"""
Exploratory Data Analysis Utilities
===================================
Shared, dataset-agnostic helpers for the pipeline's per-stage EDA scripts
(eda_sft_dataset, eda_reward_dataset, eda_ppo_dataset). They standardise what
every dataset analysis reports and how it is rendered, so the three stages
produce a consistent picture: schema and field types, split sizes, a random
sample preview, sequence-length distributions, and the length-vs-cap trade-off
that fixes each stage's max_length. Everything is printed to a recording console
so a whole report can be dumped verbatim to a text file.

Inputs
------
console     : a rich Console created with record=True (see make_console).
Other helpers take a loaded datasets.Dataset, plain NumPy length arrays, the
dataset name, and small rendering parameters (percentiles, candidate caps).

Outputs
-------
None returned; each helper prints a table or panel to the console. dump() also
writes the recorded console output to a text file.

Public API
----------
make_console()                                   - recording rich Console.
dump(console, output_path)                       - write recorded output to disk.
report_schema(console, dataset_name, example)    - columns, Python types, examples.
report_splits(console, dataset_name, roles)      - per-split row counts and roles.
preview_samples(console, dataset, fields, n, char_budget, seed) - random raw examples.
length_percentiles(console, title, columns, percentiles)        - percentile table.
cap_tradeoff(console, lengths, candidates, current, semantics)  - effect of each cap.
"""

# stdlib
import os

# third-party
import numpy as np
from datasets import get_dataset_split_names, load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Percentiles reported for every length distribution.
DEFAULT_PERCENTILES = [50, 75, 90, 95, 99, 99.9, 100]


def make_console() -> Console:
    """Return a rich Console that records everything printed, for a later dump()."""
    return Console(record=True)


def dump(console: Console, output_path: str) -> None:
    """Write the recorded console output to a text file, creating parent dirs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    console.save_text(output_path)
    console.print(f"[green]EDA written to[/green] {output_path}")


def report_schema(console: Console, dataset_name: str, example: dict) -> None:
    """Print one row per column: name, Python type, and a truncated example value."""
    table = Table(title=f"Schema - {dataset_name}")
    table.add_column("column")
    table.add_column("Python type")
    table.add_column("example value (truncated)")
    for column, value in example.items():
        preview = repr(value)
        if len(preview) > 60:
            preview = preview[:57] + "..."
        table.add_row(column, type(value).__name__, preview)
    console.print(table)


def report_splits(console: Console, dataset_name: str, roles: dict[str, str]) -> None:
    """Print the row count of every split the dataset ships with, plus a role note."""
    table = Table(title="Splits and sizes")
    table.add_column("split")
    table.add_column("rows", justify="right")
    table.add_column("role")
    for split in get_dataset_split_names(dataset_name):
        n_rows = load_dataset(dataset_name, split=split).num_rows
        table.add_row(split, f"{n_rows:,}", roles.get(split, ""))
    console.print(table)


def preview_samples(
    console: Console,
    dataset,
    fields: list[str],
    n: int,
    char_budget: int,
    seed: int,
) -> None:
    """Print n random raw examples, each listed field clipped to char_budget chars."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    def clip(text: str) -> str:
        text = (text or "").strip()
        return text if len(text) <= char_budget else text[:char_budget] + " [...]"

    console.print(
        f"[bold]Random sample preview[/bold] ({len(indices)} examples, "
        f"each field clipped to {char_budget} chars)\n"
    )
    for index in indices:
        example = dataset[int(index)]
        body = "\n\n".join(f"[bold]{field}[/bold]\n{clip(str(example[field]))}" for field in fields)
        console.print(Panel(body, title=f"example #{int(index)}", border_style="dim"))
    console.print()


def length_percentiles(
    console: Console,
    title: str,
    columns: dict[str, np.ndarray],
    percentiles: list[float] = DEFAULT_PERCENTILES,
) -> None:
    """Print a percentile table with one column per named length array."""
    table = Table(title=title)
    table.add_column("percentile", justify="right")
    for name in columns:
        table.add_column(name, justify="right")
    for p in percentiles:
        label = "max" if p == 100 else f"p{p:g}"
        row = [label] + [f"{np.percentile(values, p):.0f}" for values in columns.values()]
        table.add_row(*row)
    console.print(table)


def cap_tradeoff(
    console: Console,
    lengths: np.ndarray | dict[str, np.ndarray],
    candidates: list[int],
    current: int,
    semantics: str,
) -> None:
    """Print the effect of each candidate sequence-length cap on the data.

    semantics='filter'   : a row survives only if its (binding) length <= cap,
                           matching RewardConfig and the PPO prompt filter. The
                           table reports rows kept and rows dropped.
    semantics='truncate' : every row survives but longer rows lose tokens,
                           matching SFTConfig truncation. The table reports rows
                           left intact and rows truncated.
    lengths may be a single array, or a dict of named arrays (e.g. train and
    test splits) compared side by side in one table; the per-population count
    column is then dropped for width, and the suggestion line applies to the
    first entry, by convention the population training actually filters on.
    A suggestion line names the smallest cap that still touches < 1% of rows.
    """
    if semantics not in {"filter", "truncate"}:
        raise ValueError(f"semantics must be 'filter' or 'truncate', got {semantics!r}.")
    named = lengths if isinstance(lengths, dict) else {"": lengths}
    single = len(named) == 1

    headers = {
        "filter":   ("rows kept", "% kept", "rows dropped"),
        "truncate": ("rows intact", "% intact", "rows truncated"),
    }[semantics]
    effect = {"filter": "a row is dropped", "truncate": "a row is truncated"}[semantics]
    table = Table(title=f"Effect of each cap ({semantics}: {effect} if length > cap)")
    table.add_column("cap", justify="right")
    for name in named:
        prefix = f"{name} " if name else ""
        table.add_column(f"{prefix}{headers[0]}", justify="right")
        table.add_column(f"{prefix}{headers[1]}", justify="right", style="bold")
        if single:
            table.add_column(f"{prefix}{headers[2]}", justify="right", style="red")
    for cap in candidates:
        marker = "  (current)" if cap == current else ""
        row = [f"{cap}{marker}"]
        for values in named.values():
            n = len(values)
            kept = int(np.sum(values <= cap))
            row += [f"{kept:,}", f"{100 * kept / n:.2f}%"]
            if single:
                row.append(f"{n - kept:,}")
        table.add_row(*row)
    console.print(table)

    # The suggestion is evaluated on the first population, which callers put
    # first because it is the one the training-time filter/truncation acts on.
    first_name, first = next(iter(named.items()))
    n = len(first)
    within = [int(np.sum(first <= cap)) for cap in candidates]
    scope = "rows" if single else f"{first_name} rows"
    verb = "keeps" if semantics == "filter" else "leaves intact"
    safe = next((cap for cap, kept in zip(candidates, within) if kept / n >= 0.99), None)
    if safe is not None and safe < current:
        console.print(
            f"[green]Suggestion:[/green] cap = {safe} {verb} "
            f"{100 * np.mean(first <= safe):.2f}% of {scope} while capping sequences below "
            f"the current {current} -- lower caps trade more affected rows for more "
            f"speed/memory headroom; read the table and choose.\n"
        )
    elif safe is not None:
        console.print(
            f"[yellow]Note:[/yellow] even the current cap = {current} only {verb} "
            f"{100 * np.mean(first <= current):.2f}% of {scope}; cutting it further affects "
            f"noticeably more data.\n"
        )
    else:
        console.print(
            f"[yellow]Note:[/yellow] no candidate {verb} >= 99% of {scope}; the length "
            f"distribution is long-tailed, so any cap affects a real fraction.\n"
        )


# =============================================================================
# How it works
# =============================================================================
# - Recording console: make_console returns Console(record=True) so every table
#   printed during a run is retained and dump() can replay it to a text file,
#   giving each EDA run an auditable on-disk report beside the screen output.
# - Schema/splits: report_schema reads field types straight off one example
#   (no schema is assumed), and report_splits enumerates the splits the dataset
#   actually ships with rather than hard-coding train/test.
# - Sample preview: preview_samples uses a seeded NumPy generator so the same
#   rows are shown across runs, and clips each field to a character budget so a
#   few long dialogues do not flood the screen.
# - Length reporting: length_percentiles renders any set of named length arrays
#   in one table; cap_tradeoff unifies the two ways a cap acts on data - hard
#   filtering (RM/PPO drop a row) and truncation (SFT trims tokens) - behind a
#   single semantics flag, and flags the smallest near-lossless cap.
# - Design choice: tokenisation is deliberately left to each caller, because the
#   binding length differs per stage (paired chosen/rejected, prompt+completion,
#   or prompt only); these helpers only render arrays the caller computes.
# =============================================================================
