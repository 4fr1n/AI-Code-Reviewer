#!/usr/bin/env python3
"""
reviewer.py — Code Review Assistant CLI
----------------------------------------
Usage:
    # Review a GitHub PR
    python reviewer.py --pr https://github.com/owner/repo/pull/123

    # Review pasted code
    python reviewer.py --code path/to/file.py

    # Pipe code directly
    cat myfile.py | python reviewer.py --stdin
"""

import sys
import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from github_fetcher import fetch_pr_diff, chunk_raw_code
from reviewer_model import CodeReviewer

console = Console()

# ── Category styling ──────────────────────────────────────────────────────────

CATEGORY_STYLE = {
    "bug":         ("🐛", "bold red"),
    "security":    ("🔒", "bold magenta"),
    "performance": ("⚡", "bold yellow"),
    "style":       ("✏️ ", "bold cyan"),
    "ok":          ("✅", "bold green"),
}


# ── Output rendering ──────────────────────────────────────────────────────────

def render_header():
    console.print(Panel.fit(
        "[bold white]Code Review Assistant[/bold white]\n"
        "[dim]CodeBERT · CodeT5 · NVIDIA CUDA[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def render_file_header(filename: str, language: str, additions: int, deletions: int):
    console.rule(
        f"[bold white]{filename}[/bold white]  "
        f"[green]+{additions}[/green] [red]-{deletions}[/red]  "
        f"[dim]{language}[/dim]"
    )
    console.print()


def render_finding(result: dict, show_chunk: bool = False):
    cat   = result["category"]
    icon, style = CATEGORY_STYLE.get(cat, ("•", "white"))
    conf  = result["confidence"]
    conf_str = f"{conf*100:.0f}% confidence" if conf > 0 else "heuristic"

    # Skip clean chunks
    if cat == "ok":
        return

    console.print(
        f"  {icon}  [{style}]{cat.upper()}[/{style}]  "
        f"[dim]chunk {result['chunk_index'] + 1}  ·  {conf_str}[/dim]"
    )
    console.print(f"     [white]{result['feedback']}[/white]")

    if show_chunk:
        syntax = Syntax(
            result["chunk"], result.get("language", "text").lower(),
            theme="monokai", line_numbers=True, word_wrap=True
        )
        console.print(syntax)

    console.print()


def render_summary(all_results: list[dict], elapsed: float):
    table = Table(box=box.ROUNDED, border_style="dim", show_header=True,
                  header_style="bold white")
    table.add_column("Category",    style="bold", width=14)
    table.add_column("Count",       justify="center", width=8)
    table.add_column("Files",       width=30)

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in all_results:
        if r["category"] != "ok":
            by_cat[r["category"]].append(r["filename"])

    cat_order = ["bug", "security", "performance", "style"]
    for cat in cat_order:
        if cat in by_cat:
            icon, style = CATEGORY_STYLE[cat]
            files = ", ".join(sorted(set(by_cat[cat])))
            table.add_row(
                f"{icon} [{style}]{cat}[/{style}]",
                str(len(by_cat[cat])),
                f"[dim]{files[:28]}[/dim]",
            )

    total_issues = sum(len(v) for v in by_cat.values())
    total_chunks = len(all_results)

    console.print()
    console.rule("[bold white]Summary[/bold white]")
    console.print(table)
    console.print(
        f"\n  [dim]Reviewed [white]{total_chunks}[/white] chunks  ·  "
        f"Found [white]{total_issues}[/white] issues  ·  "
        f"Completed in [white]{elapsed:.1f}s[/white][/dim]\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI-powered code reviewer using CodeBERT + CodeT5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pr",    metavar="URL",  help="GitHub PR URL to review")
    group.add_argument("--code",  metavar="FILE", help="Path to a local code file")
    group.add_argument("--stdin", action="store_true", help="Read code from stdin")

    parser.add_argument("--show-chunks", action="store_true",
                        help="Print the code chunk alongside each finding")
    parser.add_argument("--skip-ok", action="store_true", default=True,
                        help="Hide chunks classified as OK (default: true)")

    args = parser.parse_args()

    render_header()
    start = time.time()

    # ── Load models ───────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading CodeBERT + CodeT5 …", total=None)
        reviewer = CodeReviewer()

    # ── Fetch code ────────────────────────────────────────────────────────────
    files = []

    if args.pr:
        console.print(f"[dim]Fetching PR diff:[/dim] [cyan]{args.pr}[/cyan]\n")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      console=console, transient=True) as p:
            p.add_task("Fetching from GitHub API …", total=None)
            files = fetch_pr_diff(args.pr)
        console.print(f"  [green]✓[/green] Fetched [white]{len(files)}[/white] changed file(s)\n")

    elif args.code:
        with open(args.code, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        files = chunk_raw_code(code)
        files[0]["filename"] = args.code
        console.print(f"[dim]Reviewing file:[/dim] [cyan]{args.code}[/cyan]\n")

    elif args.stdin:
        code = sys.stdin.read()
        files = chunk_raw_code(code)
        console.print("[dim]Reviewing code from stdin …[/dim]\n")

    # ── Review ────────────────────────────────────────────────────────────────
    all_results = []

    for file_info in files:
        render_file_header(
            file_info["filename"],
            file_info["language"],
            file_info["additions"],
            file_info["deletions"],
        )

        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      console=console, transient=True) as p:
            task = p.add_task(
                f"Reviewing {len(file_info['chunks'])} chunk(s) …", total=None
            )
            results = reviewer.review_file(file_info)

        for result in results:
            render_finding(result, show_chunk=args.show_chunks)

        all_results.extend(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    render_summary(all_results, elapsed=time.time() - start)


if __name__ == "__main__":
    main()
