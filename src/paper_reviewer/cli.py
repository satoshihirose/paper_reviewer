"""CLI entry point — delegates to main.py at project root when run as a module,
or used directly as a package script entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="paper-reviewer",
        description="Automated paper quality reviewer: citation verification, consistency checks, reproducibility scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper-reviewer 2310.06825
  paper-reviewer https://arxiv.org/abs/2310.06825
  paper-reviewer paper.pdf --model qwen2.5:7b --output report.json
  paper-reviewer paper.pdf --offline
        """,
    )
    parser.add_argument(
        "input",
        help="arXiv ID (e.g. 2310.06825), arXiv URL, or local PDF path",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Ollama model name (default: qwen3:8b)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save JSON report to this file",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip external API calls (Semantic Scholar etc.)",
    )
    parser.add_argument(
        "--pdf-cache",
        metavar="DIR",
        help="Directory to cache downloaded PDFs (default: system temp)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    pdf_cache = Path(args.pdf_cache) if args.pdf_cache else None

    try:
        from paper_reviewer.pipeline import run_pipeline

        run_pipeline(
            input_str=args.input,
            model=args.model,
            output_path=output_path,
            offline=args.offline,
            pdf_cache_dir=pdf_cache,
        )
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        console.print(f"[bold red]File not found:[/bold red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
