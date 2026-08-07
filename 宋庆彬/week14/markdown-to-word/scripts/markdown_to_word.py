#!/usr/bin/env python3
"""Convert a Markdown file to DOCX with Pandoc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Markdown to Word (.docx).")
    parser.add_argument("input", type=Path, help="Input .md or .markdown file")
    parser.add_argument("-o", "--output", type=Path, help="Output .docx path")
    parser.add_argument("--toc", action="store_true", help="Include a table of contents")
    parser.add_argument("--reference-doc", type=Path, help="Reference .docx for Word styles")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 2


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()

    if not source.is_file():
        return fail(f"input file not found: {source}")
    if source.suffix.lower() not in {".md", ".markdown"}:
        return fail("input must have a .md or .markdown extension")

    output = (args.output or source.with_suffix(".docx")).expanduser().resolve()
    if output.suffix.lower() != ".docx":
        return fail("output must have a .docx extension")
    if output.exists() and not args.force:
        return fail(f"output already exists (use --force to overwrite): {output}")

    reference_doc: Path | None = None
    if args.reference_doc:
        reference_doc = args.reference_doc.expanduser().resolve()
        if not reference_doc.is_file() or reference_doc.suffix.lower() != ".docx":
            return fail(f"reference document must be an existing .docx file: {reference_doc}")

    pandoc = shutil.which("pandoc")
    if not pandoc:
        return fail("pandoc is not installed or is not available on PATH")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        pandoc,
        source.name,
        "--from=markdown",
        "--to=docx",
        f"--resource-path={source.parent}",
        f"--output={output}",
    ]
    if args.toc:
        command.append("--toc")
    if reference_doc:
        command.append(f"--reference-doc={reference_doc}")

    result = subprocess.run(
        command,
        cwd=source.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Pandoc error"
        return fail(detail)
    if not output.is_file() or output.stat().st_size == 0:
        return fail(f"Pandoc reported success but no valid output was created: {output}")

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
