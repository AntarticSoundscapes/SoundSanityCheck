"""
Command-line front end for `soundsanity.batch`.

Everything here is presentation: argument parsing, table layout, colors and
exit codes. The analysis itself lives in `soundsanity.batch`.

    soundsanity-report
    soundsanity-report audio_examples/silence
    soundsanity-report --csv report.csv --jobs 4
    soundsanity-report --config thresholds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from .analysis import DEFAULT_CONFIG
from .batch import (
    ERROR_STATUS,
    FLAT_FIELDS,
    ROOT_GROUP,
    analyze_directory,
    flatten_report,
    summarize,
)

STATUS_COLORS = {
    "CLEAN": "\033[32m",
    "SILENT": "\033[90m",
    "SATURATED": "\033[31m",
    "CLICKY": "\033[33m",
    "NOISY": "\033[34m",
    ERROR_STATUS: "\033[91m",
}
RESET = "\033[0m"
BOLD = "\033[1m"

DEFAULT_DIRECTORY = Path("audio_examples")


def format_duration(seconds):
    """Render a duration in seconds as m:ss."""
    if seconds is None:
        return "-"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:d}:{secs:02d}"


def render_table(rows, colorize=False):
    """Render flat result rows as an aligned text table."""
    headers = ["FILE", "DUR", "STATUS", "FLOOR dB", "SNR dB", "SILENCE", "CLICKS/S", "SAT"]

    def cells(row):
        if row["status"] == ERROR_STATUS:
            return [row["path"], "-", ERROR_STATUS, "-", "-", "-", "-", "-"]
        return [
            row["path"],
            format_duration(row["duration_s"]),
            row["status"],
            f"{row['noise_floor_db']:.1f}",
            f"{row['snr_db']:.1f}",
            f"{row['silence_ratio']:.0%}",
            f"{row['clicks_rate']:.2f}",
            f"{row['saturation_ratio']:.2%}",
        ]

    body = [cells(row) for row in rows]
    widths = [
        max([len(headers[i])] + [len(line[i]) for line in body])
        for i in range(len(headers))
    ]

    def justify(values):
        # File paths read better left-aligned; every metric right-aligns.
        return [values[0].ljust(widths[0])] + [
            values[i].rjust(widths[i]) for i in range(1, len(values))
        ]

    header_line = "  ".join(justify(headers))
    lines = [
        f"{BOLD}{header_line}{RESET}" if colorize else header_line,
        "  ".join("-" * w for w in widths),
    ]

    for row, values in zip(rows, body):
        line = justify(values)
        if colorize and row["status"] in STATUS_COLORS:
            line[2] = f"{STATUS_COLORS[row['status']]}{line[2]}{RESET}"
        lines.append("  ".join(line))

    return "\n".join(lines)


def render_summary(results, colorize=False):
    """Render overall status counts plus a group-by-status breakdown."""
    stats = summarize(results)
    total = stats["total"]
    statuses = list(stats["by_status"])

    def heading(text):
        return f"{BOLD}{text}{RESET}" if colorize else text

    lines = ["", heading(f"Overall ({total} files)")]
    for status, count in stats["by_status"].items():
        color = STATUS_COLORS.get(status, "") if colorize else ""
        label = f"{color}{status}{RESET}" if color else status
        padding = " " * max(0, 20 - len(status))
        lines.append(f"  {label}{padding} {count:>4}  ({count / total:.0%})")

    groups = list(stats["by_group"])
    if len(groups) > 1:
        col = max(len(s) for s in statuses)
        name_width = max(len(g) for g in groups + ["GROUP"])

        header = (
            "  " + "GROUP".ljust(name_width) + "  "
            + "  ".join(s.rjust(col) for s in statuses)
        )
        lines += ["", heading("By folder"), header, "  " + "-" * (len(header) - 2)]
        for group in groups:
            counts = stats["by_group"][group]
            row = "  ".join(
                (str(counts[s]) if counts.get(s) else ".").rjust(col) for s in statuses
            )
            lines.append("  " + group.ljust(name_width) + "  " + row)

    failures = [r for r in results if r.get("status") == ERROR_STATUS]
    if failures:
        lines += ["", heading(f"Failed to analyze ({len(failures)})")]
        lines += [f"  {r['path']}: {r['error']}" for r in failures]

    return "\n".join(lines)


def write_csv(rows, path):
    """Write flat result rows to a CSV file."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FLAT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="soundsanity-report",
        description="Report the quality status of every recording in a directory.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="directory to scan recursively (default: %(default)s)",
    )
    parser.add_argument("--csv", type=Path, help="also write the results to a CSV file")
    parser.add_argument(
        "--json", type=Path, help="also write the full nested reports to a JSON file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file of threshold overrides merged into DEFAULT_CONFIG",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="worker processes (default: CPU count capped at 8; 1 runs serially)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--no-color", action="store_true", help="disable colored output")
    return parser.parse_args(argv)


def _load_config(path):
    """Load and validate a threshold-override file. Returns (config, error)."""
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read config {path}: {exc}"

    if not isinstance(config, dict):
        return None, f"config {path} must contain a JSON object"

    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        return None, f"unknown config keys: {', '.join(sorted(unknown))}"

    return config, None


def main(argv=None):
    args = parse_args(argv)
    colorize = (
        not args.no_color
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
    )

    directory = args.directory.resolve()
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    config = None
    if args.config:
        config, error = _load_config(args.config)
        if error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    # The counter rewrites a single line, so only draw it on a real terminal.
    show_progress = not args.quiet and sys.stderr.isatty()

    def progress(done, total):
        if show_progress:
            print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)

    if not args.quiet:
        print(f"Analyzing {directory} ...", file=sys.stderr)

    results = analyze_directory(
        directory, config=config, jobs=args.jobs, progress=progress
    )

    if show_progress:
        print("\r" + " " * 24 + "\r", end="", file=sys.stderr)

    if not results:
        print(f"No audio files found under {directory}", file=sys.stderr)
        return 1

    rows = [flatten_report(report) for report in results]
    # Root-level files first, then folder by folder.
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["group"] != ROOT_GROUP,
            rows[i]["group"],
            rows[i]["path"],
        ),
    )
    rows = [rows[i] for i in order]
    results = [results[i] for i in order]

    print(render_table(rows, colorize))
    print(render_summary(results, colorize))

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote {args.csv}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}", file=sys.stderr)

    return 1 if any(row["status"] == ERROR_STATUS for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
