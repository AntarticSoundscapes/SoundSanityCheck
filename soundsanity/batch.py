"""
Batch quality analysis over a directory of recordings.

`analyze_recording` handles one file; this module applies it to a whole tree of
them, in parallel, without letting a single unreadable file abort the run.
Results keep the same nested shape `analyze_recording` returns, with `path` and
`group` added so a file can be traced back to the folder it came from.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .analysis import analyze_recording
from .corpus import APPLEDOUBLE_PREFIX

#: Extensions treated as audio when scanning a directory. Matched case-insensitively.
DEFAULT_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".m4a"}
)

#: Group name used for files sitting directly in the scanned directory.
ROOT_GROUP = "(root)"

#: Status assigned to a file that could not be analyzed.
ERROR_STATUS = "ERROR"

_ON_ERROR_CHOICES = ("collect", "raise", "skip")

#: Columns produced by `flatten_report`, in order.
FLAT_FIELDS = (
    "path",
    "group",
    "file_name",
    "duration_s",
    "status",
    "noise_floor_db",
    "snr_db",
    "rms_db",
    "silence_ratio",
    "clicks_rate",
    "clicks_count",
    "saturation_ratio",
    "issues",
    "error",
)


def find_audio_files(directory, extensions=None):
    """
    Recursively collect audio files under `directory`.

    macOS AppleDouble sidecars (``._name.wav``) are skipped: they carry an audio
    extension but hold resource-fork metadata, and appear in their thousands on
    any corpus copied to a non-native filesystem.

    Args:
        directory (str | Path): Directory to scan.
        extensions (iterable of str, optional): Extensions to accept, with the
            leading dot (e.g. ``{".wav"}``). Defaults to DEFAULT_AUDIO_EXTENSIONS.

    Returns:
        list of Path: Matching files, sorted by path.

    Raises:
        NotADirectoryError: If `directory` is not a directory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    allowed = (
        DEFAULT_AUDIO_EXTENSIONS
        if extensions is None
        else {ext.lower() for ext in extensions}
    )

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not path.name.startswith(APPLEDOUBLE_PREFIX)
    )


def _group_of(relative_path):
    """Return the folder label for a path relative to the scanned root."""
    parent = str(relative_path.parent)
    return ROOT_GROUP if parent == "." else parent


def _analyze_one(task):
    """
    Analyze a single file.

    Runs in a worker process, so it must stay module-level and accept only
    picklable arguments. Exceptions are returned as error entries rather than
    raised; `analyze_directory` decides what to do with them.
    """
    path_str, root_str, config = task
    path = Path(path_str)
    relative = path.relative_to(root_str)

    try:
        report = analyze_recording(str(path), config)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
        return {
            "path": str(relative),
            "group": _group_of(relative),
            "file_name": path.name,
            "status": ERROR_STATUS,
            "error": f"{type(exc).__name__}: {exc}",
        }

    report["path"] = str(relative)
    report["group"] = _group_of(relative)
    return report


def analyze_directory(
    directory,
    config=None,
    jobs=None,
    on_error="collect",
    extensions=None,
    progress=None,
):
    """
    Run the full quality analysis on every audio file under `directory`.

    Args:
        directory (str | Path): Directory to scan recursively.
        config (dict, optional): Threshold overrides merged into DEFAULT_CONFIG.
        jobs (int, optional): Worker processes to use. ``1`` runs serially in the
            calling process, which keeps tracebacks intact when debugging.
            Defaults to the CPU count, capped at 8.
        on_error (str): What to do with a file that fails to load.
            ``"collect"`` (default) returns an entry with ``status="ERROR"`` and
            an ``error`` message, ``"skip"`` omits it, ``"raise"`` re-raises.
        extensions (iterable of str, optional): Override the audio extensions.
        progress (callable, optional): Called as ``progress(completed, total)``
            after each file. Intended for progress bars; keep it cheap.

    Returns:
        list of dict: One entry per file, ordered as `find_audio_files` returns
        them. Successful entries are `analyze_recording` reports plus ``path``
        and ``group``. Error entries carry ``path``, ``group``, ``file_name``,
        ``status`` and ``error`` only.

    Raises:
        ValueError: If `on_error` is not one of "collect", "skip", "raise".
        NotADirectoryError: If `directory` is not a directory.
        RuntimeError: If ``on_error="raise"`` and any file failed. The original
            exception cannot be re-raised faithfully across a process boundary,
            so its type and message are reported in the RuntimeError instead.
    """
    if on_error not in _ON_ERROR_CHOICES:
        raise ValueError(
            f"on_error must be one of {_ON_ERROR_CHOICES}, got {on_error!r}"
        )

    root = Path(directory).resolve()
    files = find_audio_files(root, extensions)
    if not files:
        return []

    if jobs is None:
        jobs = min(os.cpu_count() or 1, 8)

    tasks = [(str(path), str(root), config) for path in files]
    total = len(tasks)

    if jobs <= 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            results.append(_analyze_one(task))
            if progress is not None:
                progress(index, total)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            # `map` preserves input order, so results line up with `files`.
            results = []
            for index, result in enumerate(pool.map(_analyze_one, tasks), start=1):
                results.append(result)
                if progress is not None:
                    progress(index, total)

    failures = [r for r in results if r["status"] == ERROR_STATUS]
    if failures and on_error == "raise":
        raise RuntimeError(
            f"Failed to analyze {len(failures)} file(s); "
            f"first was {failures[0]['path']}: {failures[0]['error']}"
        )
    if on_error == "skip":
        return [r for r in results if r["status"] != ERROR_STATUS]

    return results


def flatten_report(report):
    """
    Collapse a nested report into a single flat row.

    Useful for CSV output or ``pandas.DataFrame(map(flatten_report, results))``.
    Error entries flatten too, with the metric columns left as None.

    Args:
        report (dict): A report from `analyze_recording` or `analyze_directory`.

    Returns:
        dict: One key per entry in FLAT_FIELDS.
    """
    clicks = report.get("clicks") or {}
    saturation = report.get("saturation") or {}
    silence = report.get("silence") or {}
    noise = report.get("noise") or {}

    return {
        "path": report.get("path", report.get("file_name", "")),
        "group": report.get("group", ROOT_GROUP),
        "file_name": report.get("file_name", ""),
        "duration_s": report.get("duration"),
        "status": report.get("status", ERROR_STATUS),
        "noise_floor_db": noise.get("noise_floor_db"),
        "snr_db": noise.get("snr_db"),
        "rms_db": noise.get("rms_db"),
        "silence_ratio": silence.get("silence_ratio"),
        "clicks_rate": clicks.get("clicks_rate"),
        "clicks_count": clicks.get("clicks_count"),
        "saturation_ratio": saturation.get("saturation_ratio"),
        "issues": "; ".join(report.get("issues", [])),
        "error": report.get("error", ""),
    }


def summarize(results):
    """
    Count statuses overall and per group.

    Args:
        results (list of dict): Output of `analyze_directory`.

    Returns:
        dict: ``{"total": int, "by_status": {status: count},
        "by_group": {group: {status: count}}}``. Counts are plain dicts, sorted
        by descending count for statuses and by name for groups.
    """
    by_status = {}
    by_group = {}

    for report in results:
        status = report.get("status", ERROR_STATUS)
        group = report.get("group", ROOT_GROUP)
        by_status[status] = by_status.get(status, 0) + 1
        by_group.setdefault(group, {})
        by_group[group][status] = by_group[group].get(status, 0) + 1

    return {
        "total": len(results),
        "by_status": dict(
            sorted(by_status.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "by_group": {group: by_group[group] for group in sorted(by_group)},
    }
