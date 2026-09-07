"""
Field-corpus inventory: what was recorded, where, when, and with what settings.
"""

from __future__ import annotations

import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: macOS writes an AppleDouble sidecar named `._<original>` beside each file
#: when copying to a non-native filesystem, as the field drive is. They carry
#: the right extension and the right timestamped name but hold resource-fork
#: metadata, not audio. Counting them inflates the corpus by a few thousand
#: phantom "failed recordings", so they are skipped wherever files are found.
APPLEDOUBLE_PREFIX = "._"

#: Filenames AudioMoth writes: YYYYMMDD_HHMMSS.WAV, timestamped in UTC.
FILENAME_PATTERN = re.compile(r"^(\d{8})_(\d{6})$")

#: The `ICMT` comment AudioMoth embeds. The wording drifts between firmware
#: revisions ("while battery was" vs "while battery state was", an optional
#: "setting" after the gain name), so each field is matched on its own.
_COMMENT_TIME = re.compile(
    r"Recorded at (\d{2}:\d{2}:\d{2}) (\d{2}/\d{2}/\d{4}) \(UTC([+-]\d{1,2})?(?::?(\d{2}))?\)"
)
_COMMENT_DEVICE = re.compile(r"by AudioMoth ([0-9A-Fa-f]+)")
_COMMENT_GAIN = re.compile(r"at (low|low-medium|medium|medium-high|high) gain")
_COMMENT_BATTERY = re.compile(r"battery (?:state )?was ([\d.]+)V")
_COMMENT_TEMPERATURE = re.compile(r"temperature was (-?[\d.]+)C")

#: AudioMoth gain settings in increasing order, with their nominal offset in dB
#: relative to `medium`. The steps are the documented ~6 dB per setting; they
#: are a first-order correction for comparing levels across deployments, not a
#: substitute for a calibrated reference tone.
GAIN_ORDER = ("low", "low-medium", "medium", "medium-high", "high")
GAIN_OFFSET_DB = {
    "low": -12.0,
    "low-medium": -6.0,
    "medium": 0.0,
    "medium-high": 6.0,
    "high": 12.0,
}

#: Season folder names on the drive, mapped to a short label.
SEASON_PATTERN = re.compile(r"CAV[ _](\d{4})-(\d{4})")

#: Site folders are spelled differently every season ("Pta. Nebles" one year,
#: "Punta Nebles" the next). Each rule is (regex over the folder name, canonical
#: site, sensor medium). Order matters: the first match wins.
_SITE_RULES = (
    (re.compile(r"hydromoth.*tanques|hydro_tr", re.I), "Tanques Rusos", "underwater"),
    (re.compile(r"hydromoth.*ardley|hydro_ard", re.I), "Ardley", "underwater"),
    (re.compile(r"hydro", re.I), "Unknown", "underwater"),
    (re.compile(r"tanques", re.I), "Tanques Rusos", "air"),
    (re.compile(r"nebles", re.I), "Punta Nebles", "air"),
    (re.compile(r"papua|panel 1|panel1", re.I), "Colonia Papua", "air"),
    (re.compile(r"adelia|panel 2|panel2", re.I), "Colonia Adelia", "air"),
    (re.compile(r"eulogia", re.I), "Punta Eulogia", "air"),
    (re.compile(r"faro|ardley", re.I), "Ardley", "air"),
    (re.compile(r"frei", re.I), "Frei", "air"),
    (re.compile(r"drake", re.I), "Drake", "air"),
    (re.compile(r"halfthree|^htp", re.I), "Halfthree Point", "air"),
    (re.compile(r"bcaa\s*1", re.I), "BCAA1", "air"),
    (re.compile(r"bcaa\s*2", re.I), "BCAA2", "air"),
)

#: Columns `inventory` produces, in order.
INVENTORY_FIELDS = (
    "path",
    "file_name",
    "season",
    "site",
    "site_folder",
    "medium",
    "timestamp_utc",
    "duration_s",
    "sample_rate",
    "channels",
    "file_bytes",
    "gain",
    "gain_offset_db",
    "device_id",
    "battery_v",
    "internal_temp_c",
    "header_error",
)


def parse_season(name):
    """
    Map a season folder name to a short label.

    Args:
        name (str): Folder name, e.g. ``"CAV_2022-2023"`` or ``"CAV 2020-2021"``.

    Returns:
        str: Label of the form ``"2022-23"``, or the input unchanged if it does
        not look like a season folder.
    """
    match = SEASON_PATTERN.search(name)
    if not match:
        return name
    start, end = match.groups()
    return f"{start}-{end[2:]}"


def parse_site(folder_name):
    """
    Map a site folder name to a canonical site and recording medium.

    Args:
        folder_name (str): The per-site folder under a season, e.g.
            ``"Pta. Nebles"`` or ``"Hydromoth_TanquesRusos"``.

    Returns:
        tuple of (str, str): Canonical site name and medium, where medium is
        ``"air"`` or ``"underwater"``. Unrecognized folders come back as
        ``(folder_name, "air")`` so nothing is silently dropped.
    """
    for pattern, site, medium in _SITE_RULES:
        if pattern.search(folder_name):
            return site, medium
    return folder_name, "air"


def parse_timestamp(file_name):
    """
    Read the UTC timestamp AudioMoth encodes in a filename.

    Args:
        file_name (str): File name with or without extension, e.g.
            ``"20231210_195000.WAV"``.

    Returns:
        datetime | None: Timezone-aware UTC datetime, or None if the name does
        not follow the AudioMoth convention.
    """
    stem = Path(file_name).stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def parse_comment(comment):
    """
    Pull the deployment settings out of an AudioMoth ``ICMT`` comment.

    Args:
        comment (str): The raw comment text.

    Returns:
        dict: Any of ``timestamp_utc``, ``device_id``, ``gain``, ``battery_v``
        and ``internal_temp_c`` that could be read. Missing fields are absent
        rather than None, so callers can distinguish "not in this firmware's
        comment" from "recorded as empty".
    """
    out = {}

    time_match = _COMMENT_TIME.search(comment)
    if time_match:
        clock, date, offset_h, offset_m = time_match.groups()
        stamp = datetime.strptime(f"{date} {clock}", "%d/%m/%Y %H:%M:%S")
        # A comment reading "(UTC-3)" means the clock face is 3 hours behind
        # UTC, so UTC is the stamp minus the offset.
        if offset_h:
            delta = timedelta(hours=int(offset_h), minutes=int(offset_m or 0))
            stamp -= delta
        out["timestamp_utc"] = stamp.replace(tzinfo=timezone.utc)

    for key, pattern, cast in (
        ("device_id", _COMMENT_DEVICE, str),
        ("gain", _COMMENT_GAIN, str),
        ("battery_v", _COMMENT_BATTERY, float),
        ("internal_temp_c", _COMMENT_TEMPERATURE, float),
    ):
        match = pattern.search(comment)
        if match:
            out[key] = cast(match.group(1))

    return out


def read_wav_header(path, probe_bytes=4096):
    """
    Read format, duration and AudioMoth comment from a WAV file's header.

    Only the first `probe_bytes` of the file are read, so this stays cheap
    enough to run over a corpus of a hundred thousand recordings. Duration is
    derived from the `data` chunk's declared size, which means a truncated
    recording reports its intended length rather than its actual one.

    Args:
        path (str | Path): WAV file to read.
        probe_bytes (int): How much of the header to pull. The AudioMoth comment
            sits within the first few hundred bytes; the default leaves room for
            files that carry extra chunks ahead of it.

    Returns:
        dict: ``sample_rate``, ``channels``, ``bits_per_sample``, ``duration_s``
        and ``comment``, with any field left as None if the header did not
        declare it.

    Raises:
        ValueError: If the file is not RIFF/WAVE.
    """
    with open(path, "rb") as handle:
        head = handle.read(probe_bytes)

    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    info = {
        "sample_rate": None,
        "channels": None,
        "bits_per_sample": None,
        "duration_s": None,
        "comment": None,
    }
    byte_rate = None
    offset = 12

    # Walk the chunk list. `data` is the last thing we need and its payload is
    # skipped rather than read, so the loop ends as soon as it is reached.
    while offset + 8 <= len(head):
        chunk_id = head[offset : offset + 4]
        (chunk_size,) = struct.unpack("<I", head[offset + 4 : offset + 8])
        body = head[offset + 8 : offset + 8 + chunk_size]

        if chunk_id == b"fmt " and len(body) >= 16:
            _, channels, sample_rate, byte_rate, _, bits = struct.unpack(
                "<HHIIHH", body[:16]
            )
            info["channels"] = channels
            info["sample_rate"] = sample_rate
            info["bits_per_sample"] = bits
        elif chunk_id == b"LIST" and body[:4] == b"INFO":
            info["comment"] = _read_info_comment(body)
        elif chunk_id == b"data":
            if byte_rate:
                info["duration_s"] = chunk_size / byte_rate
            break

        # Chunks are word-aligned: an odd size is followed by a pad byte.
        offset += 8 + chunk_size + (chunk_size % 2)

    return info


def _read_info_comment(body):
    """Return the ``ICMT`` text from the body of a ``LIST INFO`` chunk."""
    offset = 4
    while offset + 8 <= len(body):
        sub_id = body[offset : offset + 4]
        (sub_size,) = struct.unpack("<I", body[offset + 4 : offset + 8])
        if sub_id == b"ICMT":
            raw = body[offset + 8 : offset + 8 + sub_size]
            return raw.split(b"\x00", 1)[0].decode("latin-1").strip()
        offset += 8 + sub_size + (sub_size % 2)
    return None


def describe_file(path, season_root):
    """
    Build one inventory row for a recording.

    The timestamp is taken from the filename and cross-checked against the
    header comment; they agree on every AudioMoth file seen so far, but the
    comment wins if they differ, because a file copied or renamed on land keeps
    the header the recorder wrote.

    Args:
        path (str | Path): The recording.
        season_root (str | Path): The season folder it sits under. The first
            path component below this becomes the site folder.

    Returns:
        dict: One entry per name in INVENTORY_FIELDS. Header problems land in
        ``header_error`` rather than raising, so one unreadable file does not
        abort an inventory.
    """
    path = Path(path)
    season_root = Path(season_root)
    relative = path.relative_to(season_root)
    site_folder = relative.parts[0] if len(relative.parts) > 1 else "(root)"
    site, medium = parse_site(site_folder)

    row = {field: None for field in INVENTORY_FIELDS}
    row.update(
        {
            "path": str(path),
            "file_name": path.name,
            "season": parse_season(season_root.name),
            "site": site,
            "site_folder": site_folder,
            "medium": medium,
            "timestamp_utc": parse_timestamp(path.name),
            "header_error": "",
        }
    )

    try:
        row["file_bytes"] = path.stat().st_size
        header = read_wav_header(path)
    except (OSError, ValueError) as exc:
        row["header_error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["duration_s"] = header["duration_s"]
    row["sample_rate"] = header["sample_rate"]
    row["channels"] = header["channels"]

    if header["comment"]:
        meta = parse_comment(header["comment"])
        row["device_id"] = meta.get("device_id")
        row["gain"] = meta.get("gain")
        row["battery_v"] = meta.get("battery_v")
        row["internal_temp_c"] = meta.get("internal_temp_c")
        if meta.get("timestamp_utc"):
            row["timestamp_utc"] = meta["timestamp_utc"]

    row["gain_offset_db"] = GAIN_OFFSET_DB.get(row["gain"])
    return row


def inventory(season_root, extensions=(".wav",), jobs=8, progress=None):
    """
    Build an inventory row for every recording under a season folder.

    Header reads are I/O-bound, so the work is spread over threads rather than
    processes; on an external drive the walk is dominated by seek time and more
    threads stop helping well before the CPU is busy.

    Args:
        season_root (str | Path): A season folder, e.g. ``.../CAV_2022-2023``.
        extensions (iterable of str): Extensions to accept, matched
            case-insensitively.
        jobs (int): Threads used for header reads.
        progress (callable, optional): Called as ``progress(completed, total)``.

    Returns:
        list of dict: One row per file, ordered by path.

    Raises:
        NotADirectoryError: If `season_root` is not a directory.
    """
    from concurrent.futures import ThreadPoolExecutor

    root = Path(season_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    allowed = {ext.lower() for ext in extensions}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not path.name.startswith(APPLEDOUBLE_PREFIX)
    )
    total = len(files)

    rows = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        stream = pool.map(lambda path: describe_file(path, root), files)
        for index, row in enumerate(stream, start=1):
            rows.append(row)
            if progress is not None:
                progress(index, total)

    return rows
