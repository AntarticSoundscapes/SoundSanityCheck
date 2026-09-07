"""Tests for the field-corpus inventory: filenames, headers and deployment metadata."""

import struct
from datetime import datetime, timezone

import numpy as np
import pytest

from soundsanity.corpus import (
    GAIN_OFFSET_DB,
    describe_file,
    inventory,
    parse_comment,
    parse_season,
    parse_site,
    parse_timestamp,
    read_wav_header,
)

COMMENT = (
    "Recorded at 07:00:00 19/12/2022 (UTC) by AudioMoth 248D9B026037E848 "
    "at medium gain while battery was 3.8V and temperature was 3.4C."
)


@pytest.fixture
def audiomoth_wav(tmp_path):
    """
    Return a helper writing a minimal AudioMoth-shaped WAV.

    Essentia's writer emits no LIST/INFO chunk, so the header is assembled by
    hand: the comment is precisely what the inventory exists to read.
    """

    def _write(name="20221219_070000.WAV", comment=COMMENT, sample_rate=48000,
               channels=1, seconds=1.0):
        samples = np.zeros(int(sample_rate * seconds), dtype="<i2")
        data = samples.tobytes()
        byte_rate = sample_rate * channels * 2

        fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, channels * 2, 16)
        chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt

        if comment is not None:
            text = comment.encode("latin-1") + b"\x00"
            if len(text) % 2:
                text += b"\x00"
            info = b"INFO" + b"ICMT" + struct.pack("<I", len(text)) + text
            chunks += b"LIST" + struct.pack("<I", len(info)) + info

        chunks += b"data" + struct.pack("<I", len(data)) + data
        body = b"WAVE" + chunks

        path = tmp_path / name
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        return path

    return _write


class TestParseSeason:
    def test_reads_underscore_and_space_forms(self):
        assert parse_season("CAV_2022-2023") == "2022-23"
        assert parse_season("CAV 2020-2021") == "2020-21"

    def test_passes_through_unrecognized_names(self):
        assert parse_season("scratch") == "scratch"


class TestParseSite:
    @pytest.mark.parametrize(
        "folder, expected",
        [
            ("Tanques rusos", "Tanques Rusos"),
            ("Tanques Rusos", "Tanques Rusos"),
            ("Pta. Nebles", "Punta Nebles"),
            ("Punta Nebles", "Punta Nebles"),
            ("panel 1 Colonia Papua", "Colonia Papua"),
            ("colonia_Papua (panel 1)", "Colonia Papua"),
            ("panel 2 Colonia Adelia", "Colonia Adelia"),
            ("colonia_Adelia (panel 2)", "Colonia Adelia"),
            ("Frei - Vieja", "Frei"),
            ("BCAA1", "BCAA1"),
        ],
    )
    def test_spelling_variants_collapse_to_one_site(self, folder, expected):
        assert parse_site(folder)[0] == expected

    def test_hydrophone_folders_are_marked_underwater(self):
        assert parse_site("Hydromoth_TanquesRusos") == ("Tanques Rusos", "underwater")
        assert parse_site("Hydromoth_Ardley") == ("Ardley", "underwater")

    def test_airborne_deployment_at_the_same_site_stays_separate(self):
        # Same place, different medium: wind reaches the two very differently.
        assert parse_site("Tanques rusos")[1] == "air"

    def test_unknown_folder_is_kept_rather_than_dropped(self):
        assert parse_site("Caleta Nueva") == ("Caleta Nueva", "air")


class TestParseTimestamp:
    def test_reads_audiomoth_filename_as_utc(self):
        stamp = parse_timestamp("20231210_195000.WAV")
        assert stamp == datetime(2023, 12, 10, 19, 50, tzinfo=timezone.utc)

    def test_accepts_a_bare_stem(self):
        assert parse_timestamp("20231210_195000") is not None

    @pytest.mark.parametrize("name", ["notes.wav", "2023121_1950.wav", "20231345_995000.wav"])
    def test_rejects_names_that_are_not_timestamps(self, name):
        assert parse_timestamp(name) is None


class TestParseComment:
    def test_reads_every_field(self):
        meta = parse_comment(COMMENT)
        assert meta["device_id"] == "248D9B026037E848"
        assert meta["gain"] == "medium"
        assert meta["battery_v"] == 3.8
        assert meta["internal_temp_c"] == 3.4
        assert meta["timestamp_utc"] == datetime(2022, 12, 19, 7, 0, tzinfo=timezone.utc)

    def test_reads_hyphenated_gain_names(self):
        assert parse_comment(COMMENT.replace("medium gain", "medium-high gain"))["gain"] == "medium-high"

    def test_tolerates_the_alternate_battery_wording(self):
        variant = COMMENT.replace("battery was", "battery state was")
        assert parse_comment(variant)["battery_v"] == 3.8

    def test_converts_a_local_clock_to_utc(self):
        # "(UTC-3)" means the clock reads three hours behind UTC.
        local = COMMENT.replace("(UTC)", "(UTC-3)")
        assert parse_comment(local)["timestamp_utc"] == datetime(
            2022, 12, 19, 10, 0, tzinfo=timezone.utc
        )

    def test_reads_a_negative_internal_temperature(self):
        assert parse_comment(COMMENT.replace("was 3.4C", "was -7.2C"))["internal_temp_c"] == -7.2

    def test_missing_fields_are_absent_not_none(self):
        assert parse_comment("Recorded by something else.") == {}


class TestReadWavHeader:
    def test_reads_format_duration_and_comment(self, audiomoth_wav):
        header = read_wav_header(audiomoth_wav(seconds=2.0))
        assert header["sample_rate"] == 48000
        assert header["channels"] == 1
        assert header["bits_per_sample"] == 16
        assert header["duration_s"] == pytest.approx(2.0)
        assert header["comment"] == COMMENT

    def test_file_without_a_comment_reads_as_none(self, audiomoth_wav):
        assert read_wav_header(audiomoth_wav(comment=None))["comment"] is None

    def test_rejects_a_non_riff_file(self, tmp_path):
        path = tmp_path / "notes.wav"
        path.write_bytes(b"this is not audio at all")
        with pytest.raises(ValueError, match="Not a RIFF/WAVE"):
            read_wav_header(path)


class TestDescribeFile:
    def test_builds_a_full_row(self, audiomoth_wav, tmp_path):
        season = tmp_path / "CAV_2022-2023"
        site = season / "Tanques rusos"
        site.mkdir(parents=True)
        source = audiomoth_wav()
        target = site / source.name
        target.write_bytes(source.read_bytes())

        row = describe_file(target, season)
        assert row["season"] == "2022-23"
        assert row["site"] == "Tanques Rusos"
        assert row["medium"] == "air"
        assert row["gain"] == "medium"
        assert row["gain_offset_db"] == GAIN_OFFSET_DB["medium"]
        assert row["device_id"] == "248D9B026037E848"
        assert row["header_error"] == ""

    def test_unreadable_file_reports_the_error_rather_than_raising(self, tmp_path):
        season = tmp_path / "CAV_2022-2023"
        site = season / "Frei"
        site.mkdir(parents=True)
        broken = site / "20221219_070000.WAV"
        broken.write_bytes(b"truncated")

        row = describe_file(broken, season)
        # The row still carries where and when, so a bad file stays traceable.
        assert row["site"] == "Frei"
        assert row["timestamp_utc"] is not None
        assert "ValueError" in row["header_error"]

    def test_comment_timestamp_wins_over_a_renamed_file(self, audiomoth_wav, tmp_path):
        season = tmp_path / "CAV_2022-2023"
        site = season / "Frei"
        site.mkdir(parents=True)
        source = audiomoth_wav(name="20990101_000000.WAV")
        target = site / source.name
        target.write_bytes(source.read_bytes())

        row = describe_file(target, season)
        assert row["timestamp_utc"].year == 2022


class TestInventory:
    def test_walks_a_season_tree(self, audiomoth_wav, tmp_path):
        season = tmp_path / "CAV_2023-2024"
        for site_name in ("Punta Nebles", "Hydromoth_Ardley"):
            (season / site_name).mkdir(parents=True)
            source = audiomoth_wav()
            (season / site_name / source.name).write_bytes(source.read_bytes())

        rows = inventory(season, jobs=2)
        assert len(rows) == 2
        assert {row["medium"] for row in rows} == {"air", "underwater"}

    def test_rejects_a_path_that_is_not_a_directory(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            inventory(tmp_path / "missing")


class TestAppleDoubleSidecars:
    """The field drive is not HFS, so macOS litters it with `._name.WAV` files."""

    def test_inventory_skips_them(self, audiomoth_wav, tmp_path):
        season = tmp_path / "CAV_2023-2024"
        site = season / "Frei"
        site.mkdir(parents=True)
        source = audiomoth_wav()
        (site / source.name).write_bytes(source.read_bytes())
        (site / f"._{source.name}").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 100)

        rows = inventory(season, jobs=2)
        assert [row["file_name"] for row in rows] == [source.name]

    def test_they_would_otherwise_look_like_failed_recordings(self, tmp_path):
        # Left in, each sidecar reports an unreadable header and inflates any
        # per-season reliability figure.
        season = tmp_path / "CAV_2023-2024"
        site = season / "Frei"
        site.mkdir(parents=True)
        sidecar = site / "._20231210_195000.WAV"
        sidecar.write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 100)

        assert describe_file(sidecar, season)["header_error"] != ""
        assert inventory(season, jobs=1) == []
