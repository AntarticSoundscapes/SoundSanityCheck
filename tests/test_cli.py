"""Tests for the command-line front end in soundsanity.cli."""

import json

import pytest

import soundsanity as ss
from soundsanity import cli


@pytest.fixture
def audio_tree(tmp_path, clean_audio, write_wav):
    """Two files at the root and one in a labeled subfolder."""
    (tmp_path / "silence").mkdir()
    write_wav(clean_audio, "clean.wav")
    write_wav(ss.add_noise(clean_audio, target_snr_db=0), "hiss.wav")
    write_wav(ss.make_silent(clean_audio, level_db=-90), "silence/quiet.wav")
    return tmp_path


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0:00"), (5, "0:05"), (65, "1:05"), (300, "5:00"), (3600, "60:00")],
    )
    def test_formats_as_minutes_and_seconds(self, seconds, expected):
        assert cli.format_duration(seconds) == expected

    def test_missing_duration_renders_as_a_dash(self):
        assert cli.format_duration(None) == "-"


class TestRenderTable:
    def test_columns_align_across_rows(self, audio_tree):
        rows = [ss.flatten_report(r) for r in ss.analyze_directory(audio_tree, jobs=1)]
        lines = cli.render_table(rows).splitlines()
        assert len({len(line) for line in lines}) == 1

    def test_every_file_gets_a_row(self, audio_tree):
        rows = [ss.flatten_report(r) for r in ss.analyze_directory(audio_tree, jobs=1)]
        lines = cli.render_table(rows).splitlines()
        # One header line, one rule, then one line per file.
        assert len(lines) == len(rows) + 2

    def test_status_is_shown_for_each_file(self, audio_tree):
        rows = [ss.flatten_report(r) for r in ss.analyze_directory(audio_tree, jobs=1)]
        table = cli.render_table(rows)
        for row in rows:
            assert row["status"] in table
            assert row["path"] in table

    def test_error_rows_render_without_metrics(self, audio_tree):
        (audio_tree / "broken.wav").write_bytes(b"nope")
        rows = [ss.flatten_report(r) for r in ss.analyze_directory(audio_tree, jobs=1)]
        table = cli.render_table(rows)
        assert "ERROR" in table
        # A failed file must not crash the formatter on its missing numbers.
        assert "broken.wav" in table

    def test_color_is_off_by_default_and_opt_in(self, audio_tree):
        rows = [ss.flatten_report(r) for r in ss.analyze_directory(audio_tree, jobs=1)]
        assert "\033[" not in cli.render_table(rows)
        assert "\033[" in cli.render_table(rows, colorize=True)


class TestRenderSummary:
    def test_reports_the_total_and_each_status(self, audio_tree):
        results = ss.analyze_directory(audio_tree, jobs=1)
        summary = cli.render_summary(results)
        assert "Overall (3 files)" in summary
        for status in {r["status"] for r in results}:
            assert status in summary

    def test_folder_breakdown_appears_when_there_are_subfolders(self, audio_tree):
        summary = cli.render_summary(ss.analyze_directory(audio_tree, jobs=1))
        assert "By folder" in summary
        assert "silence" in summary

    def test_folder_breakdown_is_omitted_for_a_flat_directory(
        self, tmp_path, clean_audio, write_wav
    ):
        write_wav(clean_audio, "only.wav")
        summary = cli.render_summary(ss.analyze_directory(tmp_path, jobs=1))
        assert "By folder" not in summary

    def test_failures_are_listed(self, audio_tree):
        (audio_tree / "broken.wav").write_bytes(b"nope")
        summary = cli.render_summary(ss.analyze_directory(audio_tree, jobs=1))
        assert "Failed to analyze (1)" in summary
        assert "broken.wav" in summary


class TestMain:
    def test_prints_a_table_and_exits_zero(self, audio_tree, capsys):
        exit_code = cli.main([str(audio_tree), "--no-color", "-q"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "clean.wav" in out
        assert "Overall (3 files)" in out

    def test_missing_directory_exits_two(self, tmp_path, capsys):
        assert cli.main([str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_directory_without_audio_exits_one(self, tmp_path, capsys):
        (tmp_path / "notes.txt").write_text("nothing here")
        assert cli.main([str(tmp_path), "-q"]) == 1
        assert "No audio files found" in capsys.readouterr().err

    def test_a_failed_file_exits_one(self, audio_tree, capsys):
        (audio_tree / "broken.wav").write_bytes(b"nope")
        assert cli.main([str(audio_tree), "--no-color", "-q"]) == 1

    def test_csv_output_has_a_row_per_file(self, audio_tree, tmp_path, capsys):
        destination = tmp_path / "out" / "report.csv"
        destination.parent.mkdir()
        cli.main([str(audio_tree), "--no-color", "-q", "--csv", str(destination)])

        import csv

        with destination.open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert set(rows[0]) == set(cli.FLAT_FIELDS)

    def test_json_output_keeps_the_nested_reports(self, audio_tree, tmp_path):
        destination = tmp_path / "report.json"
        cli.main([str(audio_tree), "--no-color", "-q", "--json", str(destination)])

        payload = json.loads(destination.read_text())
        assert len(payload) == 3
        assert "noise" in payload[0]
        assert "noise_floor_db" in payload[0]["noise"]

    def test_config_file_changes_the_verdicts(self, audio_tree, tmp_path, capsys):
        config = tmp_path / "thresholds.json"
        config.write_text(json.dumps({"min_snr_db": 1e6}))
        cli.main([str(audio_tree), "--no-color", "-q", "--config", str(config)])
        assert "CLEAN" not in capsys.readouterr().out

    def test_unknown_config_keys_are_rejected(self, audio_tree, tmp_path, capsys):
        config = tmp_path / "bad.json"
        config.write_text(json.dumps({"not_a_real_threshold": 1}))
        assert cli.main([str(audio_tree), "--config", str(config)]) == 2
        assert "unknown config keys" in capsys.readouterr().err

    def test_malformed_config_is_rejected(self, audio_tree, tmp_path, capsys):
        config = tmp_path / "bad.json"
        config.write_text("{not json")
        assert cli.main([str(audio_tree), "--config", str(config)]) == 2
        assert "could not read config" in capsys.readouterr().err

    def test_non_object_config_is_rejected(self, audio_tree, tmp_path, capsys):
        config = tmp_path / "list.json"
        config.write_text("[1, 2, 3]")
        assert cli.main([str(audio_tree), "--config", str(config)]) == 2
        assert "must contain a JSON object" in capsys.readouterr().err

    def test_root_files_are_listed_before_subfolders(self, audio_tree, capsys):
        cli.main([str(audio_tree), "--no-color", "-q"])
        out = capsys.readouterr().out
        assert out.index("clean.wav") < out.index("silence/quiet.wav")
