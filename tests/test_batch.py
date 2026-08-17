"""Tests for the directory-level analysis in soundsanity.batch."""

import numpy as np
import pytest

import soundsanity as ss
from soundsanity.batch import ERROR_STATUS, ROOT_GROUP


@pytest.fixture
def audio_tree(tmp_path, clean_audio, write_wav):
    """
    A small directory tree mirroring the layout of audio_examples/:
    two files at the root and two labeled subfolders.
    """
    (tmp_path / "silence").mkdir()
    (tmp_path / "noisy").mkdir()

    write_wav(clean_audio, "clean.wav")
    write_wav(clean_audio, "also_clean.wav")
    write_wav(ss.make_silent(clean_audio, level_db=-90), "silence/quiet.wav")
    write_wav(ss.add_noise(clean_audio, target_snr_db=0), "noisy/hiss.wav")

    return tmp_path


class TestFindAudioFiles:
    def test_finds_files_recursively(self, audio_tree):
        found = ss.find_audio_files(audio_tree)
        assert [p.name for p in found] == [
            "also_clean.wav",
            "clean.wav",
            "hiss.wav",
            "quiet.wav",
        ]

    def test_ignores_non_audio_files(self, audio_tree):
        (audio_tree / "notes.txt").write_text("not audio")
        (audio_tree / "data.csv").write_text("a,b")
        assert all(p.suffix == ".wav" for p in ss.find_audio_files(audio_tree))

    def test_extension_match_is_case_insensitive(self, tmp_path, clean_audio, write_wav):
        write_wav(clean_audio, "UPPER.WAV")
        assert len(ss.find_audio_files(tmp_path)) == 1

    def test_extensions_can_be_narrowed(self, audio_tree):
        assert ss.find_audio_files(audio_tree, extensions={".mp3"}) == []
        assert len(ss.find_audio_files(audio_tree, extensions={".wav"})) == 4

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            ss.find_audio_files(tmp_path / "nope")

    def test_empty_directory_returns_nothing(self, tmp_path):
        assert ss.find_audio_files(tmp_path) == []


class TestAnalyzeDirectory:
    def test_returns_one_entry_per_file(self, audio_tree):
        results = ss.analyze_directory(audio_tree, jobs=1)
        assert len(results) == 4

    def test_entries_carry_path_and_group(self, audio_tree):
        results = ss.analyze_directory(audio_tree, jobs=1)
        by_path = {r["path"]: r for r in results}
        assert by_path["clean.wav"]["group"] == ROOT_GROUP
        assert by_path["silence/quiet.wav"]["group"] == "silence"
        assert by_path["noisy/hiss.wav"]["group"] == "noisy"

    def test_entries_keep_the_full_nested_report(self, audio_tree):
        result = ss.analyze_directory(audio_tree, jobs=1)[0]
        for key in ("clicks", "saturation", "silence", "noise", "issues", "duration"):
            assert key in result

    def test_statuses_match_single_file_analysis(self, audio_tree):
        results = ss.analyze_directory(audio_tree, jobs=1)
        by_path = {r["path"]: r["status"] for r in results}
        assert by_path["clean.wav"] == "CLEAN"
        assert by_path["silence/quiet.wav"] == "SILENT"
        assert by_path["noisy/hiss.wav"] == "NOISY"

    def test_results_are_ordered_by_path(self, audio_tree):
        results = ss.analyze_directory(audio_tree, jobs=1)
        assert [r["path"] for r in results] == sorted(r["path"] for r in results)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert ss.analyze_directory(tmp_path, jobs=1) == []

    def test_config_overrides_are_applied(self, audio_tree):
        results = ss.analyze_directory(audio_tree, config={"min_snr_db": 1e6}, jobs=1)
        assert all(r["status"] != "CLEAN" for r in results)

    def test_parallel_and_serial_agree(self, audio_tree):
        serial = ss.analyze_directory(audio_tree, jobs=1)
        parallel = ss.analyze_directory(audio_tree, jobs=2)
        assert [r["path"] for r in parallel] == [r["path"] for r in serial]
        assert [r["status"] for r in parallel] == [r["status"] for r in serial]

    def test_progress_callback_reports_each_file(self, audio_tree):
        seen = []
        ss.analyze_directory(audio_tree, jobs=1, progress=lambda d, t: seen.append((d, t)))
        assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            ss.analyze_directory(tmp_path / "nope")

    def test_invalid_on_error_is_rejected(self, audio_tree):
        with pytest.raises(ValueError, match="on_error"):
            ss.analyze_directory(audio_tree, on_error="explode")


class TestAnalyzeDirectoryErrorHandling:
    @pytest.fixture
    def tree_with_a_bad_file(self, audio_tree):
        (audio_tree / "broken.wav").write_bytes(b"this is not a WAV file")
        return audio_tree

    def test_collect_reports_the_failure_without_stopping(self, tree_with_a_bad_file):
        results = ss.analyze_directory(tree_with_a_bad_file, jobs=1)
        assert len(results) == 5
        broken = next(r for r in results if r["path"] == "broken.wav")
        assert broken["status"] == ERROR_STATUS
        assert broken["error"]
        # The other four files were still analyzed.
        assert sum(r["status"] != ERROR_STATUS for r in results) == 4

    def test_skip_omits_the_failure(self, tree_with_a_bad_file):
        results = ss.analyze_directory(tree_with_a_bad_file, jobs=1, on_error="skip")
        assert len(results) == 4
        assert all(r["status"] != ERROR_STATUS for r in results)

    def test_raise_propagates(self, tree_with_a_bad_file):
        with pytest.raises(RuntimeError, match="broken.wav"):
            ss.analyze_directory(tree_with_a_bad_file, jobs=1, on_error="raise")

    def test_failures_survive_the_process_boundary(self, tree_with_a_bad_file):
        results = ss.analyze_directory(tree_with_a_bad_file, jobs=2)
        broken = next(r for r in results if r["path"] == "broken.wav")
        assert broken["status"] == ERROR_STATUS
        assert broken["error"]


class TestFlattenReport:
    def test_flattens_a_successful_report(self, audio_tree):
        report = next(
            r for r in ss.analyze_directory(audio_tree, jobs=1) if r["path"] == "clean.wav"
        )
        row = ss.flatten_report(report)
        assert row["path"] == "clean.wav"
        assert row["group"] == ROOT_GROUP
        assert row["status"] == "CLEAN"
        assert row["noise_floor_db"] == report["noise"]["noise_floor_db"]
        assert row["clicks_rate"] == report["clicks"]["clicks_rate"]
        assert row["duration_s"] == report["duration"]
        assert row["error"] == ""

    def test_every_value_is_a_scalar(self, audio_tree):
        row = ss.flatten_report(ss.analyze_directory(audio_tree, jobs=1)[0])
        assert all(
            value is None or isinstance(value, (str, int, float))
            for value in row.values()
        )

    def test_issues_are_joined_into_one_field(self, audio_tree):
        report = next(
            r for r in ss.analyze_directory(audio_tree, jobs=1) if r["path"] == "noisy/hiss.wav"
        )
        assert ss.flatten_report(report)["issues"] == "; ".join(report["issues"])

    def test_error_entries_flatten_with_empty_metrics(self, audio_tree):
        (audio_tree / "broken.wav").write_bytes(b"nope")
        report = next(
            r for r in ss.analyze_directory(audio_tree, jobs=1) if r["path"] == "broken.wav"
        )
        row = ss.flatten_report(report)
        assert row["status"] == ERROR_STATUS
        assert row["noise_floor_db"] is None
        assert row["error"]

    def test_accepts_a_plain_single_file_report(self, clean_audio, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio, "solo.wav"))
        row = ss.flatten_report(report)
        assert row["path"] == "solo.wav"
        assert row["group"] == ROOT_GROUP


class TestSummarize:
    def test_counts_statuses_and_groups(self, audio_tree):
        stats = ss.summarize(ss.analyze_directory(audio_tree, jobs=1))
        assert stats["total"] == 4
        assert stats["by_status"]["CLEAN"] == 2
        assert stats["by_group"]["silence"] == {"SILENT": 1}
        assert stats["by_group"][ROOT_GROUP] == {"CLEAN": 2}

    def test_statuses_are_ordered_by_descending_count(self, audio_tree):
        stats = ss.summarize(ss.analyze_directory(audio_tree, jobs=1))
        counts = list(stats["by_status"].values())
        assert counts == sorted(counts, reverse=True)

    def test_groups_are_sorted_by_name(self, audio_tree):
        stats = ss.summarize(ss.analyze_directory(audio_tree, jobs=1))
        assert list(stats["by_group"]) == sorted(stats["by_group"])

    def test_group_counts_sum_to_the_total(self, audio_tree):
        stats = ss.summarize(ss.analyze_directory(audio_tree, jobs=1))
        total = sum(sum(counts.values()) for counts in stats["by_group"].values())
        assert total == stats["total"]

    def test_empty_input(self):
        assert ss.summarize([]) == {"total": 0, "by_status": {}, "by_group": {}}


class TestPublicApi:
    def test_batch_helpers_are_exported(self):
        for name in (
            "find_audio_files",
            "analyze_directory",
            "flatten_report",
            "summarize",
            "DEFAULT_AUDIO_EXTENSIONS",
        ):
            assert name in ss.__all__
            assert hasattr(ss, name)

    def test_library_does_not_import_the_cli(self):
        # The CLI owns argparse and all the terminal formatting; importing the
        # library must not drag it in. Checked in a clean interpreter so this
        # does not depend on what other tests have already imported.
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import soundsanity, sys; print('soundsanity.cli' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"

    def test_default_extensions_are_lowercase_with_dots(self):
        assert all(
            ext.startswith(".") and ext == ext.lower()
            for ext in ss.DEFAULT_AUDIO_EXTENSIONS
        )
