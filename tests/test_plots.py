"""Smoke tests for the diagnostic plots in soundsanity.plots.

These assert that each figure is built with the expected structure and overlays
rather than comparing rendered pixels, which would be brittle across matplotlib
versions. The Agg backend is selected in conftest so nothing opens a window.
"""

import numpy as np
from matplotlib.figure import Figure

import soundsanity as ss


class TestPlotWaveformWithIssues:
    def test_returns_a_figure_and_axis(self, clean_audio, sample_rate):
        fig, ax = ss.plot_waveform_with_issues(clean_audio, sample_rate)
        assert isinstance(fig, Figure)
        assert ax in fig.axes

    def test_title_and_axis_labels_are_set(self, clean_audio, sample_rate):
        _, ax = ss.plot_waveform_with_issues(clean_audio, sample_rate, title="My Recording")
        assert ax.get_title() == "My Recording"
        assert ax.get_xlabel() == "Time (seconds)"
        assert ax.get_ylabel() == "Amplitude"

    def test_x_axis_spans_the_recording(self, clean_audio, sample_rate):
        _, ax = ss.plot_waveform_with_issues(clean_audio, sample_rate)
        _, right = ax.get_xlim()
        # The axis ends on the last sample, at (n - 1) / sr.
        assert right == (len(clean_audio) - 1) / sample_rate

    def test_click_markers_are_drawn(self, clean_audio, sample_rate):
        clicks = [0.25, 0.75, 1.5]
        _, ax = ss.plot_waveform_with_issues(
            clean_audio, sample_rate, click_timestamps=clicks
        )
        # One Line2D for the waveform plus one vertical line per click.
        assert len(ax.lines) == 1 + len(clicks)

    def test_saturation_spans_are_drawn(self, clean_audio, sample_rate):
        _, ax = ss.plot_waveform_with_issues(
            clean_audio, sample_rate, saturation_intervals=[(0.1, 0.3), (1.0, 1.2)]
        )
        assert len(ax.patches) == 2

    def test_legend_only_appears_when_issues_are_overlaid(self, clean_audio, sample_rate):
        _, plain = ss.plot_waveform_with_issues(clean_audio, sample_rate)
        assert plain.get_legend() is None

        _, annotated = ss.plot_waveform_with_issues(
            clean_audio, sample_rate, click_timestamps=[0.5]
        )
        assert annotated.get_legend() is not None

    def test_empty_audio_does_not_raise(self, empty_audio, sample_rate):
        fig, ax = ss.plot_waveform_with_issues(empty_audio, sample_rate)
        assert isinstance(fig, Figure)
        assert ax.get_xlim() == (0.0, 1.0)


class TestPlotQualityReport:
    def test_returns_a_two_panel_figure(self, clean_audio, sample_rate, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio))
        fig, (ax1, ax2) = ss.plot_quality_report(clean_audio, sample_rate, report)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert ax1.get_ylabel() == "Amplitude"
        assert ax2.get_ylabel() == "Energy (dB)"

    def test_title_includes_the_file_name_and_status(self, clean_audio, sample_rate, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio, "clean.wav"))
        _, (ax1, _) = ss.plot_quality_report(clean_audio, sample_rate, report)
        title = ax1.get_title()
        assert "clean.wav" in title
        assert report["status"] in title

    def test_detected_issues_are_overlaid(self, clean_audio, sample_rate, write_wav):
        clipped = ss.add_clipping(clean_audio, clip_threshold=0.01)
        report = ss.analyze_recording(write_wav(clipped, "clipped.wav"))
        _, (ax1, _) = ss.plot_quality_report(clipped, sample_rate, report)
        assert len(ax1.patches) == len(report["saturation"]["saturation_intervals"])

    def test_energy_panel_marks_the_measured_levels(self, clean_audio, sample_rate, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio))
        _, (_, ax2) = ss.plot_quality_report(clean_audio, sample_rate, report)
        labels = [text.get_text() for text in ax2.get_legend().get_texts()]
        assert any("Noise Floor" in label for label in labels)
        assert any("Active Level" in label for label in labels)
        assert any("Silence Threshold" in label for label in labels)

    def test_clean_report_says_so_in_the_diagnostics_box(self, clean_audio, sample_rate, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio))
        _, (_, ax2) = ss.plot_quality_report(clean_audio, sample_rate, report)
        annotations = " ".join(text.get_text() for text in ax2.texts)
        assert "No quality issues detected." in annotations

    def test_issue_text_is_rendered_for_a_degraded_file(self, clean_audio, sample_rate, write_wav):
        noisy = ss.add_noise(clean_audio, target_snr_db=0)
        report = ss.analyze_recording(write_wav(noisy, "noisy.wav"))
        _, (_, ax2) = ss.plot_quality_report(noisy, sample_rate, report)
        annotations = " ".join(text.get_text() for text in ax2.texts)
        assert report["issues"]
        for issue in report["issues"]:
            assert issue in annotations

    def test_partial_report_falls_back_to_defaults(self, clean_audio, sample_rate):
        # Every lookup in the plot uses .get, so a bare dict must still render.
        fig, _ = ss.plot_quality_report(clean_audio, sample_rate, {})
        assert isinstance(fig, Figure)

    def test_empty_audio_does_not_raise(self, empty_audio, sample_rate):
        fig, _ = ss.plot_quality_report(empty_audio, sample_rate, {})
        assert isinstance(fig, Figure)

    def test_audio_shorter_than_one_frame_does_not_raise(self, sample_rate):
        # The manual framing loop must degrade gracefully below frame_size.
        tiny = np.zeros(100, dtype=np.float32)
        fig, _ = ss.plot_quality_report(tiny, sample_rate, {})
        assert isinstance(fig, Figure)
