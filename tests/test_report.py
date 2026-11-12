"""Tests for report.py: the part of this repo that runs anywhere and is easiest to break.

report.py turns the harness' JSON into RESULTS.md. Its inputs are plain files, so the checks below
build a scratch results/ tree, point report at it, and read the markdown back. Stdlib only, no
torch, no GPU: `python -m unittest discover -s tests` (or `./run.sh test`).
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import report  # noqa: E402


def crossing(dataset, seed, cross, speedup, flash_mse=0.5, flash_s=1.0, em_s=0.1):
    """One results/crossing/*.json record, shaped like paper_bench.crossing writes them."""
    return dict(dataset=dataset, seed=seed, crossing=cross, speedup=speedup,
                flash=dict(test_mse=flash_mse, seconds=flash_s),
                emeans=[dict(epochs=2.0, seconds=em_s, test_mse=flash_mse - 1e-4,
                             input="cast", batch=32768)])


def figure3(dataset, method, seed, points):
    """One results/figure3/*.json record, shaped like paper_bench.figure3_runs writes them."""
    return dict(dataset=dataset, method=method, seed=seed,
                points=[dict(budget=b, train=t, test=t_) for b, t, t_ in points])


class Scratch(unittest.TestCase):
    """Each test gets its own results tree and its own RESULTS.md."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="emeans-report-")
        self.out = os.path.join(self.tmp, "results")
        os.makedirs(os.path.join(self.out, "crossing"))
        os.makedirs(os.path.join(self.out, "runs"))
        os.makedirs(os.path.join(self.out, "figure3"))
        self._saved = (report.OUT, report.REPO)
        report.OUT, report.REPO = self.out, self.tmp

    def tearDown(self):
        report.OUT, report.REPO = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, kind, name, payload):
        with open(os.path.join(self.out, kind, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def render(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):      # report.main() narrates; keep the suite quiet
            self.assertEqual(report.main(), 0)
        with open(os.path.join(self.tmp, "RESULTS.md"), encoding="utf-8") as fh:
            return fh.read()


class TestScale(unittest.TestCase):
    def test_scale_is_a_token_not_a_suffix(self):
        # The bug this guards: dino10b-1M-div4096 does not end in "-1M".
        self.assertEqual(report.scale_of("dino10b-1M-div4096"), "1M")
        self.assertEqual(report.scale_of("deep1b-10M"), "10M")
        self.assertEqual(report.scale_of("spacev-100M"), "100M")

    def test_ten_m_is_not_read_as_one_m(self):
        self.assertEqual(report.scale_of("sift1b-10M-div512"), "10M")

    def test_unknown_scale_is_its_own_bucket(self):
        self.assertEqual(report.scale_of("something-else"), "?")


class TestTable3(Scratch):
    def test_scales_never_share_a_table(self):
        self.write("crossing", "deep1b-1M_cast_s10.json", crossing("deep1b-1M", 10, 1.8, 13.9))
        self.write("crossing", "deep1b-10M_cast_s10.json", crossing("deep1b-10M", 10, 2.2, 13.8))
        text = self.render()

        self.assertIn("### 1M rows", text)
        self.assertIn("### 10M rows", text)
        # the 1M heading must come before the 1M row, which must come before the 10M heading
        self.assertLess(text.index("### 1M rows"), text.index("| deep1b-1M |"))
        self.assertLess(text.index("| deep1b-1M |"), text.index("### 10M rows"))
        self.assertLess(text.index("### 10M rows"), text.index("| deep1b-10M |"))

    def test_unreached_seed_is_named_not_dropped(self):
        self.write("crossing", "a-1M_cast_s10.json", crossing("a-1M", 10, 2.0, 10.0))
        self.write("crossing", "a-1M_cast_s11.json", crossing("a-1M", 11, None, None))
        text = self.render()

        self.assertIn("## Seeds that never crossed", text)
        self.assertIn("- a-1M seed 11", text)

    def test_cell_counts_every_seed_but_only_measures_crossings(self):
        recs = [crossing("a-1M", s, c, sp) for s, c, sp in
                ((10, 1.6, 10.0), (11, 2.0, 20.0), (12, None, None))]
        cell = report.cell(recs)

        self.assertEqual(cell["seeds"], 3)
        self.assertEqual(cell["crossed"], 2)
        self.assertEqual(cell["crossing"], 2.0)      # upper median of [1.6, 2.0]
        self.assertEqual(cell["speedup"], 15.0)      # median of the crossed seeds only


class TestFigure3(Scratch):
    def test_missing_arm_renders_a_dash_not_a_silent_average(self):
        self.write("figure3", "a-1M_emeans_cast_s10.json",
                   figure3("a-1M", "emeans", 10, [(1, 0.9, 0.8), (25, 0.5, 0.4)]))
        text = self.render()

        self.assertIn("### a-1M (1 e-means seeds, 0 Faiss seeds)", text)
        self.assertIn("| 25 | 0.5 | 0.4 | - | - |", text)

    def test_two_arms_and_two_seeds_are_averaged(self):
        for seed, base in ((10, 0.5), (11, 0.7)):
            self.write("figure3", "a-1M_emeans_cast_s%d.json" % seed,
                       figure3("a-1M", "emeans", seed, [(25, base, base)]))
            self.write("figure3", "a-1M_faiss_s%d.json" % seed,
                       figure3("a-1M", "faiss", seed, [(25, base + 0.2, base + 0.2)]))
        text = self.render()

        self.assertIn("(2 e-means seeds, 2 Faiss seeds)", text)
        self.assertIn("| 25 | 0.6 | 0.6 | 0.8 | 0.8 |", text)


if __name__ == "__main__":
    unittest.main()
