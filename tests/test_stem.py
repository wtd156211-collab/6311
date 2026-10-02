"""stem.py 的验收与单元测试（unittest，纯标准库）。

运行：python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import stem  # noqa: E402

SAMPLES = ROOT / "samples"
VOCAB = SAMPLES / "words" / "vocab.tsv"
FAMILIES = SAMPLES / "words" / "families.tsv"
SCALE = SAMPLES / "words" / "scale.tsv"
CASES = SAMPLES / "cases"


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "stem.py"), *argv],
        capture_output=True, text=True, cwd=ROOT,
    )



def _write(directory, name, text):
    path = Path(directory, name)
    path.write_text(text, encoding="utf-8")
    return str(path)

class TestCliAcceptance(unittest.TestCase):
    """README 第五节的逐字节验收口径。"""

    def test_run_vocab_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli("run", "--words", str(VOCAB), "--out", tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(
                proc.stdout,
                (CASES / "run_summary.expected.txt").read_text(encoding="utf-8"),
            )
            for name in ("stem.tsv", "index.tsv"):
                self.assertEqual(
                    Path(tmp, name).read_bytes(),
                    (CASES / f"{name.split('.')[0]}_expected.tsv").read_bytes(),
                    name,
                )

    def test_run_scale_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli("run", "--words", str(SCALE), "--out", tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(
                Path(tmp, "stem.tsv").read_bytes(),
                (CASES / "scale_expected.tsv").read_bytes(),
            )

    def test_recall_byte_exact(self):
        proc = run_cli(
            "recall", "--words", str(VOCAB),
            "--queries", str(CASES / "query_cases.txt"),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout.encode(),
            (CASES / "query_expected.tsv").read_bytes(),
        )

    def test_determinism_across_hash_seeds(self):
        outputs = []
        for seed in ("0", "1", "2"):
            with tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ, PYTHONHASHSEED=seed)
                subprocess.run(
                    [sys.executable, str(ROOT / "stem.py"), "run",
                     "--words", str(VOCAB), "--out", tmp],
                    check=True, capture_output=True, cwd=ROOT, env=env,
                )
                subprocess.run(
                    [sys.executable, str(ROOT / "stem.py"), "web",
                     "--words", str(VOCAB), "--families", str(FAMILIES),
                     "--out", str(Path(tmp, "index.html"))],
                    check=True, capture_output=True, cwd=ROOT, env=env,
                )
                outputs.append(tuple(
                    Path(tmp, name).read_bytes()
                    for name in ("stem.tsv", "index.tsv", "index.html")
                ))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])


class TestGuards(unittest.TestCase):
    """过度归并守卫与标注一致性。"""

    @classmethod
    def setUpClass(cls):
        words = stem.load_words(str(VOCAB))
        cls.stemmer = stem.build_stemmer(words)

    def test_overmerge_guard_zero_violations(self):
        for lineno, line in enumerate(
            (CASES / "overmerge_guard.tsv").read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line or line.startswith("#"):
                continue
            left, right, _reason = line.split("\t")
            self.assertNotEqual(
                self.stemmer.normalize(left), self.stemmer.normalize(right),
                f"overmerge_guard.tsv:{lineno}: {left} 与 {right} 被合并",
            )

    def test_families_fully_consistent(self):
        labels = stem.load_families(str(FAMILIES))
        for word, label in labels.items():
            self.assertEqual(self.stemmer.normalize(word), label, word)


class TestPerformance(unittest.TestCase):
    """性能预算：1 万词形装载+归一+建索引 ≤1.5s，峰值内存 ≤256MiB，
    1000 条召回 ≤0.5s。"""

    def test_scale_budget(self):
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        start = time.perf_counter()
        words = stem.load_words(str(SCALE))
        stemmer = stem.build_stemmer(words)
        _, ordered = stem.build_model(words, stemmer)
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] - baseline
        tracemalloc.stop()
        self.assertLessEqual(elapsed, 1.5, f"耗时 {elapsed:.3f}s")
        self.assertLessEqual(peak, 256 * 2**20, f"峰值内存 {peak / 2**20:.1f} MiB")

        queries = [w for w, _ in words][:1000]
        start = time.perf_counter()
        for query in queries:
            ordered.get(stemmer.normalize(query), [])
        self.assertLessEqual(time.perf_counter() - start, 0.5)


class TestStemmerRules(unittest.TestCase):
    """规则与例外的单元口径。"""

    @classmethod
    def setUpClass(cls):
        words = stem.load_words(str(VOCAB))
        cls.stemmer = stem.build_stemmer(words)

    def normalize(self, word):
        return self.stemmer.normalize(word)

    def test_plural_and_sibilant(self):
        self.assertEqual(self.normalize("cats"), "cat")
        self.assertEqual(self.normalize("boxes"), "box")
        self.assertEqual(self.normalize("houses"), "house")  # 走 s 规则
        self.assertEqual(self.normalize("classes"), "class")

    def test_doubling_vs_add_e(self):
        self.assertEqual(self.normalize("hopping"), "hop")
        self.assertEqual(self.normalize("hoping"), "hope")
        self.assertEqual(self.normalize("hopped"), "hop")
        self.assertEqual(self.normalize("hoped"), "hope")
        self.assertEqual(self.normalize("stopped"), "stop")

    def test_add_not_dedoubled(self):
        # 短化①：末位相同辅音但长度 < 4 不动。
        self.assertEqual(self.normalize("added"), "add")

    def test_adjective_gate(self):
        self.assertEqual(self.normalize("worker"), "worker")
        self.assertEqual(self.normalize("honest"), "honest")
        self.assertEqual(self.normalize("bigger"), "big")
        self.assertEqual(self.normalize("largest"), "large")

    def test_min_stem_guard(self):
        self.assertEqual(self.normalize("is"), "be")   # 例外
        self.assertEqual(self.normalize("this"), "this")  # 保护
        self.assertEqual(self.normalize("yes"), "yes")

    def test_exceptions_win(self):
        self.assertEqual(self.normalize("news"), "news")
        self.assertEqual(self.normalize("worse"), "bad")
        self.assertEqual(self.normalize("children"), "child")
        self.assertEqual(self.normalize("leaves"), "leaf")
        self.assertEqual(self.normalize("fewer"), "few")

    def test_derivations_not_merged(self):
        self.assertNotEqual(self.normalize("submit"), self.normalize("submission"))
        self.assertNotEqual(self.normalize("work"), self.normalize("worker"))
        self.assertNotEqual(self.normalize("teach"), self.normalize("teacher"))


class TestValidation(unittest.TestCase):
    """非法输入在产出前失败，退出码 1，stderr 含文件与行号。"""

    def test_invalid_word_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = _write(tmp, "w.tsv", "ok\tn\nBad Word\tn\n")
            proc = run_cli("run", "--words", words, "--out", str(Path(tmp, "o")))
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(proc.stdout, "")
            self.assertIn("w.tsv:2", proc.stderr)
            self.assertFalse(Path(tmp, "o", "stem.tsv").exists())

    def test_invalid_pos_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = _write(tmp, "w.tsv", "ok\tnoun\n")
            proc = run_cli("run", "--words", words)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("w.tsv:1", proc.stderr)

    def test_conflicting_exception_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "e.tsv", "ab\tab\t保护\nab\ta\t保护\n")
            with self.assertRaises(stem.InputError) as ctx:
                stem.load_exceptions(path)
            self.assertIn("e.tsv:2", str(ctx.exception))

    def test_usage_error_exit_2(self):
        proc = run_cli("run")
        self.assertEqual(proc.returncode, 2)

    def test_empty_vocab_is_legal(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = _write(tmp, "w.tsv", "# 只有注释\n\n")
            proc = run_cli("run", "--words", words, "--out", tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "words=0 families=0\n")


class TestEvaluationAndWeb(unittest.TestCase):
    """错误合并统计与页面数据来自引擎输出。"""

    def test_evaluate_counts_bad_pairs(self):
        index = {"hope": ["hope", "hoped"], "hop": ["hop"]}
        labels = {"hope": "hope", "hoped": "hop", "hop": "hop"}
        result = stem.evaluate(index, labels)
        self.assertEqual(result["bad_families"], {"hope"})
        self.assertEqual(result["bad_pairs"], 1)
        self.assertEqual(result["total_pairs"], 1)
        self.assertEqual(result["ratio"], 1.0)

    def test_evaluate_clean(self):
        index = {"cat": ["cat", "cats"]}
        labels = {"cat": "cat", "cats": "cat"}
        result = stem.evaluate(index, labels)
        self.assertEqual(result["bad_families"], set())
        self.assertEqual(result["ratio"], 0.0)

    def test_web_page_contains_engine_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp, "index.html"))
            proc = run_cli("web", "--words", str(VOCAB),
                           "--families", str(FAMILIES), "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            html = Path(out).read_text(encoding="utf-8")
            self.assertNotIn("/*__DATA__*/null", html)
            self.assertIn('"submit"', html)
            self.assertIn('"badPairs":0', html)
            self.assertIn('"stem":"submit"', html)

    def test_web_marks_bad_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = _write(tmp, "w.tsv", "hope\tv\nhoped\tv\nhop\tv\n")
            fams = _write(tmp, "f.tsv",
                               "hope\thope\t原形\nhoped\thop\t规则\nhop\thop\t原形\n")
            out = str(Path(tmp, "index.html"))
            proc = run_cli("web", "--words", words, "--families", fams,
                           "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            html = Path(out).read_text(encoding="utf-8")
            self.assertIn('"badPairs":1', html)
            self.assertIn('"bad":true', html)


class TestLoaders(unittest.TestCase):
    def test_duplicate_word_keeps_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "w.tsv")
            path.write_text("ok\tn\nok\tv\n", encoding="utf-8")
            words = stem.load_words(str(path))
            self.assertEqual(words, [("ok", frozenset({"n"}))])

    def test_preprocess(self):
        self.assertEqual(stem.preprocess("  Submits \n"), "submits")

    def test_recall_out_of_vocab_empty(self):
        words = stem.load_words(str(VOCAB))
        stemmer = stem.build_stemmer(words)
        _, ordered = stem.build_model(words, stemmer)
        self.assertEqual(ordered.get(stemmer.normalize("bank"), []), [])


if __name__ == "__main__":
    unittest.main()
