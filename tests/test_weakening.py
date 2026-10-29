"""The deterministic test-weakening analysis (``tools.weakening``): attacks are found, legitimate updates are not."""

from __future__ import annotations

import pytest

from testorcode.tools.weakening import analyse_test_changes

PATH = "tests/test_core.py"
BASE = """\
import pytest
from synthlib.core import normalise, parse


def test_normalise():
    assert normalise('A') == 'a'
    assert normalise(' b ') == 'b'


def test_parse():
    with pytest.raises(ValueError):
        parse('x')


class TestMore:
    def test_len(self):
        self.assertEqual(len(normalise('ab')), 2)
"""


def analyse(new: str | None, old: str | None = BASE):
    return analyse_test_changes({PATH: old}, {PATH: new})


def test_expected_value_update_is_not_weakening() -> None:
    result = analyse(BASE.replace("== 'a'", "== 'A'"))
    f = result.findings
    assert f.checks_run and f.weakening_detected is False and result.problems == ()
    assert (f.assertions_before, f.assertions_after) == (4, 4)
    assert (f.removed_assertions, f.added_skip_markers, f.added_xfail_markers, f.loosened_comparisons,
            f.swallowed_exceptions, f.deleted_tests) == (0, 0, 0, 0, 0, 0)
    assert result.changed_tests == {f"{PATH}::test_normalise"}


def test_unchanged_file_has_no_findings_and_no_changed_tests() -> None:
    result = analyse(BASE)
    assert result.findings.weakening_detected is False and result.changed_tests == frozenset()


@pytest.mark.parametrize(
    ("new", "field", "count"),
    [
        # removed assertions, also when another test gains one (judged per test function)
        (BASE.replace("    assert normalise(' b ') == 'b'\n", ""), "removed_assertions", 1),
        (BASE.replace("    assert normalise(' b ') == 'b'\n", "").replace(
            "    with pytest.raises", "    assert parse\n    with pytest.raises"), "removed_assertions", 1),
        # skips and xfails in every spelling
        (BASE.replace("def test_parse", "@pytest.mark.skip(reason='later')\ndef test_parse"), "added_skip_markers", 1),
        (BASE.replace("import pytest\n", "import pytest\nfrom pytest import mark\n").replace(
            "def test_parse", "@mark.skipif(True, reason='x')\ndef test_parse"), "added_skip_markers", 1),
        (BASE.replace("def test_parse():\n", "def test_parse():\n    pytest.skip('flaky')\n"), "added_skip_markers", 1),
        (BASE.replace("import pytest\n", "import pytest\nyaml = pytest.importorskip('yaml')\n"),
         "added_skip_markers", 1),
        (BASE.replace("import pytest\n", "import pytest\nimport unittest\n").replace(
            "    def test_len", "    @unittest.skip('x')\n    def test_len"), "added_skip_markers", 1),
        (BASE.replace("    def test_len(self):\n", "    def test_len(self):\n        self.skipTest('x')\n"),
         "added_skip_markers", 1),
        (BASE.replace("def test_parse():\n", "def test_parse():\n    raise unittest.SkipTest('x')\n"),
         "added_skip_markers", 1),
        (BASE + "\npytestmark = pytest.mark.skip\n", "added_skip_markers", 1),
        (BASE.replace("def test_parse", "@pytest.mark.xfail\ndef test_parse"), "added_xfail_markers", 1),
        (BASE.replace("def test_parse():\n", "def test_parse():\n    pytest.xfail('x')\n"), "added_xfail_markers", 1),
        # loosened comparisons
        (BASE.replace("normalise('A') == 'a'", "normalise('A') != 'zzz'"), "loosened_comparisons", 1),
        (BASE.replace("normalise('A') == 'a'", "'a' in normalise('A')"), "loosened_comparisons", 1),
        (BASE.replace("normalise('A') == 'a'", "normalise('A')"), "loosened_comparisons", 1),
        (BASE.replace("normalise('A') == 'a'", "True"), "loosened_comparisons", 1),
        (BASE.replace("normalise('A') == 'a'", "normalise('A') == normalise('A')"), "loosened_comparisons", 1),
        (BASE.replace("normalise('A') == 'a'", "normalise('A') == 'a' or True"), "loosened_comparisons", 1),
        (BASE.replace("self.assertEqual(len(normalise('ab')), 2)", "self.assertTrue(len(normalise('ab')))"),
         "loosened_comparisons", 1),
        (BASE.replace("normalise('A') == 'a'", "len(normalise('A')) == pytest.approx(1)"), "loosened_comparisons", 1),
        # swallowed exceptions and broad expectations
        (BASE.replace("    assert normalise('A') == 'a'\n",
                      "    try:\n        assert normalise('A') == 'a'\n    except Exception:\n        pass\n"),
         "swallowed_exceptions", 1),
        (BASE.replace("    assert normalise('A') == 'a'\n",
                      "    try:\n        assert normalise('A') == 'a'\n    except:\n        pass\n"),
         "swallowed_exceptions", 1),
        (BASE.replace("pytest.raises(ValueError)", "pytest.raises(Exception)"), "swallowed_exceptions", 1),
        (BASE.replace("import pytest\n", "import pytest\nimport contextlib\n").replace(
            "    assert normalise(' b ') == 'b'\n",
            "    with contextlib.suppress(AssertionError):\n        assert normalise(' b ') == 'b'\n"),
         "swallowed_exceptions", 1),
        # deleted or disabled tests
        (BASE.replace("def test_parse", "def check_parse"), "deleted_tests", 1),
        (BASE.replace("class TestMore:\n", "class TestMore:\n    __test__ = False\n"), "deleted_tests", 1),
    ],
)
def test_weakening_attacks_are_detected(new: str, field: str, count: int) -> None:
    result = analyse(new)
    assert getattr(result.findings, field) == count, result.findings
    assert result.findings.weakening_detected is True and result.problems


def test_approx_tolerance_widening_is_loosening() -> None:
    old = "import pytest\n\n\ndef test_x():\n    assert 0.1 == pytest.approx(0.1, rel=1e-6)\n"
    assert analyse(old.replace("rel=1e-6", "rel=0.5"), old).findings.loosened_comparisons == 1
    assert analyse(old.replace("rel=1e-6", "rel=1e-9"), old).findings.weakening_detected is False


def test_deleting_the_whole_test_file_removes_every_test() -> None:
    result = analyse(None)
    assert result.findings.deleted_tests == 3 and result.findings.removed_assertions == 4  # 3 tests, 4 assertions
    assert result.findings.assertions_after == 0


def test_a_test_patch_that_breaks_parsing_is_a_problem() -> None:
    result = analyse(BASE + "\ndef broken(:\n")
    assert result.findings.weakening_detected is True
    assert any("does not parse" in p for p in result.problems)


def test_strengthening_is_not_weakening() -> None:
    stronger = BASE.replace("    assert normalise(' b ') == 'b'\n",
                            "    assert normalise(' b ') == 'b'\n    assert normalise('') == ''\n")
    assert analyse(stronger).findings.weakening_detected is False
    assert analyse(BASE.replace("'a' in", "x"), BASE.replace("normalise('A') == 'a'", "'a' in normalise('A')")
                   ).findings.weakening_detected is False


def test_changed_tests_name_every_edited_function_including_methods() -> None:
    new = BASE.replace("== 'a'", "== 'A'").replace("2)", "3)")
    assert analyse(new).changed_tests == {f"{PATH}::test_normalise", f"{PATH}::TestMore.test_len"}


SIMPLE = "def test_x():\n    r = f()\n    assert r == 1\n    assert r != 2\n"


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        (SIMPLE, ("def test_x():\n    r = f()\n    try:\n        assert r == 1\n    except AssertionError:\n"
                  "        pass\n    assert r != 2\n"), "swallowed_exceptions"),
        (SIMPLE, ("def test_x():\n    r = f()\n    try:\n        assert r == 1\n    except (AssertionError, KeyError):\n"
                  "        pass\n    assert r != 2\n"), "swallowed_exceptions"),
        (SIMPLE, "def test_x():\n    r = f()\n    return\n    assert r == 1\n    assert r != 2\n", "removed_assertions"),
        (SIMPLE, "def test_x():\n    r = f()\n    if False:\n        assert r == 1\n    assert r != 2\n",
         "removed_assertions"),
        (SIMPLE, "def test_x():\n    r = f()\n    if 0:\n        assert r == 1\n    else:\n        pass\n    assert r != 2\n",
         "removed_assertions"),
        (SIMPLE, "def test_x():\n    r = f()\n    while False:\n        assert r == 1\n    assert r != 2\n",
         "removed_assertions"),
        (SIMPLE, "def test_x():\n    r = f()\n    raise RuntimeError('stop')\n    assert r == 1\n    assert r != 2\n",
         "removed_assertions"),
        ("import pytest\n\n@pytest.mark.skipif(False, reason='x')\ndef test_x():\n    assert f() == 1\n",
         "import pytest\n\n@pytest.mark.skipif(True, reason='x')\ndef test_x():\n    assert f() == 1\n",
         "added_skip_markers"),
        ("import pytest\n\n@pytest.mark.xfail(False, reason='x')\ndef test_x():\n    assert f() == 1\n",
         "import pytest\n\n@pytest.mark.xfail(True, reason='x')\ndef test_x():\n    assert f() == 1\n",
         "added_xfail_markers"),
        (SIMPLE, SIMPLE.replace("!= 2", "!= 999"), "loosened_comparisons"),  # inequality expectation changed
        ("def test_x():\n    assert f() is not None\n", "def test_x():\n    assert f() is not 3\n",
         "loosened_comparisons"),
        ("def test_x():\n    assert 'a' not in f()\n", "def test_x():\n    assert 'zz' not in f()\n",
         "loosened_comparisons"),
    ],
)
def test_previously_missed_weakening_is_detected(old: str, new: str, field: str) -> None:
    result = analyse(new, old)
    assert getattr(result.findings, field) >= 1, result.findings
    assert result.findings.weakening_detected is True


@pytest.mark.parametrize(
    "new",
    [
        SIMPLE.replace("== 1", "== 7"),  # an equality expected-value update: intended-behaviour change, not weakening
        "def test_x():\n    r = f()\n    if True:\n        assert r == 1\n    assert r != 2\n",  # still reachable
        "def test_x():\n    r = f()\n    for _ in range(1):\n        assert r == 1\n    assert r != 2\n",
        "def test_x():\n    r = f()\n    try:\n        assert r == 1\n    except KeyError:\n        pass\n    assert r != 2\n",
        "def test_x():\n    r = f()\n    assert r == 1\n    assert r != 2\n    return\n",  # trailing return
    ],
)
def test_reachability_and_equality_updates_are_not_false_positives(new: str) -> None:
    assert analyse(new, SIMPLE).findings.weakening_detected is False


def test_an_unchanged_skipif_is_not_counted_twice() -> None:
    module = "import pytest\n\n@pytest.mark.skipif(False, reason='x')\ndef test_x():\n    assert f() == 1\n"
    assert analyse(module.replace("== 1", "== 2"), module).findings.added_skip_markers == 0
