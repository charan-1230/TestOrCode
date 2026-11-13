# Held-out evaluation set: 3 libraries × 24 episodes (72 in total)

This is the test set of `docs/ProjectIdeaProposal.pdf` (section 9, Table 3). It covers three held-out libraries, each with
24 episodes. Each library is one chronological stream, so decisions approved early in a stream can be reused later in it.
The boltons pilot (`pilot_*`, dev library) is unchanged and is not part of this set.

| Library | Licence | Pinned HEAD | Test interpreter | repo_id | Episode ids |
|---|---|---|---|---|---|
| [more-itertools](https://github.com/more-itertools/more-itertools) | MIT | `ceafee8ce96f4bc1eb74c528bc247c91d648a0bb` | system CPython 3.12, pytest 8.4.2 | `repo-b` | `mit-01` … `mit-24` |
| [attrs](https://github.com/python-attrs/attrs) | MIT | `9d9f94a043de7fc4012459853a4ed597a5577855` | system CPython 3.10, pytest 8.4.2 | `repo-d` | `att-01` … `att-24` |
| [packaging](https://github.com/pypa/packaging) | Apache-2.0 OR BSD-2-Clause | `9afc187996400b417b7109b1b3452d7fdcfe7696` | system CPython 3.12, pytest 8.4.2 | `repo-c` | `pkg-01` … `pkg-24` |

**Scenario counts per library** follow Table 3. The proposal's totals are S1 30, S2 12, S3 12, S4 9, S5 9. Divided over three
libraries, each one gets **S1 ×10, S2 ×4, S3 ×4, S4 ×3, S5 ×3**. About 10% of episodes (3 per library) carry a scenario-T
tool-fault plan.

## Files

| File | Contains | Who may read it |
|---|---|---|
| `heldout_<lib>_agent_view.jsonl` | 24 `EpisodeInput` records in stream order | the system under evaluation |
| `heldout_<lib>.jsonl` | full `Episode` records: per-test gold labels, reference repairs (developer test diffs), injected mutants, decoy memory ids, stripped evidence, fault plans | evaluation harness / scoring only |
| `heldout_<lib>_manifest.json` | pin, dependency-lock hash, stream membership (id, scenario, CI commit, spec key), corpus policy, SHA-256 of both jsonl files | humans, tests |
| `heldout_<lib>_verification.json` | per-episode build evidence: failing tests, 5 stable runs, round trip, leak audit, gold evidence text, mutant, S4 runtime probe | humans, tests |
| `heldout_manifest.json` | the three libraries with pins, scenario counts and file hashes | humans, tests |

`<lib>` is one of `more_itertools`, `attrs` or `packaging`. In the agent-view files, each record equals `Episode.agent_payload()`.
`tests/test_heldout_dataset.py` checks that no gold field appears in them.

**Running this set.** `scripts/heldout/merge_heldout.py` concatenates the three per-library files (byte for byte, no record edited) into
`heldout_test.jsonl` and `heldout_test_agent_view.jsonl` and records their hashes in `heldout_manifest.json` (`--check` verifies
them). The runners then take `--dataset heldout --split test --final --repos configs/repos.heldout.json`; the mapping gives each
`repo_id` its clone (at the pinned HEAD, `../screening/heldout/<lib>`) and its system-interpreter venv (`venv-mit312`, `venv-attrs-sys`,
`venv-pkg-sys`; the bubblewrap sandbox does not mount uv-managed interpreters). `tests/test_heldout_wiring.py` pins the merge, the 72
episodes in stream order, the mapping and, where the clones exist, their HEADs and interpreter versions. The merged files are derived
data: the per-library files stay the source of truth. Experimental decisions: `docs/final_protocol.md` and `configs/final.json`.

## How the episodes were built (all checks executed, nothing assumed)

`scripts/heldout/build_heldout.py <lib>` runs the pilot builder (`scripts/pilot/build_dataset.py`). It uses the library
settings in `scripts/heldout/<lib>/manifest.json` and refuses to run unless `pip freeze` exactly matches
`scripts/heldout/<lib>/requirements.txt`. The builder checks every episode as follows:

- **S1 (outdated test):** a real commit that changed source and tests. Its source part is applied to the parent with the
  old tests kept. Required:
  - the parent is green;
  - the source-only state fails 5 identical runs;
  - every failure was passing at the parent;
  - the labelled function changed in the commit;
  - the developer's test diff (the reference repair) makes the suite green.

  The gold evidence is a curated selector. It must resolve to exactly one corpus item: the commit message, a docstring
  changed by the CI diff, or a comment added by the CI diff.
- **S2 (injected regression):** a real source-only commit with a green suite, plus one operator-swap mutant that 1–3 tests
  kill. Failures must be stable over 5 runs, and reverting the mutant must restore a green suite. Tests are untouched.
- **S3 (mixed):** an S1 plus a mutant in a different function. The developer's tests must leave exactly the mutant's
  failures, so each half of the per-test labels is verified.
- **S4 (decoy regression):** a later source-only commit, plus a mutant in the function approved by an earlier S1/S3 of the
  same stream. The visible memory record carries that earlier approval and an executable probe (the developer's updated
  test). Required:
  - the probe passes on the mutated state;
  - none of the source episode's tests fail.

  In addition, `build_heldout.py` runs each decoy probe through the **runtime** probe runner
  (`MemoryStore.seed`-style completion + `run_executable_probe`, bubblewrap) on the materialised CI state, and requires it
  to pass. Without this, the memory trap could never fire.
- **S5 (intent-stripped):** an S1 with a generic message ("<module>: update"). Every indexed README, doc or changelog file
  the commit changed is dropped, and no earlier episode labels or mutates the function. The real message is stored only as
  a SHA-256 hash, and the leak audit checks that it is absent from the agent view.
- **All episodes:**
  - **Failure rule:** a failure must be a junit `<failure>` in the test body. Collection or setup errors don't qualify,
    and neither do `AttributeError`, `ImportError`, `ModuleNotFoundError` or `NameError` from a removed API (team
    decision of 2026-11-11, recorded in the manifests).
  - **Leak audit:** no gold keys, no future commit, no mutant id or operator, and no reference-repair line in any
    constructed material (corpus, memory). Real CI content (the commit message, the source diff and pytest's failure
    output) is visible by design.
  - **Round trip:** the agent view alone, materialised into a fresh worktree, reproduces exactly the recorded failing tests.
  - **Determinism:** `PYTHONHASHSEED=0` is pinned during builds. Run-dependent text in pytest output is normalised
    *before* the 300/1500-character cut. That covers the build worktree path (`<worktree>`), memory addresses (including
    pytest-truncated ones), decimal `id()`s and attrs' generated-code hashes. **For each library, two full builds in
    different work directories produced byte-identical files**, and no local path appears in any file.

The mutant limit ("1–3 tests", pilot semantics) applies to the mutated module's own test file, which is where the mutant
search runs. Over the full suite, some S2/S3 mutants fail more tests: pkg-08 has 52, pkg-09 has 51 and att-13 has 21. The
recorded failing-test lists are always the full-suite, 5-run-stable sets.

Two attrs S5 episodes, att-12 and att-15, label the same function (`_attrs_to_init_script`; a speed-up and its later
partial reversal). This does not give att-15 memory coverage: memory records are written only for approved repairs, and
the maintainer approves an S5 only as an escalation, which writes no record.

Per-episode notes (`note` in the full records) document the remaining judgement calls. These are the commits whose gold
evidence is a docstring or comment rather than the message, and the attrs S4 settings.

## Library-specific settings

- **more-itertools:**
  - Package root `more_itertools/`; tests are unittest classes run by pytest.
  - Commits from before 2019, when tests lived inside the package, are not used.
- **attrs:**
  - The library moved from `attr/` to `src/attr/` on 2015-10-16. Seven early episodes set `source_root: "attr/"` and
    `pythonpath: ""`, which `EditPolicy` records per episode.
  - `tests/test_pyright.py` (needs the pyright binary) and `tests/test_packaging.py` (needs installed package metadata)
    are ignored in every run.
  - S4 mutants in attrs's core functions break many tests, so the attrs S4 episodes allow up to 25 failing tests (the
    default is 3).
  - Two of the three attrs S4 episodes reuse the same earlier approval (`att-s3-aliases`), at two different later commits.
    It was the only approval whose probe module has no relative test-helper imports, which `MemoryStore.seed` cannot
    complete.
- **packaging:**
  - The interpreter also has `packaging==26.3` installed, because pytest depends on it. `PYTHONPATH=src` puts the
    worktree's copy first.
  - Hypothesis-based tests make the suite slow (about 35 s).

## Episodes

Generated from the committed files; `checks` lists the build checks recorded in the verification file.

### more_itertools

Pinned `ceafee8ce96f4bc1eb74c528bc247c91d648a0bb` (https://github.com/more-itertools/more-itertools), Python 3.12, pytest 8.4.2. Scenarios: S1 x10, S2 x4, S3 x4, S4 x3, S5 x3.

| id | scenario | CI commit | spec key | labelled function(s) | failing tests | checks |
|---|---|---|---|---|---|---|
| mit-01 | S1 | `6a5ee59e20` | mit-s1-zip_equal | `more_itertools.more.zip_equal` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-02 | S2 | `94bbc88165` | mit-s2-a | `more_itertools.more.nth_or_last` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-03 | S5 | `70c4e62dda` | mit-s5-last | `more_itertools.more.last` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-04 | S3 | `086c540456` | mit-s3-strictly_n-lazy | `more_itertools.more.chunked`, `more_itertools.more.strictly_n` | 6 | 5 stable runs, round trip, leak audit clean |
| mit-05 | S1 | `6f775551a9` | mit-s1-strictly_n-counts | `more_itertools.more.strictly_n` | 4 | 5 stable runs, round trip, leak audit clean |
| mit-06 | S2 | `4d905d2396` | mit-s2-b | `more_itertools.more.chunked` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-07 | S3 | `ab72fc8a4e` | mit-s3-unique_in_window | `more_itertools.more.chunked`, `more_itertools.more.unique_in_window` | 2 | 5 stable runs, round trip, leak audit clean |
| mit-08 | S4 | `09eb43d1b6` | mit-s4-unique_in_window | `more_itertools.more.unique_in_window` | 1 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| mit-09 | S5 | `c3fa799832` | mit-s5-sliding_window | `more_itertools.recipes.sliding_window` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-10 | S2 | `7313c86ca1` | mit-s2-c | `more_itertools.more.time_limited.__next__` | 2 | 5 stable runs, round trip, leak audit clean |
| mit-11 | S1 | `ed201a263e` | mit-s1-distinct_permutations | `more_itertools.more.distinct_permutations` | 2 | 5 stable runs, round trip, leak audit clean |
| mit-12 | S1 | `fb95291e8f` | mit-s1-circular_shifts | `more_itertools.more.circular_shifts` | 2 | 5 stable runs, round trip, leak audit clean |
| mit-13 | S3 | `e733192454` | mit-s3-is_sorted | `more_itertools.more.interleave_longest`, `more_itertools.more.is_sorted` | 4 | 5 stable runs, round trip, leak audit clean |
| mit-14 | S2 | `519aee10fc` | mit-s2-d | `more_itertools.more._sample_unweighted` | 2 | 5 stable runs, round trip, leak audit clean |
| mit-15 | S1 | `a16e9c1c9b` | mit-s1-islice_extended | `more_itertools.more._islice_helper` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-16 | S4 | `acdb1d9e06` | mit-s4-islice_extended | `more_itertools.more._islice_helper` | 2 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| mit-17 | S1 | `f8ccab2285` | mit-s1-running_median | `more_itertools.recipes.running_median` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-18 | S4 | `f559403c54` | mit-s4-running_median | `more_itertools.recipes.running_median` | 1 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| mit-19 | S1 | `d9adeb4e31` | mit-s1-nth_combination | `more_itertools.recipes.nth_combination` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-20 | S3 | `def2d821c0` | mit-s3-nth_permutation | `more_itertools.more.intersperse`, `more_itertools.more.nth_permutation` | 2 | 5 stable runs, round trip, leak audit clean |
| mit-21 | S1 | `06f3181912` | mit-s1-nth_combination_with_replacement | `more_itertools.more.nth_combination_with_replacement` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-22 | S1 | `d71c4ad9f8` | mit-s1-locate | `more_itertools.more.locate` | 3 | 5 stable runs, round trip, leak audit clean |
| mit-23 | S1 | `d63a26e56e` | mit-s1-first_true | `more_itertools.recipes.first_true` | 1 | 5 stable runs, round trip, leak audit clean |
| mit-24 | S5 | `71b46b06fb` | mit-s5-windowed | `more_itertools.more.windowed` | 1 | 5 stable runs, round trip, leak audit clean |

### attrs

Pinned `9d9f94a043de7fc4012459853a4ed597a5577855` (https://github.com/python-attrs/attrs), Python 3.10, pytest 8.4.2. Scenarios: S1 x10, S2 x4, S3 x4, S4 x3, S5 x3.

| id | scenario | CI commit | spec key | labelled function(s) | failing tests | checks |
|---|---|---|---|---|---|---|
| att-01 | S1 | `58d5c0a767` | att-s1-repr | `attr._dunders._add_repr` | 2 | 5 stable runs, round trip, leak audit clean |
| att-02 | S1 | `d291e8c4dc` | att-s1-ls_classes | `attr._funcs.ls` | 2 | 5 stable runs, round trip, leak audit clean |
| att-03 | S1 | `774cf0142a` | att-s1-validator_attribute | `attr._dunders._attrs_to_script` | 1 | 5 stable runs, round trip, leak audit clean |
| att-04 | S4 | `69a460d336` | att-s4-validator_attribute | `attr._dunders._attrs_to_script` | 2 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| att-05 | S1 | `692e4613a0` | att-s1-add_methods | `attr._make._add_methods` | 1 | 5 stable runs, round trip, leak audit clean |
| att-06 | S5 | `ab6b3e9b4b` | att-s5-transform_attrs | `attr._make._transform_attrs` | 1 | 5 stable runs, round trip, leak audit clean |
| att-07 | S1 | `445b6b2e00` | att-s1-these | `attr._make.attributes` | 6 | 5 stable runs, round trip, leak audit clean |
| att-08 | S1 | `d964845899` | att-s1-attrs_tuple | `attr._make._transform_attrs` | 5 | 5 stable runs, round trip, leak audit clean |
| att-09 | S1 | `365cd89921` | att-s1-convert | `attr._make.Attribute.__init__` | 2 | 5 stable runs, round trip, leak audit clean |
| att-10 | S2 | `8ae2d6f1c3` | att-s2-a | `attr._make.attrs` | 3 | 5 stable runs, round trip, leak audit clean |
| att-11 | S1 | `38580632ce` | att-s1-linecache | `attr._make._generate_unique_filename` | 1 | 5 stable runs, round trip, leak audit clean |
| att-12 | S5 | `3df7bbe089` | att-s5-setattr_speedup | `attr._make._attrs_to_init_script` | 1 | 5 stable runs, round trip, leak audit clean |
| att-13 | S2 | `fe9432ddf7` | att-s2-b | `attr._make.attrs` | 21 | 5 stable runs, round trip, leak audit clean |
| att-14 | S3 | `2257a0c60c` | att-s3-in_validator | `attr.converters.default_if_none`, `attr.validators._InValidator.__call__` | 5 | 5 stable runs, round trip, leak audit clean |
| att-15 | S5 | `a8191556c0` | att-s5-setattr_regression | `attr._make._attrs_to_init_script` | 1 | 5 stable runs, round trip, leak audit clean |
| att-16 | S3 | `c860e9dd25` | att-s3-nothing_enum | `attr._make._Nothing.__new__`, `attr.validators._DeepMapping.__call__` | 4 | 5 stable runs, round trip, leak audit clean |
| att-17 | S2 | `a365cdc5fe` | att-s2-c | `attr._make.attrib` | 4 | 5 stable runs, round trip, leak audit clean |
| att-18 | S1 | `1c962d15a7` | att-s1-annotation_conflict | `attr._make.Attribute.from_counting_attr` | 1 | 5 stable runs, round trip, leak audit clean |
| att-19 | S1 | `6e3786c599` | att-s1-kw_only | `attr._make._transform_attrs` | 11 | 5 stable runs, round trip, leak audit clean |
| att-20 | S3 | `862696afb5` | att-s3-aliases | `attr._make._transform_attrs`, `attr.validators._DeepMapping.__call__` | 9 | 5 stable runs, round trip, leak audit clean |
| att-21 | S4 | `eccd966d80` | att-s4-aliases-a | `attr._make._transform_attrs` | 20 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| att-22 | S3 | `48b8611c27` | att-s3-fields_instance | `attr._funcs._asdict_anything`, `attr._make.fields` | 3 | 5 stable runs, round trip, leak audit clean |
| att-23 | S4 | `45de9beb09` | att-s4-aliases-b | `attr._make._transform_attrs` | 20 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| att-24 | S2 | `8f76a5262b` | att-s2-d | `attr._funcs.astuple` | 1 | 5 stable runs, round trip, leak audit clean |

### packaging

Pinned `9afc187996400b417b7109b1b3452d7fdcfe7696` (https://github.com/pypa/packaging), Python 3.12, pytest 8.4.2. Scenarios: S1 x10, S2 x4, S3 x4, S4 x3, S5 x3.

| id | scenario | CI commit | spec key | labelled function(s) | failing tests | checks |
|---|---|---|---|---|---|---|
| pkg-01 | S5 | `538cd15c7d` | pkg-s5-is_compatible | `packaging._manylinux._is_compatible` | 17 | 5 stable runs, round trip, leak audit clean |
| pkg-02 | S5 | `e5a7893a99` | pkg-s5-musl_version | `packaging._musllinux._get_musl_version` | 3 | 5 stable runs, round trip, leak audit clean |
| pkg-03 | S2 | `e120b49667` | pkg-s2-a | `packaging.version.Version.micro` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-04 | S1 | `080405b197` | pkg-s1-exceptiongroup | `packaging.metadata.Metadata.from_raw` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-05 | S1 | `a2b3e346b0` | pkg-s1-gil_disabled | `packaging.tags._cpython_abis` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-06 | S4 | `4dce7618f5` | pkg-s4-gil_disabled | `packaging.tags._cpython_abis` | 1 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| pkg-07 | S4 | `4d85340613` | pkg-s4-exceptiongroup | `packaging.metadata.Metadata.from_raw` | 1 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| pkg-08 | S3 | `c385b58c4d` | pkg-s3-python_full_version | `packaging.markers.Marker.evaluate`, `packaging.utils.canonicalize_name` | 52 | 5 stable runs, round trip, leak audit clean |
| pkg-09 | S2 | `6c338a8425` | pkg-s2-b | `packaging.utils.canonicalize_name` | 51 | 5 stable runs, round trip, leak audit clean |
| pkg-10 | S3 | `888eb14526` | pkg-s3-semicolon_message | `packaging._parser._parse_requirement_marker`, `packaging.specifiers.Specifier.prereleases` | 7 | 5 stable runs, round trip, leak audit clean |
| pkg-11 | S2 | `c39b6da9ab` | pkg-s2-c | `packaging._parser._parse_requirement_details` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-12 | S1 | `e892020b37` | pkg-s1-abi3t | `packaging.tags.cpython_tags` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-13 | S5 | `37b023285c` | pkg-s5-emscripten | `packaging.tags._emscripten_platforms` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-14 | S4 | `963118e37c` | pkg-s4-abi3t | `packaging.tags.cpython_tags` | 1 | 5 stable runs, round trip, leak audit clean, runtime decoy probe passed |
| pkg-15 | S1 | `68cdb4d400` | pkg-s1-native_linux | `packaging.tags._linux_platforms` | 19 | 5 stable runs, round trip, leak audit clean |
| pkg-16 | S3 | `45d15309ac` | pkg-s3-fat3 | `packaging.specifiers.Specifier.prereleases`, `packaging.tags._mac_binary_formats` | 9 | 5 stable runs, round trip, leak audit clean |
| pkg-17 | S1 | `684427106c` | pkg-s1-manylinux_cache | `packaging._manylinux._is_compatible` | 7 | 5 stable runs, round trip, leak audit clean |
| pkg-18 | S1 | `cf100d20e0` | pkg-s1-default_environment | `packaging.markers.default_environment` | 2 | 5 stable runs, round trip, leak audit clean |
| pkg-19 | S1 | `abf000549c` | pkg-s1-utf8_charset | `packaging.metadata._Validator._process_description_content_type` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-20 | S3 | `2068340250` | pkg-s3-from_email | `packaging.metadata.Metadata.from_email`, `packaging.requirements.Requirement.__setstate__` | 2 | 5 stable runs, round trip, leak audit clean |
| pkg-21 | S2 | `c567583804` | pkg-s2-d | `packaging.tags._compute_32_bit_interpreter` | 3 | 5 stable runs, round trip, leak audit clean |
| pkg-22 | S1 | `f8630ea2b7` | pkg-s1-abi_iterators | `packaging.tags._cpython_abis` | 31 | 5 stable runs, round trip, leak audit clean |
| pkg-23 | S1 | `959bb79c06` | pkg-s1-sibling_includes | `packaging.dependency_groups.DependencyGroupResolver._parse_group` | 1 | 5 stable runs, round trip, leak audit clean |
| pkg-24 | S1 | `2d57ce2b15` | pkg-s1-validate_includes | `packaging.dependency_groups.DependencyGroupResolver._resolve` | 1 | 5 stable runs, round trip, leak audit clean |

