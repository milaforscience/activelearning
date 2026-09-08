# DOCK3 oracle: protomer handling — questions for a DOCK3 user

**Status:** the correctness concern is resolved empirically (see below). Two quality questions
remain for someone who knows the DOCK3 ligand pipeline.
**Code:** `src/activelearning/applications/molecules/dock3_oracle.py`
**Last updated:** 2026-09-08

## Background — what our code does

We wrap the DOCK3 toolchain as an oracle for an active-learning loop. Per SMILES string:

1. Write a one-line `.smi` file.
2. `source dockenv.sh && ligbuild <lig.smi> <out_dir> <custom_parms.json>`
   with `custom_parms = {"verbose": 1, "timeout": 150}`.
   `dockenv.sh` is `/project/rrg-mailhoto/share/dockingpackages/dockenv.sh`.
3. Untar the resulting `.tgz` bundle and take the `.db2` inside it.
4. Patch the receptor's INDOCK template (header `DOCK 3.7 parameter` → `DOCK 3.8 parameter`, and
   substitute `split_database_index` with the ligand path), then run
   `/project/rrg-mailhoto/share/dock64` against a private copy of the dockfiles.
5. Parse `OUTDOCK` and take the most negative `Total` across all pose lines.

Receptor is AmpC β-lactamase (`~/dock_smiles/ampc_dockfiles`). The pipeline is ported from a
standalone `dock_smiles` project, preserving its behaviour.

## The concern that was raised, and why it is closed

`_extract_db2` picks a single file out of the bundle:

```python
db2_files = list(extract_dir.rglob("*.db2"))
return db2_files[0]
```

`ligbuild` enumerates protomers (protonation states at physiological pH), and protonation sets the
formal charge, which dominates the electrostatic term. If a bundle held one `.db2` *per protomer*,
this would dock an arbitrary protonation state — and since `glob` returns unsorted filesystem
order, potentially a different one on each run.

**That is not what happens.** Two independent runs of the standalone pipeline docked the same 1000
molecules:

- `~/dock_smiles/all_scores.csv` — the full ~10M-molecule run
- `~/dock_smiles/dock_batch_62723188.csv` — a 1000-molecule benchmark

Of the 1000 molecules present in both, **998 have byte-identical scores**. The 2 that differ are
`0.0` (the old failure sentinel) versus a real score, i.e. one run failed to build or dock the
molecule — not a different protomer. There are **zero** score-vs-score disagreements.

Two runs on different nodes, with different scratch directories and therefore different directory
ordering, cannot agree 998 times if there were multiple `.db2` candidates to choose between. This is
consistent with the `.db2` format itself: it is a *database* format designed to hold many molecules
and conformations, which is why INDOCK's ligand field is named `split_database_index` and normally
points at a list of db2 files each holding thousands of entries. A single-molecule build producing
one db2 containing that molecule's forms is the format working as intended. The bundle path
(`bundle_lig_000/lig.db2`) numbers the input batch, not the protomer, and we submit one SMILES per
`.smi` file.

**Conclusion: our port does not silently pick among protomers, and its scores are reproducible.**

## What remains open

These are quality questions about the toolchain's behaviour, not correctness bugs in our wrapper.
The evidence above is equally consistent with "one db2 containing every protomer" and "one db2
containing only one form" — those differ in score quality, not in reproducibility.

1. **Does `ligbuild`, invoked as above, enumerate protomers and tautomers into the `.db2`?**
   If it builds only one form (e.g. the neutral one), our scores are consistent but may
   systematically misrepresent ionizable compounds. AmpC binders are typically anionic, so this
   matters for this receptor in particular.

2. **What do `number_save` and `number_write` control in `OUTDOCK`?** Our INDOCK template contains:

   ```
   number_save                   1
   number_write                  1
   ```

   We take the minimum `Total` over every pose line in `OUTDOCK`. If those settings write one line
   per molecule entry in the db2, that minimum is the best score across protomers, which is what we
   want. If they cap the whole run at a single written pose, we would be reporting one arbitrary
   entry even when the db2 contains several.

3. **Is best-across-protomers the right convention** for a virtual-screening-style objective here,
   or is there a reason to prefer a specific protonation state?

## The measurement that would answer 1 and 2

Inside a compute allocation, dock one molecule with an ionizable group — e.g.
`O=C(O)c1cc(Cl)cc(NC(=O)N2CCCC3(CC3)C2)c1`, taken from `all_scores.csv` — with `tmp_dir` set to a
short path so the work directory survives:

```sh
find <workdir> -name '*.db2' | wc -l          # confirms one bundle file
grep -c 'lig' <workdir>/run/OUTDOCK           # how many entries were scored
```

More than one scored entry answers both questions at once: the db2 holds multiple forms, and
`OUTDOCK` reports each of them.

## If the answers turn out badly

| Finding | Change |
|---|---|
| `ligbuild` builds only one form | Enable protomer enumeration in `custom_parms.json`, then re-measure the per-molecule cost. |
| `number_write 1` caps the whole run | Raise `number_save`/`number_write` so every entry is written, then re-parse. |
| Neither — all forms built and reported | No change. |

**Cost note.** `ligbuild` already pays for whatever it enumerates today, and the 20 MB dockfiles copy
is made once per molecule regardless. So enabling more enumeration adds `dock64` runtime, not
ligand-build time. We have not measured how the ~32 core-seconds per molecule splits between
`ligbuild` and `dock64`; our expectation is that `ligbuild` dominates (conformer generation plus
AMSOL semi-empirical charges), but that is a guess. It matters because
`config/molecules/s3gfn_minimol_dock3.yaml` sets `fidelity_costs: {1: 32.0}` from measurement, and
the active-learning budget is denominated in those units — any change to the pipeline requires
re-measuring that constant.

**Optional belt-and-braces.** Wrapping the two globs (`.tgz` in `_run_ligbuild`, `.db2` in
`_extract_db2`) in `sorted()` costs nothing and would keep behaviour deterministic if a bundle ever
did contain more than one file. Given the evidence above it is a no-op today, so it has not been
applied.

## Reference: measured behaviour of the pipeline

From the 1000-molecule benchmark (`dock_batch_62723188.csv`):

| | |
|---|---|
| molecules | 1000 |
| failed (sentinel score `0.0`) | 95 (9.5%) |
| seconds per molecule | mean 31.9, median 23.9, max 300.1 |
| scores | min −73.9, mean −34.2, max +0.2 |
| hit the 300 s subprocess timeout | 1 |

Note that a *positive* DOCK3 score occurs (+0.2), which is why our oracle uses `NaN` rather than the
original `0.0` as its failure sentinel — `0.0` is a legitimate, if weak, score.

Also note the 2 molecules that failed in one run and succeeded in the other: some failures are
transient (node conditions, timeouts) rather than properties of the molecule, so a failed docking is
not necessarily permanent.
