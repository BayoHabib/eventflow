# Releasing Shikong

How the maintainer ships a version to PyPI and tags it on GitHub.
**No upload has been performed** — every step below is manual, and the
maintainer decides the release moment. Nothing in this procedure
publishes anything on its own; CI does not upload to PyPI.

## Preconditions

- Working tree clean on `main`, CI green on the merge commit.
- `make audit` PASS 5/5 (ruff, black, mypy, pytest, Chicago replay).
- A PyPI account with access to the `shikong` project, and a PyPI
  API token stored in `~/.pypirc` (or as `TWINE_USERNAME=__token__`
  / `TWINE_PASSWORD=<token>`). Never commit a token, and never paste
  one into chat.
- `build` and `twine` available. If they are missing from the dev
  venv, install them into a scratch venv — never into the shared venv:
  `python -m venv /tmp/ef-build-venv && /tmp/ef-build-venv/bin/python
  -m pip install build twine`.

## Steps

### 1. Decide the version

- Pre-1.0, Alpha: versions are `0.1.0`, `0.2.0`, … for feature
  releases and `0.1.1`, … for fixes. The Development Status classifier
  in `pyproject.toml` stays `3 - Alpha` until the maintainer
  deliberately moves to Beta.
- Bump the version in **two** places, and keep them in sync:
  `pyproject.toml` (`[project] version`) and
  `src/shikong/__init__.py` (`__version__`). `shikong version`
  prints the latter; PyPI shows the former.

### 2. Update the changelog

- In `CHANGELOG.md`, change the `## [Unreleased]` heading (or the
  dated `## [0.1.0]` heading, if the entry was pre-written) to
  `## [<version>] - <YYYY-MM-DD>` with the release date, and add the
  link reference at the bottom of the file.
- Every entry must cite a measured number (`make audit`,
  `docs/audit/traceability.md`) or be marked as unverified. Do not
  copy numbers from memory.
- Add a fresh empty `## [Unreleased]` section at the top for the next
  cycle.

### 3. Check the package metadata

- `pyproject.toml` audit checklist (see also the notes in the file):
  - `name`, `version`, `description`, `readme`, `requires-python`,
    `license`, `authors`, `classifiers`, `dependencies`,
    `project.scripts` (`shikong` console script), `project.urls`.
  - `authors` currently has no email: add a real contact email before
    the first upload. Do not re-add `team@shikong.dev` — it is not
    a real address.
  - Dependencies must be the ones the library imports at runtime.
    A missing runtime dependency only shows up on a clean install,
    so always run step 5 before uploading.
  - `dist/` and `build/` are gitignored; never commit them.

### 4. Build and check

From a clean checkout (or a clean copy of the tree, no `dist/`,
`build/`, `__pycache__`, venv):

```bash
python -m build
twine check dist/*
```

Expected: `Successfully built shikong-<version>.tar.gz and
shikong-<version>-py3-none-any.whl`, and `twine check` PASSED for
both files. If `twine check` fails, fix the metadata and rebuild —
do not upload a distribution that fails the check.

Inspect the sdist at least once per release (`tar tzf
dist/shikong-<version>.tar.gz`): it must contain `LICENSE`,
`README.md`, `pyproject.toml` and `src/shikong/`. The wheel must
contain `shikong-*.dist-info/METADATA` with the right
`Requires-Dist` entries and `licenses/LICENSE`.

### 5. Verify a clean install

In a fresh venv, install the built wheel only — no `-e`, no dev
extras — and run the smoke checks (full procedure and expected
output: `examples/README.md`):

```bash
python -m venv /tmp/ef-clean-venv
/tmp/ef-clean-venv/bin/python -m pip install dist/shikong-<version>-py3-none-any.whl
/tmp/ef-clean-venv/bin/shikong --help
/tmp/ef-clean-venv/bin/shikong version
/tmp/ef-clean-venv/bin/shikong validate --config configs/recipes/chicago_crime_v1.yaml
/tmp/ef-clean-venv/bin/python examples/chicago_quickstart.py
```

All four must pass. If the wheel install exposes a missing runtime
dependency, add it to `pyproject.toml`, rebuild, and re-verify —
do not work around it by installing the dependency by hand.

### 6. Upload to PyPI

```bash
twine upload dist/*
```

This is the only step that publishes anything. Use TestPyPI first
for a first-time release (`twine upload --repository testpypi
dist/*`), then the real index. After the upload, confirm the project
page renders (README, license, classifiers) at
`https://pypi.org/project/shikong/`.

### 7. Tag and release on GitHub

```bash
git tag -a v<version> -m "Release <version>"
git push origin v<version>
```

Then create the GitHub release from the tag
(`https://github.com/BayoHabib/eventflow/releases/new?tag=v<version>`)
and paste the `CHANGELOG.md` entry for this version as the release
notes. The tag name is `v<version>` (for example `v0.1.0`); the
PyPI version itself has no `v` prefix.

## What is deliberately not automated

- No CI job uploads to PyPI (no PyPI credentials in CI).
- No `Dockerfile` for deployment: `examples/Dockerfile` builds an
  examples-only image that runs the smoke example; it is not a
  prediction API and not a deployment service (see
  `examples/README.md`).
- No FastAPI service, model-training utilities or prediction serving:
  those are downstream applications, explicitly out of scope for the
  library release.
