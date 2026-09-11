# Releasing

[Back to the README](../README.md)

Releases use the literal `project.version` in `pyproject.toml`, setuptools
(`setuptools>=77.0.3`) and GitHub Actions Trusted Publishing. The runtime
`texture_generators.__version__` reads installed distribution metadata.

## One-time account setup

Before the first upload, the maintainer must have access to the GitHub repository
and a PyPI account with a verified email address and two-factor authentication.
Check that the PyPI project name is available, or that the account has permission
to publish the existing project. A pending publisher does not reserve a name.

Create the GitHub repository environment **`release`**. In PyPI's
[publishing settings](https://pypi.org/manage/account/publishing/), add a pending
publisher for a new project (or a Trusted Publisher in an existing project's
settings) with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `texture-generator` |
| Owner | `nmpowell` |
| Repository name | `texture-generator` |
| Workflow name | `publish.yml` |
| Environment name | `release` |

The workflow field is the filename, not the workflow's display name or its full
path. The file is
[`.github/workflows/publish.yml`](https://github.com/nmpowell/texture-generator/blob/main/.github/workflows/publish.yml).
Trusted Publishing uses the publishing job's `id-token: write` permission;
no PyPI password or long-lived upload token belongs in repository secrets.

## Prepare and check the artifacts

Update `project.version` in `pyproject.toml`, record the changes in
`CHANGELOG.md`, and refresh `uv.lock` with `uv lock`. From the repository root:

```bash
set -eu

project_environment_root="$(mktemp -d)"
for python_version in 3.12 3.13 3.14
do
  project_environment="$project_environment_root/$python_version"
  UV_PROJECT_ENVIRONMENT="$project_environment" uv sync --locked --python "$python_version"
  "$project_environment/bin/python" -c \
    "import sys; assert sys.version_info[:2] == tuple(map(int, '$python_version'.split('.'))), sys.version"
  uv pip list --python "$project_environment/bin/python"
  uv pip check --python "$project_environment/bin/python"
  "$project_environment/bin/python" -m pytest

  uv pip install --python "$project_environment/bin/python" --reinstall \
    --resolution lowest-direct --only-binary numpy --only-binary pillow \
    -r pyproject.toml
  uv pip list --python "$project_environment/bin/python"
  uv pip check --python "$project_environment/bin/python"
  "$project_environment/bin/python" -m pytest
done

quality_environment="$project_environment_root/3.14"
UV_PROJECT_ENVIRONMENT="$quality_environment" uv run --locked --python 3.14 ruff check .
UV_PROJECT_ENVIRONMENT="$quality_environment" uv run --locked --python 3.14 ruff format --check .
UV_PROJECT_ENVIRONMENT="$project_environment_root/3.12" uv run --locked --python 3.12 mypy
uv build --no-sources --python 3.14
UV_PROJECT_ENVIRONMENT="$quality_environment" uv run --locked --python 3.14 twine check --strict dist/*
```

The distinct temporary `UV_PROJECT_ENVIRONMENT` values keep each interpreter's
locked environment separate without adding unignored environments to the
checkout. Python 3.14 remains the development version pinned in
`.python-version`; the explicit `--python` values make every sync and run
independent of that pin. This loop runs sequentially: packaging tests build in
the checkout, so parallel runs need separate full project copies as well as
separate environments.

After each locked run, `uv pip install -r pyproject.toml` reads the declared
runtime requirements, without selecting the development or examples groups.
`lowest-direct` selects their lowest compatible versions; `--reinstall` prevents
the installed locked versions from being retained. NumPy and Pillow must have
compatible binary wheels. Development tools retain their locked versions, and
`uv.lock` is unchanged. Save the printed package lists with the test results:
wheel availability varies by interpreter, platform and package index.

Run the environment's Python directly after lowering dependencies. A normal
`uv run` would synchronise the environment back to the lockfile. The quality
commands above deliberately restore the locked 3.14 environment. Minimum-wheel
testing does not establish whether older NumPy or Pillow source releases can be
built on a newer interpreter, or whether upstream officially supports that
combination.

Start with an empty `dist/` so it contains only the current release's wheel and
source archive. `--no-sources` excludes local dependency overrides. Twine should
report `PASSED` for both files. Inspect the wheel to confirm it contains the
`texture_generators` modules, `py.typed`, distribution metadata and the license.

Install each artifact into a fresh environment outside the checkout and run the
README example on Python 3.12, 3.13 and 3.14:

```bash
set -eu

release_root="$(pwd)"
smoke_dir="$(mktemp -d)"
unset PYTHONPATH PYTHONHOME
cd "$smoke_dir"

for python_version in 3.12 3.13 3.14
do
  for artifact in "$release_root"/dist/*.whl "$release_root"/dist/*.tar.gz
  do
    artifact_environment="$smoke_dir/$python_version-$(basename "$artifact").venv"
    uv venv --python "$python_version" "$artifact_environment"
    uv pip install --python "$artifact_environment/bin/python" "$artifact"
    "$artifact_environment/bin/python" - <<'PY'
from importlib.metadata import version

from texture_generators import __version__, generate, generate_array

assert __version__ == version("texture-generator")
image = generate("wood", size=(640, 480), seed=42, variant="board")
assert image.mode == "RGB"
assert image.size == (640, 480)
assert generate_array("plastic", size=16, seed=7).shape == (16, 16, 3)
image.save("wood.png")
print(image.mode, image.size)
PY
    "$artifact_environment/bin/texture-gen" wood --size 48x32 --seed 42 -o artifact-smoke.png
    "$artifact_environment/bin/python" -m texture_generators list --json
    "$artifact_environment/bin/texture-gen" samples --only plastic/matte --count 2 --size 16 --no-sweeps --jobs 2 --outdir "$artifact_environment/samples"
    uv pip check --python "$artifact_environment/bin/python"
  done
done

cd "$release_root"
```

Open `wood.png` in the temporary directory to check the rendered result.
The source archive check also confirms a user can build and install the package
without the checkout. These smoke checks supplement the full test suite.

## Publish the checked version

Commit and push the version, changelog and lockfile changes before creating the
release. Create a GitHub Release targeting that exact commit, with a tag matching
`project.version` (for example, `0.3.0` for `version = "0.3.0"`), write release
notes, then publish the release. The workflow checks tag/version agreement and
must pass its checks before uploading. Pushing a bare tag does not publish.

The reusable test workflow runs the full suite with both locked and minimum
runtime dependencies on Python 3.12, 3.13 and 3.14, with fail-fast disabled and
an assertion of each requested interpreter. It records installed versions and
checks dependency consistency before testing. Ruff runs once on the 3.14
development version, and mypy runs once on 3.12, the oldest supported version,
in its own `typecheck` job that fails the run on any type error. A separate
job builds
the wheel and source archive once on 3.14 and uploads one `dist` artifact. The
artifact smoke jobs download that same build, install both distributions
separately on all three Python versions, and check their dependencies. The
publishing job waits for the complete reusable workflow and downloads the
validated `dist` artifact; only that job can request the identity credential
used by PyPI. GitHub Actions are pinned to reviewed commits.

Watch the [Actions run](https://github.com/nmpowell/texture-generator/actions),
then check the [PyPI project page](https://pypi.org/project/texture-generator/).
Install the exact released version from PyPI in a fresh environment and rerun
the README example. PyPI does not permit reusing an uploaded artifact filename;
correct a bad release with a new version.
