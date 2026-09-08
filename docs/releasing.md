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
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build --no-sources
uv run --locked twine check --strict dist/*
```

Start with an empty `dist/` so it contains only the current release's wheel and
source archive. `--no-sources` excludes local dependency overrides. Twine should
report `PASSED` for both files. Inspect the wheel to confirm it contains the
`texture_generators` modules, `py.typed`, distribution metadata and the license.

Install each artifact into a fresh environment outside the checkout and run the
README example. The following POSIX-shell example uses Python 3.14; use the
project's supported interpreter when changing its Python requirement:

```bash
release_version="$(uv version --short)"
release_root="$(pwd)"
smoke_dir="$(mktemp -d)"
unset PYTHONPATH PYTHONHOME
cd "$smoke_dir"

for artifact in \
  "$release_root/dist/texture_generator-$release_version-py3-none-any.whl" \
  "$release_root/dist/texture_generator-$release_version.tar.gz"
do
  uv run --isolated --no-project --python 3.14 --with "$artifact" python - <<'PY'
from importlib.metadata import version

from texture_generators import __version__, generate

assert __version__ == version("texture-generator")
image = generate("wood", size=(640, 480), seed=42, variant="board")
assert image.mode == "RGB"
assert image.size == (640, 480)
image.save("wood.png")
print(image.mode, image.size)
PY
  uv run --isolated --no-project --python 3.14 --with "$artifact" texture-gen --help
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

The checks build both artifacts and exercise them outside the checkout. The
publishing job downloads those same artifacts; only that job can request the
identity credential used by PyPI. GitHub Actions are pinned to reviewed commits.

Watch the [Actions run](https://github.com/nmpowell/texture-generator/actions),
then check the [PyPI project page](https://pypi.org/project/texture-generator/).
Install the exact released version from PyPI in a fresh environment and rerun
the README example. PyPI does not permit reusing an uploaded artifact filename;
correct a bad release with a new version.
