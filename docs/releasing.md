# Releasing UnitRefine on PyPI

UnitRefine is published on PyPI by the GitHub workflow `.github/workflows/publish.yml`, with PyPI's
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts this repository's workflow, so no
password or token is stored anywhere.

## Once: set up trusted publishing

Do this with the PyPI account which will own the project. Do it on both TestPyPI (for trying a release) and PyPI.

1. Create an account on [TestPyPI](https://test.pypi.org/account/register/) and on [PyPI](https://pypi.org/account/register/)
   (they are separate), and turn on two-factor authentication, which PyPI requires.
2. On each, go to *Your account > Publishing > Add a new pending publisher > GitHub* and fill in:

   | Field             | TestPyPI         | PyPI             |
   |-------------------|------------------|------------------|
   | PyPI Project Name | `unitrefine`     | `unitrefine`     |
   | Owner             | `anoushkajain`   | `anoushkajain`   |
   | Repository name   | `UnitRefine`     | `UnitRefine`     |
   | Workflow name     | `publish.yml`    | `publish.yml`    |
   | Environment name  | `testpypi`       | `pypi`           |

   A *pending* publisher reserves the name until the first upload creates the project.
3. On GitHub, in the repository's *Settings > Environments*, create the environments `testpypi` and `pypi`.
   For `pypi`, you can add yourself as a *required reviewer*, so every release to PyPI waits for your approval.

## Each release

1. Set the new version in `pyproject.toml` (e.g. `version = "0.2.1"`), then run `uv lock`, which records it in
   `uv.lock`. Merge this into `main`.
2. **Try it on TestPyPI:** on GitHub, *Actions > Publish to PyPI > Run workflow* (on `main`). It builds the
   packages, checks them, installs them, and uploads them to TestPyPI. To check the result in a new environment:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "unitrefine[gui]==0.2.1"
   ```

   TestPyPI accepts each version only once: to try again, raise the version (e.g. `0.2.1rc1`, then `0.2.1rc2`).
3. **Publish on PyPI:** on GitHub, *Releases > Draft a new release*, create the tag `v0.2.1` (the version with a
   `v`), write what changed, and *Publish release*. The workflow checks that the tag is the version in
   `pyproject.toml`, and publishes. A version can't be changed on PyPI once uploaded, only a new one published.
4. Check [pypi.org/project/unitrefine](https://pypi.org/project/unitrefine/), and that `pip install "unitrefine[gui]"`
   installs it.

## What is published

- The wheel and the source package contain `src/unitrefine` (with the images and fonts in `resources/`). The source
  package also has the tests (see `MANIFEST.in`). The notebooks, datasets and tutorials in `UnitRefine/` are not
  published: they stay on GitHub.
- `pip install unitrefine` installs what applying models needs (e.g. on a computing cluster, see `docs/hpc.md`);
  `pip install "unitrefine[gui]"` adds the GUI.
- PyPI shows `README.md` as the project page, so links and images in it must be full URLs
  (`https://github.com/...` or `https://raw.githubusercontent.com/...`): relative ones don't work there.
