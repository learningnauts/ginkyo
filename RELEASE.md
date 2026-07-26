# Release checklist (v0.1+)　

Human steps to publish source + Windows/macOS binaries.

## Before tagging

1. Working tree clean; `pytest` green
2. `pyproject.toml` / `ginkyo.__version__` match the tag (e.g. `0.1.0` ↔ `v0.1.0`)
3. `CHANGELOG.md` has a section for this version
4. README Status / Download / Known limitations look current
5. Remotes: `dev` → `learningnauts/ginkyo-dev` (development), `public` → `learningnauts/ginkyo` (users / Releases)

## Tag and push

```bash
git push dev main            # keep development remote in sync
git push public main         # publish the release commit
git tag -a v0.1.0 -m "ginkyo 0.1.0"
git push public v0.1.0       # this tag triggers the build
git push dev v0.1.0
```

Pushing a `v*` tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml):
builds Windows + macOS PyInstaller bundles and attaches zip files to a GitHub Release.

## After the workflow finishes

1. Open the Release on GitHub; confirm both zips are attached
2. Spot-check download + launch on Windows and macOS (Gatekeeper / SmartScreen warnings are expected while unsigned)
3. Optional: announce via note / social with the Release URL

## Local binary (without Actions)

```bash
pip install -e . pyinstaller
pyinstaller packaging/ginkyo.spec --noconfirm
# output under dist/ginkyo/
```

## Notes

- Public repos: Actions minutes are generally free enough for release builds
- Code signing (Apple / Windows) is optional and can wait until later
