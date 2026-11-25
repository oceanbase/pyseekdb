# Redistribution requirements

pyseekdb is Apache-2.0 (see LICENSE and NOTICE). It contains no bundled
third-party code; every component in `THIRD_PARTY_COMPONENTS.md` is a
separately installed runtime dependency.

## What to retain

- Keep `LICENSE` and `NOTICE` with the pyseekdb distribution.
- Keep `THIRD_PARTY_COMPONENTS.md`, `THIRD_PARTY_COMPONENTS.json` and this
  file so downstream redistributors can identify every dependency.
- For each separately installed dependency, retain the license text shipped
  in its own distribution (`<package>-<version>.dist-info/licenses/`), using
  the version actually resolved for your distribution.
- For pylibseekdb, retain the complete `pylibseekdb/licenses/` bundle shipped
  in its wheel: it contains the engine's component inventory, license texts
  and redistribution requirements.

## Per-component rules

| Component | Action |
| --- | --- |
| pylibseekdb | Retain the complete pylibseekdb/licenses/ bundle shipped in the pylibseekdb wheel, including the engine inventory and license texts. |
| PyMySQL | Retain the license text from the resolved pymysql distribution. |
| MNN | Retain the license text from the resolved MNN distribution. |
| tokenizers | Retain the license text from the resolved tokenizers distribution. |
| httpx | Retain the license text from the resolved httpx distribution. |
| tqdm | Retain the MPL-2.0 and MIT texts from the resolved tqdm distribution. |
| tenacity | Retain the license text from the resolved tenacity distribution. |
| numpy | Retain the license texts from the resolved numpy distribution. |

## Notes

- No component requires corresponding source or relink materials beyond the
  notices and license texts supplied by its own distribution.
- MNN's license is declared as BSD-3-Clause in the PyPI metadata; verify the
  resolved distribution when redistributing it.
- numpy is distributed under a composite license; retain all license texts
  from the resolved distribution.
