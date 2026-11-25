# Redistribution requirements

pyseekdb is Apache-2.0 (see LICENSE and NOTICE). It contains no bundled
third-party code; every component in `THIRD_PARTY_COMPONENTS.md` is a
separately installed runtime dependency.

## What to retain

- Keep `LICENSE` and `NOTICE` with the pyseekdb distribution.
- Keep `THIRD_PARTY_COMPONENTS.md`, `THIRD_PARTY_COMPONENTS.json` and this
  file so downstream redistributors can identify every direct dependency.
- For each separately installed dependency, retain the license text shipped
  in its own distribution at the path recorded in the inventory, using the
  version actually resolved for your distribution.
- For pylibseekdb, retain the complete `pylibseekdb/licenses/` bundle shipped
  from 1.4.1 in its wheel: it contains the engine's component inventory,
  license texts and redistribution requirements.

## Per-component rules

| Component | Action |
| --- | --- |
| pylibseekdb | Retain the complete pylibseekdb/licenses/ bundle shipped in the pylibseekdb wheel (>=1.4.1), including the engine inventory and license texts. |
| PyMySQL | Retain the license text from the resolved pymysql distribution. |
| MNN | Retain the upstream MNN license text; the resolved wheel does not bundle one. |
| tokenizers | Retain the upstream tokenizers license text; the resolved wheel does not bundle one. |
| httpx | Retain the license text from the resolved httpx distribution. |
| tqdm | Retain the MPL-2.0 and MIT texts from the resolved tqdm distribution (dist-info/LICENCE). |
| tenacity | Retain the license text from the resolved tenacity distribution. |
| numpy | Retain the core license texts. If you redistribute the Linux wheels, also satisfy the vendored library licenses, including the LGPL-2.1 section 6 source conditions for libquadmath. |

## Vendored libraries inside dependencies

The license expression recorded for numpy covers its core without vendored
libraries. The numpy 2.4.2 Linux wheels bundle OpenBLAS
(BSD-3-Clause AND BSD-3-Clause-Attribution), libgfortran
(GPL-3.0-with-GCC-exception) and libquadmath (LGPL-2.1-or-later) under
`numpy.libs/`; these texts are not in `dist-info/licenses/`. A distributor
that redistributes those wheels must also meet the LGPL-2.1 section 6
conditions for libquadmath, including providing its corresponding source or
equivalent access to it. This applies only to the components and platforms a
distributor actually ships.

## Notes

- The inventory covers direct dependencies only. Transitive dependencies come
  with their own license texts and must be identified from the resolved
  environment when a downstream distribution bundles them.
- The locked tqdm wheel stores its license at `dist-info/LICENCE`, not under
  `licenses/`.
- MNN and tokenizers ship no license text in their wheels; take the texts from
  the upstream repositories listed in the inventory.
