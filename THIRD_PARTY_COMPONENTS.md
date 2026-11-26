# Third-party components

pyseekdb does not bundle third-party code. The table lists the **direct**
runtime dependencies declared in `pyproject.toml`; transitive dependencies
are resolved at install time and ship their own license texts. The locked
column records the version resolved by `uv.lock`; downstream distributions
must use the license texts of the versions they actually resolve.

| Component | Constraint | Locked | Upstream source | License | License text |
| --- | --- | --- | --- | --- | --- |
| pylibseekdb | `pylibseekdb (linux, macOS arm64)` | 1.3.0 | https://github.com/oceanbase/seekdb-bindings | Apache-2.0 | `none in 1.3.0; pylibseekdb/licenses/ and pylibseekdb-<version>.dist-info/licenses/ in >=1.4.1` |
| PyMySQL | `pymysql>=1.1.1` | 1.1.2 | https://github.com/PyMySQL/PyMySQL | MIT | `pymysql-<version>.dist-info/licenses/LICENSE` |
| MNN | `MNN>=3.6.1` | 3.6.1 | https://github.com/alibaba/MNN | BSD-3-Clause (declared in the PyPI metadata) | `no license file in the 3.6.1 wheel; upstream https://github.com/alibaba/MNN/blob/master/LICENSE` |
| tokenizers | `tokenizers>=0.15.0` | 0.22.2 | https://github.com/huggingface/tokenizers | Apache-2.0 | `no license file in the 0.22.2 wheel; upstream https://github.com/huggingface/tokenizers/blob/main/LICENSE` |
| httpx | `httpx` | 0.28.1 | https://github.com/encode/httpx | BSD-3-Clause | `httpx-<version>.dist-info/licenses/LICENSE.md` |
| tqdm | `tqdm` | 4.67.1 | https://github.com/tqdm/tqdm | MPL-2.0 AND MIT | `tqdm-<version>.dist-info/LICENCE` |
| tenacity | `tenacity` | 9.1.2 | https://github.com/jd/tenacity | Apache-2.0 | `tenacity-<version>.dist-info/licenses/LICENSE` |
| numpy | `numpy>=1.26,!=2.4.0` | 2.4.2 | https://github.com/numpy/numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (core only) | `numpy-<version>.dist-info/licenses/ (core)` |

Notes:

- **pylibseekdb**: The 1.3.0 wheel ships no license text. The complete engine and bindings bundle ships under pylibseekdb/licenses/ from 1.4.1 (oceanbase/seekdb-bindings#95).
- **MNN**: The resolved wheel bundles no license text; take it from the upstream repository.
- **tokenizers**: The resolved wheel bundles no license text; take it from the upstream repository.
- **tqdm**: The locked wheel stores the license at LICENCE, not under licenses/.
- **numpy**: The expression covers the numpy core without vendored libraries. The 2.4.2 Linux wheels bundle OpenBLAS (BSD-3-Clause AND BSD-3-Clause-Attribution), libgfortran (GPL-3.0-with-GCC-exception) and libquadmath (LGPL-2.1-or-later) under numpy.libs/; their texts are not in dist-info/licenses/.

See `REDISTRIBUTION.md` for the per-component retention rules.
