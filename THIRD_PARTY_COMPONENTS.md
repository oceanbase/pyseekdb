# Third-party components

pyseekdb does not bundle third-party code. The components below are
separately installed runtime dependencies. The locked column records the
version resolved by `uv.lock` at the time of this release; downstream
distributions must use the license texts of the versions they actually
resolve.

| Component | Constraint | Locked | Upstream source | License | License text |
| --- | --- | --- | --- | --- | --- |
| pylibseekdb | `pylibseekdb (linux, macOS arm64)` | 1.3.0 | https://github.com/oceanbase/seekdb-bindings | Apache-2.0 | `pylibseekdb/licenses/ and pylibseekdb-<version>.dist-info/licenses/` |
| PyMySQL | `pymysql>=1.1.1` | 1.1.2 | https://github.com/PyMySQL/PyMySQL | MIT | `pymysql-<version>.dist-info/licenses/` |
| MNN | `MNN>=3.6.1` | 3.6.1 | https://github.com/alibaba/MNN | BSD-3-Clause (declared in the PyPI metadata) | `mnn-<version>.dist-info/licenses/` |
| tokenizers | `tokenizers>=0.15.0` | 0.22.2 | https://github.com/huggingface/tokenizers | Apache-2.0 | `tokenizers-<version>.dist-info/licenses/` |
| httpx | `httpx` | 0.28.1 | https://github.com/encode/httpx | BSD-3-Clause | `httpx-<version>.dist-info/licenses/` |
| tqdm | `tqdm` | 4.67.1 | https://github.com/tqdm/tqdm | MPL-2.0 AND MIT | `tqdm-<version>.dist-info/licenses/` |
| tenacity | `tenacity` | 9.1.2 | https://github.com/jd/tenacity | Apache-2.0 | `tenacity-<version>.dist-info/licenses/` |
| numpy | `numpy>=1.26,!=2.4.0` | 2.4.2 | https://github.com/numpy/numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `numpy-<version>.dist-info/licenses/` |

pylibseekdb additionally ships the complete engine license bundle under
`pylibseekdb/licenses/` in its wheel. See `REDISTRIBUTION.md` for the
per-component retention rules.
