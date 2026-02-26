"""
BM25 sparse embedding function.

A pure-Python BM25 sparse embedding function that tokenizes text, applies
stopword filtering and Snowball stemming, hashes tokens with MurmurHash3,
and computes BM25 term-frequency scores as sparse vector weights.

No external model download is needed — only one required dependency:
  - ``snowballstemmer`` for stemming

The token hasher uses a pure-Python MurmurHash3 (x86_32) implementation, so no
C extension is required — results are always consistent across platforms.

Install:
    pip install snowballstemmer

Example:
    >>> from pyseekdb.utils.embedding_functions import BM25SparseEmbeddingFunction
    >>> ef = BM25SparseEmbeddingFunction()
    >>> sparse_vectors = ef(["machine learning algorithms", "python tutorial"])
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from collections.abc import Iterable
from typing import Any, ClassVar, Protocol, cast

from pyseekdb.client.sparse_embedding_function import (
    Documents,
    SparseEmbeddingFunction,
    SparseVector,
    SparseVectors,
    register_sparse_embedding_function,
)

# ── BM25 defaults ────────────────────────────────────────────────────

DEFAULT_K = 1.2
DEFAULT_B = 0.75
DEFAULT_AVG_DOC_LENGTH = 256.0
DEFAULT_TOKEN_MAX_LENGTH = 40

# fmt: off
DEFAULT_ENGLISH_STOPWORDS: list[str] = [
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have",
    "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't",
    "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",
    "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn",
    "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've",
    "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that",
    "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
]
# fmt: on

# ── Stemmer ──────────────────────────────────────────────────────────


class _SnowballStemmer(Protocol):
    def stem(self, token: str) -> str: ...


class _SnowballStemmerAdapter:
    """Adapter wrapping ``snowballstemmer`` to provide a uniform ``stem`` API."""

    def __init__(self, language: str = "english") -> None:
        try:
            import snowballstemmer
        except ImportError as exc:
            raise ValueError(
                "The snowballstemmer package is not installed. Please install it with `pip install snowballstemmer`"
            ) from exc
        self._stemmer = snowballstemmer.stemmer(language)

    def stem(self, token: str) -> str:
        return cast(str, self._stemmer.stemWord(token))


def _get_english_stemmer() -> _SnowballStemmer:
    return _SnowballStemmerAdapter("english")


# ── Hasher (pure-Python MurmurHash3 x86_32) ─────────────────────────

_U32 = 0xFFFFFFFF


def _murmurhash3_x86_32(data: bytes, seed: int = 0) -> int:
    """Pure-Python MurmurHash3 x86_32, matching the C reference implementation."""
    length = len(data)
    h1 = seed & _U32
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    nblocks = length // 4

    for i in range(nblocks):
        k1 = struct.unpack_from("<I", data, i * 4)[0]
        k1 = (k1 * c1) & _U32
        k1 = ((k1 << 15) | (k1 >> 17)) & _U32
        k1 = (k1 * c2) & _U32
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & _U32
        h1 = (h1 * 5 + 0xE6546B64) & _U32

    tail_index = nblocks * 4
    tail = data[tail_index:]
    k1 = 0
    tail_len = length & 3
    if tail_len >= 3:
        k1 ^= tail[2] << 16
    if tail_len >= 2:
        k1 ^= tail[1] << 8
    if tail_len >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & _U32
        k1 = ((k1 << 15) | (k1 >> 17)) & _U32
        k1 = (k1 * c2) & _U32
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & _U32
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & _U32
    h1 ^= h1 >> 16

    return h1


class _Murmur3AbsHasher:
    """
    MurmurHash3 x86_32 hasher returning non-negative signed-32-bit values.

    Uses a pure-Python implementation so the result is identical on every
    platform regardless of whether a C extension (mmh3) is installed.
    The output matches ``abs(mmh3.hash(token, seed=seed))``.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def hash(self, token: str) -> int:
        unsigned = _murmurhash3_x86_32(token.encode("utf-8"), self._seed)
        signed = struct.unpack("<i", struct.pack("<I", unsigned))[0]
        return abs(signed)


# ── Tokenizer ────────────────────────────────────────────────────────

_NON_ALNUM_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


class _BM25Tokenizer:
    """Tokenizer with stopword filtering and stemming for BM25."""

    def __init__(
        self,
        stemmer: _SnowballStemmer,
        stopwords: Iterable[str],
        token_max_length: int,
    ) -> None:
        self._stemmer = stemmer
        self._stopwords = {w.lower() for w in stopwords}
        self._max_len = token_max_length

    def tokenize(self, text: str) -> list[str]:
        cleaned = _NON_ALNUM_RE.sub(" ", text)
        raw_tokens = [t for t in cleaned.lower().split() if t]
        tokens: list[str] = []
        for tok in raw_tokens:
            if tok in self._stopwords:
                continue
            if len(tok) > self._max_len:
                continue
            stemmed = self._stemmer.stem(tok).strip()
            if stemmed:
                tokens.append(stemmed)
        return tokens


# ── Hashed token (for Counter dedup) ────────────────────────────────


class _HashedToken:
    __slots__ = ("hash", "label")

    def __init__(self, hash_val: int, label: str | None):
        self.hash = hash_val
        self.label = label

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HashedToken):
            return NotImplemented
        return self.hash == other.hash

    def __lt__(self, other: _HashedToken) -> bool:
        return self.hash < other.hash


# ── BM25 Sparse Embedding Function ──────────────────────────────────


@register_sparse_embedding_function
class BM25SparseEmbeddingFunction(SparseEmbeddingFunction):
    """
    BM25 sparse embedding function.

    Tokenizes text (lowercase, remove punctuation, filter stopwords, stem),
    hashes each stemmed token to a dimension index via MurmurHash3, and computes
    a BM25-style term frequency weight:

        score = tf * (k + 1) / (tf + k * (1 - b + b * doc_len / avg_doc_length))

    This is the *query-independent* part of BM25 (no IDF), suitable for building
    a sparse vector index. The inverse document frequency component can be handled
    at search time by the database engine.

    Args:
        k: BM25 k1 parameter controlling term-frequency saturation. Default 1.2.
        b: BM25 b parameter controlling document-length normalization. Default 0.75.
        avg_doc_length: Assumed average document length in tokens. Default 256.0.
        token_max_length: Maximum token length; longer tokens are dropped. Default 40.
        stopwords: Custom stopword list. ``None`` uses built-in English stopwords.

    Example:
        >>> ef = BM25SparseEmbeddingFunction(k=1.5, b=0.8)
        >>> vectors = ef(["machine learning algorithms"])
        >>> print(vectors[0])
        SparseVector(3 non-zero entries)
    """

    _stemmer_cache: ClassVar[_SnowballStemmer | None] = None

    def __init__(
        self,
        k: float = DEFAULT_K,
        b: float = DEFAULT_B,
        avg_doc_length: float = DEFAULT_AVG_DOC_LENGTH,
        token_max_length: int = DEFAULT_TOKEN_MAX_LENGTH,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.k = float(k)
        self.b = float(b)
        self.avg_doc_length = float(avg_doc_length)
        self.token_max_length = int(token_max_length)

        if stopwords is not None:
            self.stopwords: list[str] | None = [str(w) for w in stopwords]
            self._stopword_list: Iterable[str] = self.stopwords
        else:
            self.stopwords = None
            self._stopword_list = DEFAULT_ENGLISH_STOPWORDS

        self._hasher = _Murmur3AbsHasher()

    def _get_stemmer(self) -> _SnowballStemmer:
        if BM25SparseEmbeddingFunction._stemmer_cache is None:
            BM25SparseEmbeddingFunction._stemmer_cache = _get_english_stemmer()
        return BM25SparseEmbeddingFunction._stemmer_cache

    def _encode(self, text: str) -> SparseVector:
        tokenizer = _BM25Tokenizer(self._get_stemmer(), self._stopword_list, self.token_max_length)
        tokens = tokenizer.tokenize(text)

        if not tokens:
            return SparseVector.from_dict({0: 1e-6})

        doc_len = float(len(tokens))
        counts = Counter(_HashedToken(self._hasher.hash(tok), None) for tok in tokens)

        sorted_keys = sorted(counts.keys())
        indices: list[int] = []
        values: list[float] = []

        for key in sorted_keys:
            tf = float(counts[key])
            denominator = tf + self.k * (1 - self.b + (self.b * doc_len) / self.avg_doc_length)
            score = tf * (self.k + 1) / denominator
            indices.append(key.hash)
            values.append(score)

        return SparseVector.from_indices(indices, values)

    def __call__(self, documents: Documents) -> SparseVectors:
        if isinstance(documents, str):
            documents = [documents]
        return [self._encode(doc) for doc in documents]

    def embed_query(self, documents: Documents) -> SparseVectors:
        """Alias — BM25 uses the same encoding for documents and queries."""
        return self(documents)

    # ── Persistence ──────────────────────────────────────────────────

    @staticmethod
    def name() -> str:
        return "bm25"

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "k": self.k,
            "b": self.b,
            "avg_doc_length": self.avg_doc_length,
            "token_max_length": self.token_max_length,
        }
        if self.stopwords is not None:
            config["stopwords"] = list(self.stopwords)
        return config

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> BM25SparseEmbeddingFunction:
        return BM25SparseEmbeddingFunction(
            k=config.get("k", DEFAULT_K),
            b=config.get("b", DEFAULT_B),
            avg_doc_length=config.get("avg_doc_length", DEFAULT_AVG_DOC_LENGTH),
            token_max_length=config.get("token_max_length", DEFAULT_TOKEN_MAX_LENGTH),
            stopwords=config.get("stopwords"),
        )
