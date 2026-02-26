"""
Unit tests for BM25SparseEmbeddingFunction.

Tests tokenization, BM25 scoring, persistence (get_config / build_from_config),
protocol compliance, and registry integration.

Requires: snowballstemmer

To run:
    pytest tests/unit_tests/test_bm25_sparse_embedding_function.py -v
"""

import importlib.util
import struct

import pytest

from pyseekdb.client.sparse_embedding_function import (
    SparseEmbeddingFunction,
    SparseEmbeddingFunctionRegistry,
    SparseVector,
)


def _deps_available() -> bool:
    return importlib.util.find_spec("snowballstemmer") is not None


_skip_no_deps = pytest.mark.skipif(
    not _deps_available(),
    reason="snowballstemmer not installed",
)


@pytest.fixture(autouse=True)
def _clear_stemmer_cache():
    """Reset the class-level stemmer cache between tests."""
    from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
        BM25SparseEmbeddingFunction,
    )

    saved = BM25SparseEmbeddingFunction._stemmer_cache
    BM25SparseEmbeddingFunction._stemmer_cache = None
    yield
    BM25SparseEmbeddingFunction._stemmer_cache = saved


# ────────────────────────────────────────────────────────────────────
# MurmurHash3 correctness (reference test vectors from SMHasher)
# ────────────────────────────────────────────────────────────────────


class TestMurmurHash3Correctness:
    """
    Verify the pure-Python MurmurHash3 x86_32 against known reference vectors
    from the original C implementation by Austin Appleby (SMHasher test suite).
    """

    @pytest.mark.parametrize(
        ("data", "seed", "expected_unsigned"),
        [
            # --- empty key ---
            (b"", 0, 0x00000000),
            (b"", 1, 0x514E28B7),
            (b"", 0xFFFFFFFF, 0x81F16F39),
            # --- 4-byte keys (exactly one block, no tail) ---
            (b"\xff\xff\xff\xff", 0, 0x76293B50),
            (b"\x21\x43\x65\x87", 0, 0xF55B516B),
            (b"\x21\x43\x65\x87", 0x5082EDEE, 0x2362F9DE),
            # --- tail handling: 3, 2, 1 byte(s) ---
            (b"\x21\x43\x65", 0, 0x7E4A8634),
            (b"\x21\x43", 0, 0xA0F7B07A),
            (b"\x21", 0, 0x72661CF4),
        ],
        ids=[
            "empty_seed0",
            "empty_seed1",
            "empty_seedMax",
            "4byte_ff",
            "4byte_2143",
            "4byte_2143_seedCustom",
            "3byte_tail",
            "2byte_tail",
            "1byte_tail",
        ],
    )
    def test_reference_vectors(self, data: bytes, seed: int, expected_unsigned: int):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            _murmurhash3_x86_32,
        )

        result = _murmurhash3_x86_32(data, seed)
        assert result == expected_unsigned, (
            f"murmurhash3_x86_32({data!r}, seed={seed:#x}) = {result:#010x}, expected {expected_unsigned:#010x}"
        )

    @pytest.mark.parametrize(
        ("text", "seed", "expected_unsigned"),
        [
            ("", 0, 0x00000000),
            ("Hello", 0, 0x12DA77C8),
            ("hello", 0, 0x248BFA47),
            ("Hello, world!", 0, 0xC0363E43),
            ("test", 0, 0xBA6BD213),
            ("abc", 0, 0xB3DD93FA),
            ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 0, 0xEE925B90),
        ],
        ids=["str_empty", "str_Hello", "str_hello", "str_HelloWorld", "str_test", "str_abc", "str_long"],
    )
    def test_string_vectors(self, text: str, seed: int, expected_unsigned: int):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            _murmurhash3_x86_32,
        )

        result = _murmurhash3_x86_32(text.encode("utf-8"), seed)
        assert result == expected_unsigned, (
            f'murmurhash3_x86_32("{text}", seed={seed}) = {result:#010x}, expected {expected_unsigned:#010x}'
        )

    def test_abs_hasher_matches_signed_interpretation(self):
        """
        _Murmur3AbsHasher.hash(token) should equal abs(signed_32bit(murmurhash3(token))).
        This is the same as abs(mmh3.hash(token, seed=0)).
        """
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            _Murmur3AbsHasher,
            _murmurhash3_x86_32,
        )

        hasher = _Murmur3AbsHasher(seed=0)
        for token in ["hello", "world", "test", "python", "机器学习", ""]:
            unsigned = _murmurhash3_x86_32(token.encode("utf-8"), 0)
            signed = struct.unpack("<i", struct.pack("<I", unsigned))[0]
            expected = abs(signed)
            assert hasher.hash(token) == expected, f"Mismatch for token={token!r}"

    def test_deterministic_across_calls(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            _murmurhash3_x86_32,
        )

        for _ in range(100):
            assert _murmurhash3_x86_32(b"deterministic", 42) == _murmurhash3_x86_32(b"deterministic", 42)


# ────────────────────────────────────────────────────────────────────
# Initialization
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25Init:
    def test_default_params(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        assert ef.k == pytest.approx(1.2)
        assert ef.b == pytest.approx(0.75)
        assert ef.avg_doc_length == pytest.approx(256.0)
        assert ef.token_max_length == 40
        assert ef.stopwords is None

    def test_custom_params(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction(
            k=2.0, b=0.5, avg_doc_length=128.0, token_max_length=20, stopwords=["the", "a"]
        )
        assert ef.k == pytest.approx(2.0)
        assert ef.b == pytest.approx(0.5)
        assert ef.avg_doc_length == pytest.approx(128.0)
        assert ef.token_max_length == 20
        assert ef.stopwords == ["the", "a"]

    def test_stopwords_converted_to_strings(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction(stopwords=[1, 2, 3])
        assert ef.stopwords == ["1", "2", "3"]


# ────────────────────────────────────────────────────────────────────
# Encoding / __call__
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25Encode:
    def test_basic_encoding(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        result = ef(["machine learning algorithms"])

        assert len(result) == 1
        sv = result[0]
        assert isinstance(sv, SparseVector)
        assert sv.embeddings is not None
        assert len(sv.embeddings) > 0
        for idx, val in sv.embeddings.items():
            assert isinstance(idx, int)
            assert isinstance(val, float)
            assert idx >= 0
            assert val > 0

    def test_single_string_input(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        result = ef("hello world")

        assert len(result) == 1
        assert isinstance(result[0], SparseVector)
        assert result[0].embeddings is not None

    def test_multiple_documents(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        docs = ["first document", "second document", "third one"]
        result = ef(docs)

        assert len(result) == 3
        for sv in result:
            assert isinstance(sv, SparseVector)
            assert sv.embeddings is not None

    def test_empty_list(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        result = ef([])
        assert result == []

    def test_stopword_only_text(self):
        """Text that is all stopwords should produce a minimal placeholder vector."""
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        result = ef(["the and or but"])

        assert len(result) == 1
        sv = result[0]
        assert sv.embeddings is not None
        assert 0 in sv.embeddings
        assert sv.embeddings[0] == pytest.approx(1e-6)

    def test_repeated_token_higher_weight(self):
        """Repeating a token should increase its BM25 weight."""
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        r_single = ef(["python"])
        r_repeated = ef(["python python python"])

        sv_single = r_single[0]
        sv_repeated = r_repeated[0]

        assert len(sv_single.embeddings) == 1
        assert len(sv_repeated.embeddings) == 1

        key = next(iter(sv_single.embeddings.keys()))
        assert key in sv_repeated.embeddings
        assert sv_repeated.embeddings[key] > sv_single.embeddings[key]

    def test_deterministic(self):
        """Same input should produce identical output."""
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        r1 = ef(["hello world foo bar baz"])
        r2 = ef(["hello world foo bar baz"])

        assert r1[0].embeddings == r2[0].embeddings

    def test_punctuation_stripped(self):
        """Punctuation should be stripped so 'hello!' == 'hello'."""
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        r1 = ef(["hello"])
        r2 = ef(["hello!!!"])

        assert r1[0].embeddings == r2[0].embeddings

    def test_case_insensitive(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        r1 = ef(["Python"])
        r2 = ef(["python"])

        assert r1[0].embeddings == r2[0].embeddings

    def test_long_token_filtered(self):
        """Tokens exceeding token_max_length should be dropped."""
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction(token_max_length=5)
        result = ef(["hi longtokenhere"])

        sv = result[0]
        assert sv.embeddings is not None
        assert len(sv.embeddings) == 1

    def test_custom_stopwords(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef_default = BM25SparseEmbeddingFunction()
        ef_custom = BM25SparseEmbeddingFunction(stopwords=["python", "java"])

        r_default = ef_default(["python tutorial"])
        r_custom = ef_custom(["python tutorial"])

        assert len(r_default[0].embeddings) == 2
        assert len(r_custom[0].embeddings) == 1


# ────────────────────────────────────────────────────────────────────
# BM25 Score Correctness
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25ScoreCorrectness:
    """Verify the BM25 TF scoring formula."""

    def test_single_token_score(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        k, b, avg_dl = 1.2, 0.75, 256.0
        ef = BM25SparseEmbeddingFunction(k=k, b=b, avg_doc_length=avg_dl)

        result = ef(["python"])
        sv = result[0]
        val = next(iter(sv.embeddings.values()))

        tf = 1.0
        doc_len = 1.0
        expected = tf * (k + 1) / (tf + k * (1 - b + b * doc_len / avg_dl))
        assert val == pytest.approx(expected, rel=1e-6)

    def test_multi_token_scores(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        k, b, avg_dl = 1.5, 0.8, 100.0
        ef = BM25SparseEmbeddingFunction(k=k, b=b, avg_doc_length=avg_dl)

        result = ef(["alpha alpha beta"])
        sv = result[0]

        doc_len = 3.0

        alpha_hash = None
        beta_hash = None
        result_alpha_only = ef(["alpha"])
        result_beta_only = ef(["beta"])
        alpha_hash = next(iter(result_alpha_only[0].embeddings.keys()))
        beta_hash = next(iter(result_beta_only[0].embeddings.keys()))

        tf_alpha = 2.0
        expected_alpha = tf_alpha * (k + 1) / (tf_alpha + k * (1 - b + b * doc_len / avg_dl))
        assert sv.embeddings[alpha_hash] == pytest.approx(expected_alpha, rel=1e-6)

        tf_beta = 1.0
        expected_beta = tf_beta * (k + 1) / (tf_beta + k * (1 - b + b * doc_len / avg_dl))
        assert sv.embeddings[beta_hash] == pytest.approx(expected_beta, rel=1e-6)


# ────────────────────────────────────────────────────────────────────
# embed_query
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25EmbedQuery:
    def test_embed_query_same_as_call(self):
        """BM25 uses the same encoding for docs and queries."""
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        r_call = ef(["search query terms"])
        r_embed = ef.embed_query(["search query terms"])

        assert r_call[0].embeddings == r_embed[0].embeddings


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25Persistence:
    def test_name(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        assert BM25SparseEmbeddingFunction.name() == "bm25"

    def test_get_config_defaults(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        config = ef.get_config()

        assert config["k"] == pytest.approx(1.2)
        assert config["b"] == pytest.approx(0.75)
        assert config["avg_doc_length"] == pytest.approx(256.0)
        assert config["token_max_length"] == 40
        assert "stopwords" not in config
        assert "name" not in config

    def test_get_config_with_custom_stopwords(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction(stopwords=["the", "a", "is"])
        config = ef.get_config()

        assert config["stopwords"] == ["the", "a", "is"]

    def test_get_config_custom_params(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction(k=2.0, b=0.5, avg_doc_length=128.0, token_max_length=20)
        config = ef.get_config()

        assert config["k"] == pytest.approx(2.0)
        assert config["b"] == pytest.approx(0.5)
        assert config["avg_doc_length"] == pytest.approx(128.0)
        assert config["token_max_length"] == 20

    def test_build_from_config_defaults(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        config = {}
        ef = BM25SparseEmbeddingFunction.build_from_config(config)

        assert isinstance(ef, BM25SparseEmbeddingFunction)
        assert ef.k == pytest.approx(1.2)
        assert ef.b == pytest.approx(0.75)
        assert ef.avg_doc_length == pytest.approx(256.0)
        assert ef.token_max_length == 40
        assert ef.stopwords is None

    def test_build_from_config_custom(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        config = {
            "k": 2.0,
            "b": 0.5,
            "avg_doc_length": 128.0,
            "token_max_length": 20,
            "stopwords": ["x", "y"],
        }
        ef = BM25SparseEmbeddingFunction.build_from_config(config)

        assert ef.k == pytest.approx(2.0)
        assert ef.b == pytest.approx(0.5)
        assert ef.avg_doc_length == pytest.approx(128.0)
        assert ef.token_max_length == 20
        assert ef.stopwords == ["x", "y"]

    def test_roundtrip(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        original = BM25SparseEmbeddingFunction(
            k=1.5, b=0.6, avg_doc_length=200.0, token_max_length=30, stopwords=["foo", "bar"]
        )
        config = original.get_config()
        restored = BM25SparseEmbeddingFunction.build_from_config(config)

        assert restored.k == pytest.approx(original.k)
        assert restored.b == pytest.approx(original.b)
        assert restored.avg_doc_length == pytest.approx(original.avg_doc_length)
        assert restored.token_max_length == original.token_max_length
        assert restored.stopwords == original.stopwords

    def test_roundtrip_produces_same_vectors(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        original = BM25SparseEmbeddingFunction(k=1.5, b=0.6)
        config = original.get_config()
        restored = BM25SparseEmbeddingFunction.build_from_config(config)

        text = ["machine learning algorithms and data science"]
        r_orig = original(text)
        r_restored = restored(text)

        assert r_orig[0].embeddings == r_restored[0].embeddings


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25Registry:
    def test_registered_in_registry(self):
        assert SparseEmbeddingFunctionRegistry.get_class("bm25") is not None

    def test_listed_in_registry(self):
        names = SparseEmbeddingFunctionRegistry.list_registered()
        assert "bm25" in names

    def test_registry_build_from_config(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        config = {"k": 1.0, "b": 0.5}
        ef = SparseEmbeddingFunctionRegistry.build_from_config("bm25", config)

        assert isinstance(ef, BM25SparseEmbeddingFunction)
        assert ef.k == pytest.approx(1.0)
        assert ef.b == pytest.approx(0.5)


# ────────────────────────────────────────────────────────────────────
# Protocol Compliance
# ────────────────────────────────────────────────────────────────────


@_skip_no_deps
class TestBM25Protocol:
    def test_is_sparse_embedding_function(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        assert isinstance(ef, SparseEmbeddingFunction)

    def test_support_persistence(self):
        from pyseekdb.utils.embedding_functions.bm25_sparse_embedding_function import (
            BM25SparseEmbeddingFunction,
        )

        ef = BM25SparseEmbeddingFunction()
        assert SparseEmbeddingFunction.support_persistence(ef)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
