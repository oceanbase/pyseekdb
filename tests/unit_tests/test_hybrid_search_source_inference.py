"""
Unit tests for hybrid_search `_source` inference from include.

Public API exposes only `include`. The SDK infers a minimal OceanBase GET_SQL `_source`
allowlist from `include` to reduce response payload size.
"""

from typing import Any

import pytest

from pyseekdb.client.client_base import BaseClient
from pyseekdb.client.collection import Collection


class _CapturingClient:
    mode = "dummy"

    def __init__(self) -> None:
        self.captured: dict[str, Any] | None = None

    def _collection_hybrid_search(self, **kwargs: Any) -> dict[str, Any]:
        self.captured = kwargs
        return {"captured": kwargs}


class _DummyClient:
    def _build_source_fields(self, include: list[str] | None) -> list[str]:
        return BaseClient._build_source_fields(self, include)


class TestHybridSearchPublicSurfaceUnit:
    def test_collection_forwards_include_only(self) -> None:
        client = _CapturingClient()
        collection = Collection(client=client, name="test", dimension=3)
        collection.hybrid_search(query={"where_document": {"$contains": "hi"}}, include=["documents"])
        assert client.captured is not None
        assert client.captured["include"] == ["documents"]
        assert "return_fields" not in client.captured


class TestBuildSearchParmSourceInferenceUnit:
    def test_build_search_parm_sets_source_from_default_include(self) -> None:
        dummy = _DummyClient()
        result = BaseClient._build_search_parm(
            dummy,
            query=None,
            knn=None,
            rank=None,
            n_results=10,
            include=None,
        )
        assert result["_source"] == ["_id", "document", "metadata"]

    def test_build_search_parm_sets_source_from_empty_include(self) -> None:
        dummy = _DummyClient()
        result = BaseClient._build_search_parm(
            dummy,
            query=None,
            knn=None,
            rank=None,
            n_results=10,
            include=[],
        )
        assert result["_source"] == ["_id"]


class TestBuildSourceFieldsUnit:
    def test_build_source_fields_defaults_to_documents_and_metadatas(self) -> None:
        dummy = _DummyClient()
        assert BaseClient._build_source_fields(dummy, include=None) == ["_id", "document", "metadata"]

    def test_build_source_fields_empty_include_is_id_only(self) -> None:
        dummy = _DummyClient()
        assert BaseClient._build_source_fields(dummy, include=[]) == ["_id"]

    def test_build_source_fields_includes_embedding_only_when_requested(self) -> None:
        dummy = _DummyClient()
        assert BaseClient._build_source_fields(dummy, include=["embeddings"]) == ["_id", "embedding"]

    def test_build_source_fields_rejects_singular_aliases(self) -> None:
        dummy = _DummyClient()
        with pytest.raises(ValueError, match=r"include only supports"):
            BaseClient._build_source_fields(dummy, include=["document"])

    def test_build_source_fields_rejects_unknown_fields(self) -> None:
        dummy = _DummyClient()
        with pytest.raises(ValueError, match=r"include only supports"):
            BaseClient._build_source_fields(dummy, include=["ids"])

