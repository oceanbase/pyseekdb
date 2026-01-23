"""
Unit tests for HybridSearch/Collection return_fields API surface and internal mapping.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

from pyseekdb.client.client_base import BaseClient
from pyseekdb.client.collection import Collection
from pyseekdb.client.hybrid_search import HybridSearch

# Ensure local src/ is on sys.path so we import the in-repo pyseekdb,
# not an already-installed version in the virtualenv.
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))


class _CapturingClient:
    mode = "dummy"

    def __init__(self) -> None:
        self.captured = None

    def _collection_hybrid_search(self, **kwargs: Any) -> dict[str, Any]:
        self.captured = kwargs
        return {"captured": kwargs}


class _DummyClient:
    """Minimal object used as `self` when calling BaseClient._build_search_parm."""


class TestHybridSearchReturnFieldsUnit:
    def test_hybrid_search_builder_return_fields_in_to_params(self) -> None:
        hs = HybridSearch().return_fields(["document", "metadata"])
        params = hs.to_params()
        assert params["return_fields"] == ["document", "metadata"]

    def test_hybrid_search_builder_return_fields_in_constructor(self) -> None:
        hs = HybridSearch(return_fields=["document"])
        params = hs.to_params()
        assert params["return_fields"] == ["document"]

    def test_collection_hybrid_search_passes_return_fields_arg(self) -> None:
        client = _CapturingClient()
        collection = Collection(client=client, name="test", dimension=3)

        collection.hybrid_search(query={"where_document": {"$contains": "hi"}}, return_fields=["document"])
        assert client.captured is not None
        assert client.captured["return_fields"] == ["document"]

    def test_collection_hybrid_search_passes_return_fields_from_builder(self) -> None:
        client = _CapturingClient()
        collection = Collection(client=client, name="test", dimension=3)

        hs = HybridSearch().query({"where_document": {"$contains": "hi"}}).return_fields([])
        collection.hybrid_search(hs)
        assert client.captured is not None
        assert client.captured["return_fields"] == ["_id"]

    def test_collection_hybrid_search_rejects_builder_and_arg_return_fields_mix(
        self,
    ) -> None:
        client = _CapturingClient()
        collection = Collection(client=client, name="test", dimension=3)

        hs = HybridSearch().return_fields(["document"])
        with pytest.raises(
            ValueError,
            match=r"Do not mix HybridSearch\.return_fields\(\) with return_fields=",
        ):
            collection.hybrid_search(hs, return_fields=["metadata"])

    def test_collection_hybrid_search_rejects_legacy_source_kwarg(self) -> None:
        client = _CapturingClient()
        collection = Collection(client=client, name="test", dimension=3)

        with pytest.raises(TypeError, match=r"Use return_fields="):
            collection.hybrid_search(query={"where_document": {"$contains": "hi"}}, _source=["document"])

    def test_hybrid_search_rejects_legacy_source_kwarg(self) -> None:
        with pytest.raises(TypeError, match=r"Use return_fields="):
            HybridSearch(_source=["document"])


class TestBuildSearchParmReturnFieldsUnit:
    def test_build_search_parm_does_not_add_source_by_default(self) -> None:
        dummy = _DummyClient()
        result = BaseClient._build_search_parm(  # type: ignore[misc]
            dummy,
            query=None,
            knn=None,
            rank=None,
            n_results=10,
        )
        assert "_source" not in result

    def test_build_search_parm_maps_return_fields_to_source(self) -> None:
        dummy = _DummyClient()
        result = BaseClient._build_search_parm(  # type: ignore[misc]
            dummy,
            query=None,
            knn=None,
            rank=None,
            n_results=10,
            return_fields=["document", "metadata"],
        )
        assert result["_source"] == ["document", "metadata"]


class TestBuildSourceFieldsUnit:
    def test_build_source_fields_defaults_to_documents_and_metadatas(self) -> None:
        dummy = _DummyClient()
        assert BaseClient._build_source_fields(  # type: ignore[misc]
            dummy, include=None
        ) == ["_id", "document", "metadata"]

    def test_build_source_fields_empty_include_is_id_only(self) -> None:
        dummy = _DummyClient()
        assert BaseClient._build_source_fields(  # type: ignore[misc]
            dummy, include=[]
        ) == ["_id"]

    def test_build_source_fields_includes_embedding_only_when_requested(self) -> None:
        dummy = _DummyClient()
        assert BaseClient._build_source_fields(  # type: ignore[misc]
            dummy, include=["embeddings"]
        ) == ["_id", "embedding"]


class _HybridSearchInferenceClient:
    """Minimal object used as `self` when calling BaseClient._collection_hybrid_search."""

    def __init__(self) -> None:
        self.captured_return_fields = None

    def _ensure_connection(self) -> None:
        return None

    def _use_context_manager_for_cursor(self) -> bool:
        return False

    def _build_source_fields(self, include: list[str] | None) -> list[str]:
        return BaseClient._build_source_fields(self, include)  # type: ignore[misc]

    def _build_search_parm(
        self,
        query: dict[str, Any] | None,
        knn: dict[str, Any] | list[dict[str, Any]] | None,
        rank: dict[str, Any] | None,
        n_results: int,
        return_fields: list[str] | None = None,
        dimension: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.captured_return_fields = return_fields
        return BaseClient._build_search_parm(
            self,
            query=query,
            knn=knn,
            rank=rank,
            n_results=n_results,
            return_fields=return_fields,
            dimension=dimension,
            **kwargs,
        )

    def _execute_query_with_cursor(
        self, _conn: Any, sql: str, _params: Any, _use_context_manager: bool = True
    ) -> list[dict[str, Any]]:
        if isinstance(sql, str) and sql.strip().upper().startswith("SELECT DBMS_HYBRID_SEARCH.GET_SQL"):
            return [{"query_sql": "SELECT 1"}]
        return []

    def _transform_sql_result(self, _result_rows: list[dict[str, Any]], _include: list[str] | None) -> dict[str, Any]:
        return {"captured_return_fields": self.captured_return_fields}


class TestHybridSearchReturnFieldsInferenceUnit:
    def test_hybrid_search_infers_return_fields_when_omitted(self) -> None:
        client = _HybridSearchInferenceClient()
        result = BaseClient._collection_hybrid_search(  # type: ignore[misc]
            client,
            collection_id=None,
            collection_name="test",
            query=None,
            knn=None,
            rank=None,
            n_results=2,
            include=None,
            return_fields=None,
            dimension=None,
        )
        assert result["captured_return_fields"] == ["_id", "document", "metadata"]

    def test_hybrid_search_does_not_override_explicit_return_fields(self) -> None:
        client = _HybridSearchInferenceClient()
        result = BaseClient._collection_hybrid_search(  # type: ignore[misc]
            client,
            collection_id=None,
            collection_name="test",
            query=None,
            knn=None,
            rank=None,
            n_results=2,
            include=None,
            return_fields=["document"],
            dimension=None,
        )
        assert result["captured_return_fields"] == ["document"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
