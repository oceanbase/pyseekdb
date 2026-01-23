"""
Unit tests for HybridSearch/Collection return_fields API surface and internal mapping.
"""

from pathlib import Path
import sys

import pytest

# Ensure local src/ is on sys.path so we import the in-repo pyseekdb,
# not an already-installed version in the virtualenv.
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # type: ignore
from pyseekdb.client.collection import Collection  # type: ignore
from pyseekdb.client.hybrid_search import HybridSearch  # type: ignore


class _CapturingClient:
    mode = "dummy"

    def __init__(self) -> None:
        self.captured = None

    def _collection_hybrid_search(self, **kwargs):  # type: ignore[no-untyped-def]
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

        collection.hybrid_search(
            query={"where_document": {"$contains": "hi"}}, return_fields=["document"]
        )
        assert client.captured is not None
        assert client.captured["return_fields"] == ["document"]

    def test_collection_hybrid_search_passes_return_fields_from_builder(self) -> None:
        client = _CapturingClient()
        collection = Collection(client=client, name="test", dimension=3)

        hs = HybridSearch().query({"where_document": {"$contains": "hi"}}).return_fields(
            []
        )
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
            collection.hybrid_search(
                query={"where_document": {"$contains": "hi"}}, _source=["document"]
            )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

