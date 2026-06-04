from unittest.mock import patch

import pytest

from pyseekdb.client.client_base import BaseClient


class _MinimalClientStub:
    """Minimal stub that supplies `self` for BaseClient method tests without full client initialization."""

    pass


@pytest.mark.parametrize("method_name", ["_collection_add", "_collection_update"])
def test_add_and_update_do_not_emit_embedding_success_debug_log(method_name: str) -> None:
    method = getattr(BaseClient, method_name)
    with patch("pyseekdb.client.client_base.logger.debug") as debug_mock, pytest.raises(
        ValueError, match="Number of embeddings"
    ):
        method(
            _MinimalClientStub(),
            collection_id=None,
            collection_name="test_collection",
            ids=["id1", "id2"],
            documents=["doc1", "doc2"],
            embedding_function=lambda docs: [[0.1, 0.2] for _ in docs[:1]],
        )

    assert all("✅ Successfully generated" not in str(call.args[0]) for call in debug_mock.call_args_list)


def test_upsert_does_not_emit_embedding_success_info_log() -> None:
    with patch("pyseekdb.client.client_base.logger.info") as info_mock, pytest.raises(
        ValueError, match="Number of embeddings"
    ):
        BaseClient._collection_upsert(
            _MinimalClientStub(),
            collection_id=None,
            collection_name="test_collection",
            ids=["id1", "id2"],
            documents=["doc1", "doc2"],
            embedding_function=lambda docs: [[0.1, 0.2] for _ in docs[:1]],
        )

    info_mock.assert_not_called()
