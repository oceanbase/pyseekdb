from unittest.mock import patch

import pytest

from pyseekdb.client.client_base import BaseClient


class _DummyClient:
    pass


def test_upsert_does_not_log_embedding_generation_success_at_info_level() -> None:
    with patch("pyseekdb.client.client_base.logger.info") as info_mock, pytest.raises(
        ValueError, match="Number of embeddings"
    ):
        BaseClient._collection_upsert(
            _DummyClient(),
            collection_id=None,
            collection_name="test_collection",
            ids=["id1", "id2"],
            documents=["doc1", "doc2"],
            embedding_function=lambda docs: [[0.1, 0.2] for _ in docs[:1]],
        )

    info_mock.assert_not_called()
