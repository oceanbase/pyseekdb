"""Weighted RRF requests retain the existing rank dictionary interface."""

import copy
import json
from unittest.mock import Mock

import pytest
from pymysql import err

from pyseekdb.client.client_seekdb_server import RemoteServerClient
from pyseekdb.client.collection import Collection


@pytest.fixture
def client():
    client = RemoteServerClient()
    client._ensure_connection = Mock()
    client._execute_query_with_cursor = Mock(
        side_effect=[[], [{"query_sql": "SELECT _id, _score FROM results"}], [{"_id": "doc", "_score": 0.02}]]
    )
    return client


@pytest.mark.parametrize("weights", [None, [1, 1], [0.5, 1.0], [1, 0], [0, 1]])
def test_rank_reaches_serialized_request_unchanged(client, weights):
    rank = {"rrf": {"rank_constant": 60, "rank_window_size": 20}}
    if weights is not None:
        rank["rrf"]["weights"] = weights
    original = copy.deepcopy(rank)
    collection = Collection(client=client, name="rrf", dimension=3, embedding_function=None)

    result = collection.hybrid_search(
        query={"where_document": {"$contains": "quasar"}},
        knn={"query_embeddings": [1.0, 0.0, 0.0], "n_results": 3},
        rank=rank,
        n_results=5,
        include=[],
    )

    # Only double quotes are escaped in this controlled request.
    set_sql = client._execute_query_with_cursor.call_args_list[0].args[1]
    payload = set_sql.removeprefix("SET @search_parm = '").removesuffix("'")
    request = json.loads(payload.replace('\\"', '"'))
    assert request == {
        "query": {"query_string": {"fields": ["document"], "query": "quasar"}},
        "knn": {"field": "embedding", "query_vector": [1.0, 0.0, 0.0], "k": 3},
        "rank": original,
        "size": 5,
        "_source": ["_id"],
    }
    assert rank == original
    assert result == {"ids": [["doc"]], "distances": [[0.02]]}


def test_omitting_rank_preserves_request(client):
    request = client._build_search_parm(query=None, knn=None, rank=None, n_results=5, include=[])
    assert request == {"size": 5, "_source": ["_id"]}


@pytest.mark.parametrize("error_code", [1064, 1149, 1210, 1235])
def test_backend_rejection_propagates_without_retry(client, error_code):
    rejection = err.OperationalError(error_code, "weighted RRF rejected")
    client._execute_query_with_cursor.side_effect = [[], rejection]
    collection = Collection(client=client, name="rrf", dimension=3, embedding_function=None)

    with pytest.raises(err.OperationalError) as caught:
        collection.hybrid_search(
            query={"where_document": {"$contains": "quasar"}},
            knn={"query_embeddings": [1.0, 0.0, 0.0]},
            rank={"rrf": {"weights": [0.5, 1.0]}},
        )

    assert caught.value is rejection
    assert client._execute_query_with_cursor.call_count == 2
