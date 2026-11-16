"""RRF route weights against real engines.

Set SEEKDB_REQUIRE_WEIGHTED_RRF=1 when testing an engine containing seekdb #2382.
This makes missing support a failure instead of a skip on older CI engines.
"""

import os
import uuid

import pytest

from pyseekdb import HNSWConfiguration
from pyseekdb.client.client_base import _has_database_error_code

QUERY = {"where_document": {"$contains": "quasar"}}
KNN = {"query_embeddings": [0.0, 0.0, 0.0], "n_results": 3}
RRF = {"rank_constant": 60, "rank_window_size": 10}
UNSUPPORTED_RRF_CODES = (1064, 1149, 1235, 5006, 4007, -5006, -4007)


def _search(collection, weights=None):
    rank = dict(RRF)
    if weights is not None:
        rank["weights"] = weights
    return collection.hybrid_search(query=QUERY, knn=KNN, rank={"rrf": rank}, n_results=10, include=[])


@pytest.fixture
def rrf_collection(db_client):
    name = f"weighted_rrf_{uuid.uuid4().hex[:12]}"
    collection = db_client.create_collection(
        name=name,
        configuration=HNSWConfiguration(dimension=3, distance="l2"),
        embedding_function=None,
    )
    try:
        # The default four decimal places obscure small differences in RRF scores.
        collection._client._execute("SET SESSION div_precision_increment = 12")
        collection.add(
            ids=["text_first", "shared", "vector_first", "vector_second"],
            documents=[
                "quasar quasar quasar",
                "quasar astronomy astronomy astronomy",
                "galaxy galaxy",
                "planet planet",
            ],
            embeddings=[[3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        collection.refresh_index()
        yield collection
    finally:
        db_client.delete_collection(name)


@pytest.fixture
def weighted_rrf_collection(rrf_collection):
    # Verify the same request works without weights before identifying old engines.
    _search(rrf_collection)
    try:
        _search(rrf_collection, [1, 1])
    except Exception as exc:
        if os.environ.get("SEEKDB_REQUIRE_WEIGHTED_RRF") == "1":
            raise
        # MySQL and embedded error codes for unsupported options/parser syntax.
        if not _has_database_error_code(exc, UNSUPPORTED_RRF_CODES):
            raise
        pytest.skip("Engine lacks weighted RRF; use a build containing seekdb #2382")
    return rrf_collection


def test_unweighted_rrf_and_multiknn_remain_available(rrf_collection):
    assert set(_search(rrf_collection)["ids"][0]) == {"text_first", "shared", "vector_first", "vector_second"}
    result = rrf_collection.hybrid_search(
        knn={"query_embeddings": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], "n_results": 3},
        rank={"rrf": RRF},
        n_results=10,
        include=[],
    )
    assert set(result["ids"][0]) == {"text_first", "shared", "vector_first", "vector_second"}


def test_weighted_request_propagates_unsupported_engine_error(rrf_collection):
    default = _search(rrf_collection)
    try:
        unit = _search(rrf_collection, [1, 1])
    except Exception as exc:
        assert _has_database_error_code(exc, UNSUPPORTED_RRF_CODES), str(exc)
        if os.environ.get("SEEKDB_REQUIRE_WEIGHTED_RRF") == "1":
            raise
        assert _search(rrf_collection) == default
    else:
        assert unit == default


def test_unweighted_candidates_and_scores(rrf_collection):
    text = rrf_collection.hybrid_search(query=QUERY, n_results=10, include=[])
    vector = rrf_collection.hybrid_search(knn=KNN, n_results=10, include=[])
    assert text["ids"][0] == ["text_first", "shared"]
    assert vector["ids"][0] == ["vector_first", "vector_second", "shared"]
    expected = {
        "text_first": 1 / 61,
        "shared": 1 / 62 + 1 / 63,
        "vector_first": 1 / 61,
        "vector_second": 1 / 62,
    }
    fused = _search(rrf_collection)
    assert dict(zip(fused["ids"][0], fused["distances"][0], strict=True)) == pytest.approx(expected)


def test_unit_weights_preserve_results_and_ties(weighted_rrf_collection):
    collection = weighted_rrf_collection
    assert _search(collection, [1, 1]) == _search(collection)

    # Symmetric route rankings produce equal RRF scores for these two documents.
    args = {
        "query": {"where_document": {"$contains": "quasar"}},
        "knn": {"query_embeddings": [2.0, 0.0, 0.0], "n_results": 2},
        "n_results": 10,
        "include": [],
    }
    # Filter both routes to the same pair with opposite rankings.
    args["query"]["where"] = {"#id": {"$in": ["text_first", "shared"]}}
    args["knn"]["where"] = args["query"]["where"]
    default = collection.hybrid_search(**args, rank={"rrf": RRF})
    unit = collection.hybrid_search(**args, rank={"rrf": {**RRF, "weights": [1, 1]}})
    assert unit == default
    assert set(default["ids"][0]) == {"text_first", "shared"}
    assert default["distances"][0][0] == pytest.approx(default["distances"][0][1])


@pytest.mark.parametrize("weights", [[0.5, 1.0], [1.0, 0.5], [1, 0], [0, 1]])
def test_weighted_scores_and_missing_routes(weighted_rrf_collection, weights):
    collection = weighted_rrf_collection
    text = collection.hybrid_search(query=QUERY, n_results=10, include=[])
    vector = collection.hybrid_search(knn=KNN, n_results=10, include=[])
    assert text["ids"][0] == ["text_first", "shared"]
    assert vector["ids"][0] == ["vector_first", "vector_second", "shared"]

    expected = dict.fromkeys(["text_first", "shared", "vector_first", "vector_second"], 0.0)
    for weight, result in zip(weights, (text, vector), strict=True):
        for rank, doc_id in enumerate(result["ids"][0], start=1):
            expected[doc_id] += weight / (60 + rank)

    fused = _search(collection, weights)
    scores = dict(zip(fused["ids"][0], fused["distances"][0], strict=True))
    assert scores == pytest.approx(expected)
    assert fused["distances"][0] == sorted(fused["distances"][0], reverse=True)
    if weights[0] < weights[1]:
        assert fused["ids"][0].index("vector_first") < fused["ids"][0].index("text_first")
    else:
        assert fused["ids"][0].index("text_first") < fused["ids"][0].index("vector_first")

    # Weighting cannot alter either route's own candidates or order.
    assert collection.hybrid_search(query=QUERY, n_results=10, include=[]) == text
    assert collection.hybrid_search(knn=KNN, n_results=10, include=[]) == vector


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [1],
        [1, 1, 1],
        [0, 0],
        [-1, 1],
        [1, -1],
        ["bad", 1],
        [None, 1],
        [True, 1],
        "bad",
        None,
        {},
        [[1], 1],
        [float("nan"), 1],
        [float("inf"), 1],
        [float("-inf"), 1],
    ],
)
def test_invalid_weights_raise(weighted_rrf_collection, weights):
    with pytest.raises(Exception) as caught:
        weighted_rrf_collection.hybrid_search(
            query=QUERY, knn=KNN, rank={"rrf": {**RRF, "weights": weights}}, n_results=10, include=[]
        )
    # A backend rejection, rather than a Python request-building error.
    assert _has_database_error_code(
        caught.value,
        (1064, 1149, 1210, 1235, 3064, 3140, 4002, 4007, 5006, 5351, -4002, -4007, -5006, -5351),
    ), str(caught.value)


@pytest.mark.parametrize("multiple_knn_dicts", [False, True])
def test_weighted_multiknn_rejected(weighted_rrf_collection, multiple_knn_dicts):
    knn = (
        [KNN, {**KNN, "query_embeddings": [3.0, 0.0, 0.0]}]
        if multiple_knn_dicts
        else {**KNN, "query_embeddings": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}
    )
    with pytest.raises(Exception) as caught:
        weighted_rrf_collection.hybrid_search(query=QUERY, knn=knn, rank={"rrf": {**RRF, "weights": [1, 1]}})
    assert _has_database_error_code(caught.value, (1235, 4007, -4007)), str(caught.value)
