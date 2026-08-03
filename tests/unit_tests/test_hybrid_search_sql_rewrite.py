"""Regression tests for SQL returned by DBMS_HYBRID_SEARCH.GET_SQL."""

import pytest

from pyseekdb.client.client_base import _unquote_json_extract_expressions


@pytest.mark.parametrize(
    "query_sql",
    [
        (
            "SELECT MATCH(`document`) AGAINST ('machine') "
            "WHERE (JSON_EXTRACT(metadata, '$.category')) = 'AI' "
            "ORDER BY `_score` DESC, `__pk_increment`"
        ),
        ("SELECT * FROM `c$v1$test` WHERE JSON_EXTRACT(metadata, '$.score') >= 90 ORDER BY `_distance`"),
    ],
)
def test_unquoted_json_extract_does_not_remove_adjacent_identifier_quotes(query_sql: str) -> None:
    assert _unquote_json_extract_expressions(query_sql) == query_sql


@pytest.mark.parametrize(
    ("query_sql", "expected"),
    [
        (
            "WHERE `(JSON_EXTRACT(metadata, '$.category'))` = 'AI' ORDER BY `_score`",
            "WHERE (JSON_EXTRACT(metadata, '$.category')) = 'AI' ORDER BY `_score`",
        ),
        (
            "WHERE (`JSON_EXTRACT(metadata, '$.score')`) >= 90 ORDER BY `_distance`",
            "WHERE (JSON_EXTRACT(metadata, '$.score')) >= 90 ORDER BY `_distance`",
        ),
        (
            "WHERE `json_extract(metadata, '$.tag')` = 'ml'",
            "WHERE json_extract(metadata, '$.tag') = 'ml'",
        ),
    ],
)
def test_quoted_json_extract_expression_is_unquoted(query_sql: str, expected: str) -> None:
    assert _unquote_json_extract_expressions(query_sql) == expected
