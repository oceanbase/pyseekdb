"""
Utility functions and classes for SQL string generation and escaping in seekdb client.

Provides helpers to safely stringify values and SQL identifiers for insertion into SQL expressions.
"""

from collections.abc import Sequence
from typing import Any

from pymysql.converters import escape_string


def escape_percent_for_sql(value: str) -> str:
    """
    Escape percent signs in SQL string values to prevent format string interpretation.

    When pymysql's cursor.execute() processes SQL strings, it may interpret % as format
    specifiers. This function escapes % to %% to prevent that.

    Args:
        value: String value that may contain % characters

    Returns:
        String with % escaped as %%
    """
    return value.replace("%", "%%")


def is_query_sql(sql: str) -> bool:
    if not sql:
        return False
    sql_upper = sql.strip().upper()
    return (
        sql_upper.startswith("SELECT")
        or sql_upper.startswith("SHOW")
        or sql_upper.startswith("DESCRIBE")
        or sql_upper.startswith("DESC")
    )


def render_sql_with_params(sql: str, params: Sequence[Any]) -> str:
    if not params:
        return sql
    parts = sql.split("%s")
    placeholder_count = len(parts) - 1
    if placeholder_count != len(params):
        raise ValueError(f"Expected {placeholder_count} parameters, got {len(params)}")
    rendered_parts = [parts[0]]
    for param, part in zip(params, parts[1:], strict=True):
        if param is None:
            replacement = "NULL"
        elif isinstance(param, (bytes, bytearray, memoryview)):
            text = bytes(param).decode("utf-8", errors="replace")
            replacement = f"'{escape_string(text)}'"
        elif isinstance(param, (int, float)):
            replacement = str(param)
        elif isinstance(param, str):
            replacement = f"'{escape_string(param)}'"
        else:
            replacement = f"'{escape_string(str(param))}'"
        rendered_parts.append(replacement)
        rendered_parts.append(part)
    return "".join(rendered_parts)
