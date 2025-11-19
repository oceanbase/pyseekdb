"""
Utility functions and classes for SQL string generation and escaping in seekdb client.

Provides helpers to safely stringify values and SQL identifiers for insertion into SQL expressions.
"""

from typing import Optional, Union


def _quote_string(value, quote: str):
    return quote + str(value) + quote


class SqlStringifier:
    """
    Translate values into strings in SQL.
    """

    def __init__(self, *, quote: str = "'", identifier: str = "`"):
        self._quote = quote
        self._identifier = identifier

    def stringify_value(self, value: Optional[Union[str, int, float, bytes]]):
        if value is None:
            return "NULL"
        if isinstance(value, bytes):
            # For varbinary type, convert bytes to hex string and use UNHEX function
            hex_str = value.hex()
            return f"UNHEX('{hex_str}')"
        if isinstance(value, str):
            # Check if it's a hex string (for varbinary IDs)
            # If it looks like a hex string (even length, only hex chars), use UNHEX
            # Otherwise, treat as regular string
            if len(value) > 0 and len(value) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in value):
                # Likely a hex string for varbinary, use UNHEX
                return f"UNHEX('{value}')"
            formatted = value.replace('\\', '\\\\').replace(self._quote, f"\\{self._quote}")
            return _quote_string(formatted, self._quote)
        if isinstance(value, (int, float)):
            return str(value)
        return _quote_string(str(value), self._quote)

    def stringify_id(self, id_name: str):
        if id_name is None:
            raise ValueError("Identifier shouldn't be null")
        if not isinstance(id_name, str):
            raise ValueError(f"Identifier should be string type, but got {type(id_name).__name__}")
        return _quote_string(id_name, self._identifier)
