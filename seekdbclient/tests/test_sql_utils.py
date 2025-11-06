"""
Test cases for sql_utils module
Tests SqlStringifier class and _quote_string function
"""
import pytest
import sys
from pathlib import Path
import importlib.util

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import directly from module files to avoid triggering __init__.py imports
sys_modules = sys.modules

# Import sql_utils
sql_utils_path = project_root / "seekdbclient" / "client" / "sql_utils.py"
spec = importlib.util.spec_from_file_location("seekdbclient.client.sql_utils", str(sql_utils_path))
sql_utils = importlib.util.module_from_spec(spec)
sys_modules['seekdbclient.client.sql_utils'] = sql_utils
spec.loader.exec_module(sql_utils)

SqlStringifier = sql_utils.SqlStringifier
_quote_string = sql_utils._quote_string


class TestQuoteString:
    """Test cases for _quote_string function"""
    
    def test_quote_string_with_default_quote(self):
        """Test _quote_string with default quote character"""
        result = _quote_string("test", "'")
        assert result == "'test'"
    
    def test_quote_string_with_custom_quote(self):
        """Test _quote_string with custom quote character"""
        result = _quote_string("test", '"')
        assert result == '"test"'
    
    def test_quote_string_with_backtick(self):
        """Test _quote_string with backtick quote"""
        result = _quote_string("test", "`")
        assert result == "`test`"
    
    def test_quote_string_with_empty_string(self):
        """Test _quote_string with empty string"""
        result = _quote_string("", "'")
        assert result == "''"
    
    def test_quote_string_with_numeric_string(self):
        """Test _quote_string with numeric string"""
        result = _quote_string("123", "'")
        assert result == "'123'"
    
    def test_quote_string_converts_to_string(self):
        """Test _quote_string converts non-string values to string"""
        result = _quote_string(123, "'")
        assert result == "'123'"


class TestSqlStringifier:
    """Test cases for SqlStringifier class"""
    
    def test_init_with_defaults(self):
        """Test SqlStringifier initialization with default parameters"""
        stringifier = SqlStringifier()
        assert stringifier._quote == "'"
        assert stringifier._identifier == "`"
    
    def test_init_with_custom_quote(self):
        """Test SqlStringifier initialization with custom quote"""
        stringifier = SqlStringifier(quote='"')
        assert stringifier._quote == '"'
        assert stringifier._identifier == "`"
    
    def test_init_with_custom_identifier(self):
        """Test SqlStringifier initialization with custom identifier"""
        stringifier = SqlStringifier(identifier='"')
        assert stringifier._quote == "'"
        assert stringifier._identifier == '"'
    
    def test_init_with_both_custom(self):
        """Test SqlStringifier initialization with both custom parameters"""
        stringifier = SqlStringifier(quote='"', identifier='[')
        assert stringifier._quote == '"'
        assert stringifier._identifier == '['
    
    # ==================== stringify_value Tests ====================
    
    def test_stringify_value_none(self):
        """Test stringify_value with None returns NULL"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(None)
        assert result == "NULL"
    
    def test_stringify_value_string_simple(self):
        """Test stringify_value with simple string"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("hello")
        assert result == "'hello'"
    
    def test_stringify_value_string_with_quotes(self):
        """Test stringify_value escapes quotes in strings"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("It's a test")
        assert result == "'It\\'s a test'"
    
    def test_stringify_value_string_with_multiple_quotes(self):
        """Test stringify_value escapes multiple quotes"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("It's a 'test'")
        assert result == "'It\\'s a \\'test\\''"
    
    def test_stringify_value_string_with_backslash(self):
        """Test stringify_value escapes backslashes"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("path\\to\\file")
        assert result == "'path\\\\to\\\\file'"
    
    def test_stringify_value_string_with_backslash_and_quote(self):
        """Test stringify_value escapes both backslash and quote"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("It's C:\\path")
        assert result == "'It\\'s C:\\\\path'"
    
    def test_stringify_value_string_empty(self):
        """Test stringify_value with empty string"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("")
        assert result == "''"
    
    def test_stringify_value_string_with_newline(self):
        """Test stringify_value with newline character"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("line1\nline2")
        assert result == "'line1\nline2'"
    
    def test_stringify_value_string_with_tab(self):
        """Test stringify_value with tab character"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("col1\tcol2")
        assert result == "'col1\tcol2'"
    
    def test_stringify_value_integer(self):
        """Test stringify_value with integer"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(42)
        assert result == "42"
    
    def test_stringify_value_negative_integer(self):
        """Test stringify_value with negative integer"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(-42)
        assert result == "-42"
    
    def test_stringify_value_zero(self):
        """Test stringify_value with zero"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(0)
        assert result == "0"
    
    def test_stringify_value_float(self):
        """Test stringify_value with float"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(3.14)
        assert result == "3.14"
    
    def test_stringify_value_negative_float(self):
        """Test stringify_value with negative float"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(-3.14)
        assert result == "-3.14"
    
    def test_stringify_value_float_zero(self):
        """Test stringify_value with float zero"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(0.0)
        assert result == "0.0"
    
    def test_stringify_value_large_float(self):
        """Test stringify_value with large float"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(1.23456789e10)
        assert result == "12345678900.0"
    
    def test_stringify_value_other_type(self):
        """Test stringify_value with other type (fallback)"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value([1, 2, 3])
        assert result == "'[1, 2, 3]'"
    
    def test_stringify_value_boolean_true(self):
        """Test stringify_value with boolean True"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(True)
        # Boolean values are converted to string and quoted
        assert result == 'True'
    
    def test_stringify_value_boolean_false(self):
        """Test stringify_value with boolean False"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value(False)
        # Boolean values are converted to string and quoted
        assert result == 'False'
    
    def test_stringify_value_with_custom_quote(self):
        """Test stringify_value with custom quote character"""
        stringifier = SqlStringifier(quote='"')
        result = stringifier.stringify_value('test with "quotes"')
        assert result == '"test with \\"quotes\\""'
    
    def test_stringify_value_unicode_string(self):
        """Test stringify_value with unicode string"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("测试")
        assert result == "'测试'"
    
    def test_stringify_value_unicode_with_quotes(self):
        """Test stringify_value with unicode string containing quotes"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_value("测试's value")
        assert result == "'测试\\'s value'"
    
    # ==================== stringify_id Tests ====================
    
    def test_stringify_id_valid_string(self):
        """Test stringify_id with valid string identifier"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_id("table_name")
        assert result == "`table_name`"
    
    def test_stringify_id_with_underscore(self):
        """Test stringify_id with underscore in identifier"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_id("my_table_name")
        assert result == "`my_table_name`"
    
    def test_stringify_id_with_digits(self):
        """Test stringify_id with digits in identifier"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_id("table123")
        assert result == "`table123`"
    
    def test_stringify_id_empty_string(self):
        """Test stringify_id with empty string"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_id("")
        assert result == "``"
    
    def test_stringify_id_with_special_chars(self):
        """Test stringify_id with special characters (should be quoted as-is)"""
        stringifier = SqlStringifier()
        result = stringifier.stringify_id("table-name")
        assert result == "`table-name`"
    
    def test_stringify_id_with_custom_identifier(self):
        """Test stringify_id with custom identifier quote"""
        stringifier = SqlStringifier(identifier='"')
        result = stringifier.stringify_id("table_name")
        assert result == '"table_name"'
    
    def test_stringify_id_raises_error_when_none(self):
        """Test stringify_id raises ValueError when id_name is None"""
        stringifier = SqlStringifier()
        with pytest.raises(ValueError, match="Identifier shouldn't be null"):
            stringifier.stringify_id(None)
    
    def test_stringify_id_raises_error_when_not_string(self):
        """Test stringify_id raises ValueError when id_name is not a string"""
        stringifier = SqlStringifier()
        with pytest.raises(ValueError, match="Identifier should be string type"):
            stringifier.stringify_id(123)
    
    def test_stringify_id_raises_error_with_integer(self):
        """Test stringify_id raises ValueError with integer"""
        stringifier = SqlStringifier()
        with pytest.raises(ValueError, match="Identifier should be string type.*int"):
            stringifier.stringify_id(42)
    
    def test_stringify_id_raises_error_with_list(self):
        """Test stringify_id raises ValueError with list"""
        stringifier = SqlStringifier()
        with pytest.raises(ValueError, match="Identifier should be string type.*list"):
            stringifier.stringify_id(["table", "name"])
    
    def test_stringify_id_raises_error_with_dict(self):
        """Test stringify_id raises ValueError with dict"""
        stringifier = SqlStringifier()
        with pytest.raises(ValueError, match="Identifier should be string type.*dict"):
            stringifier.stringify_id({"key": "value"})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

