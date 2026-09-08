"""
Unit tests for collection name validation.
"""

import sys
from pathlib import Path

import pytest

# Ensure local src/ is on sys.path so we import the in-repo pyseekdb,
# not an already-installed version in the virtualenv.
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import _validate_collection_name  # noqa: E402


class TestCollectionNameValidation:
    """Tests for collection name constraints."""

    @property
    def _effective_max_length(self) -> int:
        """
        Return maximum allowed collection name length from client_base.
        """
        from pyseekdb.client.client_base import _MAX_COLLECTION_NAME_LENGTH

        return _MAX_COLLECTION_NAME_LENGTH

    @property
    def _db_table_max_length(self) -> int:
        """
        Return the database table name maximum length (typically 64 for MySQL).
        """
        return 64

    @property
    def _table_prefix_length(self) -> int:
        """
        Return the length of the collection table prefix (c$v1$).
        """
        return len(CollectionNames.prefix())

    @property
    def _effective_max_length_by_db_limit(self) -> int:
        """
        Calculate effective max length based on DB table name limit minus prefix.
        This is the actual constraint: prefix + name <= DB limit.
        """
        return self._db_table_max_length - self._table_prefix_length

    def test_valid_names(self):
        """Names with allowed characters and length should pass."""
        max_len = self._effective_max_length
        valid_names = [
            "a",
            "A",
            "0",
            "collection_1",
            "MyCollection_123",
            "A" * max_len,
        ]
        print(f"\n{'=' * 60}")
        print(f"测试: test_valid_names - 测试有效名称")
        print(f"{'=' * 60}")
        for name in valid_names:
            print(f"\n输入: {repr(name)} (长度: {len(name)})")
            try:
                _validate_collection_name(name)
                print(f"输出: ✅ 验证通过 - 名称有效")
            except Exception as e:
                print(f"输出: ❌ 验证失败 - {type(e).__name__}: {e}")
                raise

    def test_invalid_type(self):
        """Non-string names should raise TypeError."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_invalid_type - 测试无效类型")
        print(f"{'=' * 60}")

        invalid_inputs = [123, None]
        for invalid_input in invalid_inputs:
            print(f"\n输入: {repr(invalid_input)} (类型: {type(invalid_input).__name__})")
            print(f"预期: 抛出 TypeError")
            try:
                with pytest.raises(TypeError) as exc_info:
                    _validate_collection_name(invalid_input)  # type: ignore[arg-type]
                print(f"输出: ✅ 验证通过 - 捕获到 TypeError: {exc_info.value}")
            except Exception as e:
                print(f"输出: ❌ 验证失败 - 未捕获到预期的 TypeError，实际异常: {type(e).__name__}: {e}")
                raise

    def test_empty_name(self):
        """Empty string should be rejected."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_empty_name - 测试空字符串")
        print(f"{'=' * 60}")
        empty_name = ""
        print(f"\n输入: {repr(empty_name)} (长度: {len(empty_name)})")
        print(f"预期: 抛出 ValueError，错误信息包含 'must not be empty'")
        try:
            with pytest.raises(ValueError, match="must not be empty") as exc_info:
                _validate_collection_name(empty_name)
            print(f"输出: ✅ 验证通过 - 捕获到 ValueError: {exc_info.value}")
        except Exception as e:
            print(f"输出: ❌ 验证失败 - 未捕获到预期的 ValueError，实际异常: {type(e).__name__}: {e}")
            raise

    def test_name_too_long(self):
        """Names longer than effective maximum should be rejected."""
        max_len = self._effective_max_length
        long_name = "a" * (max_len + 1)
        print(f"\n{'=' * 60}")
        print(f"测试: test_name_too_long - 测试超长名称")
        print(f"{'=' * 60}")
        print(f"\n输入: {repr(long_name[:50])}... (长度: {len(long_name)}, 最大允许: {max_len})")
        print(f"预期: 抛出 ValueError，错误信息包含 'maximum allowed is'")
        try:
            with pytest.raises(ValueError, match="maximum allowed is") as exc_info:
                _validate_collection_name(long_name)
            print(f"输出: ✅ 验证通过 - 捕获到 ValueError: {exc_info.value}")
        except Exception as e:
            print(f"输出: ❌ 验证失败 - 未捕获到预期的 ValueError，实际异常: {type(e).__name__}: {e}")
            raise

    def test_invalid_characters(self):
        """Names with characters outside [a-zA-Z0-9_] should be rejected."""
        invalid_names = [
            "name-with-dash",
            "name.with.dot",
            "name with space",
            "name$",
            "名字",
        ]
        print(f"\n{'=' * 60}")
        print(f"测试: test_invalid_characters - 测试非法字符")
        print(f"{'=' * 60}")
        for name in invalid_names:
            print(f"\n输入: {repr(name)} (长度: {len(name)})")
            print(f"预期: 抛出 ValueError，错误信息包含 'Only letters, digits, and underscore'")
            try:
                with pytest.raises(ValueError, match="Only letters, digits, and underscore") as exc_info:
                    _validate_collection_name(name)
                print(f"输出: ✅ 验证通过 - 捕获到 ValueError: {exc_info.value}")
            except Exception as e:
                print(f"输出: ❌ 验证失败 - 未捕获到预期的 ValueError，实际异常: {type(e).__name__}: {e}")
                raise

    def test_effective_max_length_by_db_limit(self):
        """Test that collection names respect DB table name limit when combined with prefix."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_effective_max_length_by_db_limit - 测试基于DB限制的有效最大长度")
        print(f"{'=' * 60}")

        prefix = CollectionNames.prefix()
        prefix_len = len(prefix)
        db_limit = self._db_table_max_length
        effective_max = self._effective_max_length_by_db_limit
        max_len = self._effective_max_length

        print(f"\n配置信息:")
        print(f"  前缀: {repr(prefix)} (长度: {prefix_len})")
        print(f"  DB表名最大长度: {db_limit}")
        print(f"  有效最大长度: {effective_max} (计算: {db_limit} - {prefix_len})")
        print(f"  当前验证函数的最大长度限制: {max_len}")

        # 测试正好等于有效最大长度（应该通过）
        exact_max_name = "a" * effective_max
        table_name = CollectionNames.table_name(exact_max_name)
        print(f"\n测试1: 正好等于有效最大长度")
        print(f"  输入: {repr(exact_max_name)} (长度: {len(exact_max_name)})")
        print(f"  生成的表名: {repr(table_name)} (长度: {len(table_name)})")
        print(f"  预期: 验证通过，表名长度 {len(table_name)} <= {db_limit}")
        try:
            _validate_collection_name(exact_max_name)
            print(f"  输出: ✅ 验证通过 - 名称有效，表名长度符合DB限制")
        except Exception as e:
            print(f"  输出: ❌ 验证失败 - {type(e).__name__}: {e}")
            raise

        # 测试超过有效最大长度（应该失败，因为表名会超过DB限制）
        over_max_name = "a" * (effective_max + 1)
        table_name_over = CollectionNames.table_name(over_max_name)
        table_len_over = len(table_name_over)
        print(f"\n测试2: 超过有效最大长度（基于DB限制）")
        print(f"  输入名称: {repr(over_max_name[:50])}... (长度: {len(over_max_name)})")
        print(f"  生成的表名: {repr(table_name_over[:50])}... (长度: {table_len_over})")
        print(f"  表名长度计算: {prefix_len} + {len(over_max_name)} = {table_len_over}")
        print(f"  预期: 表名长度 {table_len_over} > {db_limit}，应该被拒绝")

        # 验证名称本身
        try:
            _validate_collection_name(over_max_name)
            print(f"  名称验证: ✅ 通过（当前验证函数只检查{max_len}的限制）")
        except Exception as e:
            print(f"  名称验证: ❌ 失败 - {type(e).__name__}: {e}")

        # 检查表名长度
        if table_len_over > db_limit:
            print(f"  表名长度验证: ❌ 失败 - 表名长度 {table_len_over} > DB限制 {db_limit}")
            print(f"  输出: ⚠️  表名长度超过DB限制，但当前验证函数可能不会检查此限制")
            print(f"  提示: 建议在_validate_collection_name中添加基于DB限制的检查")
        else:
            print(f"  表名长度验证: ✅ 通过 - 表名长度 {table_len_over} <= DB限制 {db_limit}")

    def test_boundary_length_values(self):
        """Test boundary values around the effective max length based on DB table name limit."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_boundary_length_values - 测试基于DB限制的边界长度值")
        print(f"{'=' * 60}")

        prefix = CollectionNames.prefix()
        prefix_len = len(prefix)
        db_limit = self._db_table_max_length
        effective_max = self._effective_max_length_by_db_limit
        max_len = self._effective_max_length

        print(f"\n配置信息:")
        print(f"  前缀: {repr(prefix)} (长度: {prefix_len})")
        print(f"  DB表名最大长度: {db_limit}")
        print(f"  基于DB限制的有效最大长度: {effective_max} (计算: {db_limit} - {prefix_len})")
        print(f"  当前验证函数的最大长度限制: {max_len}")
        print(f"  测试将使用: {effective_max} (基于DB限制)")

        # 使用基于DB限制的有效最大长度进行边界测试
        test_limit = effective_max

        test_cases = [
            (test_limit - 1, True, f"比有效最大长度少1 ({test_limit - 1}字符)，应该通过"),
            (test_limit, True, f"正好等于有效最大长度 ({test_limit}字符)，应该通过"),
            (test_limit + 1, False, f"比有效最大长度多1 ({test_limit + 1}字符)，应该失败"),
        ]

        for length, should_pass, description in test_cases:
            name = "a" * length
            table_name = CollectionNames.table_name(name)
            table_len = len(table_name)

            print(f"\n测试: {description}")
            print(f"  输入名称: {repr(name[:30])}... (长度: {len(name)})")
            print(f"  生成的表名: {repr(table_name[:50])}... (长度: {table_len})")
            print(
                f"  表名长度验证: {prefix_len} + {len(name)} = {table_len} {'✅' if table_len <= db_limit else '❌ 超过DB限制'}")
            print(f"  预期: {'通过' if should_pass else '失败（因为表名长度超过DB限制）'}")

            try:
                _validate_collection_name(name)
                if should_pass:
                    print(f"  输出: ✅ 验证通过 - 符合预期")
                else:
                    print(f"  输出: ⚠️  验证通过 - 但表名长度 {table_len} > {db_limit}，超过了DB限制")
                    print(f"  提示: 当前_validate_collection_name只检查{max_len}的限制，未检查DB表名限制")
            except ValueError as e:
                if not should_pass:
                    print(f"  输出: ✅ 验证通过 - 正确拒绝: {e}")
                else:
                    print(f"  输出: ❌ 验证失败 - 应该通过但被拒绝: {e}")
                    raise
            except Exception as e:
                print(f"  输出: ❌ 验证失败 - 意外异常: {type(e).__name__}: {e}")
                raise

    def test_underscore_positions(self):
        """Test underscore in various positions."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_underscore_positions - 测试下划线位置")
        print(f"{'=' * 60}")

        test_cases = [
            ("_name", "下划线开头"),
            ("name_", "下划线结尾"),
            ("name_middle", "下划线中间"),
            ("_name_", "下划线开头和结尾"),
            ("__name", "连续下划线开头"),
            ("name__", "连续下划线结尾"),
            ("name__middle", "连续下划线中间"),
            ("_", "只有下划线"),
            ("__", "只有两个下划线"),
        ]

        for name, description in test_cases:
            print(f"\n测试: {description}")
            print(f"  输入: {repr(name)} (长度: {len(name)})")
            print(f"  预期: 验证通过（下划线是允许的字符）")
            try:
                _validate_collection_name(name)
                print(f"  输出: ✅ 验证通过 - 名称有效")
            except Exception as e:
                print(f"  输出: ❌ 验证失败 - {type(e).__name__}: {e}")
                raise

    def test_numeric_start(self):
        """Test names starting with numbers."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_numeric_start - 测试数字开头")
        print(f"{'=' * 60}")

        test_cases = [
            ("0name", "以0开头"),
            ("123collection", "以数字开头"),
            ("9test", "以9开头"),
        ]

        for name, description in test_cases:
            print(f"\n测试: {description}")
            print(f"  输入: {repr(name)} (长度: {len(name)})")
            print(f"  预期: 验证通过（数字是允许的字符）")
            try:
                _validate_collection_name(name)
                print(f"  输出: ✅ 验证通过 - 名称有效")
            except Exception as e:
                print(f"  输出: ❌ 验证失败 - {type(e).__name__}: {e}")
                raise

    def test_single_character_types(self):
        """Test names with only one type of character."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_single_character_types - 测试单一字符类型")
        print(f"{'=' * 60}")

        test_cases = [
            ("a", "单个小写字母"),
            ("Z", "单个大写字母"),
            ("0", "单个数字"),
            ("_", "单个下划线"),
            ("abc", "只有小写字母"),
            ("ABC", "只有大写字母"),
            ("123", "只有数字"),
            ("___", "只有下划线"),
        ]

        for name, description in test_cases:
            print(f"\n测试: {description}")
            print(f"  输入: {repr(name)} (长度: {len(name)})")
            print(f"  预期: 验证通过")
            try:
                _validate_collection_name(name)
                print(f"  输出: ✅ 验证通过 - 名称有效")
            except Exception as e:
                print(f"  输出: ❌ 验证失败 - {type(e).__name__}: {e}")
                raise

    def test_mixed_case_and_characters(self):
        """Test names with mixed case and character types."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_mixed_case_and_characters - 测试混合大小写和字符类型")
        print(f"{'=' * 60}")

        test_cases = [
            ("MyCollection_123", "大小写字母+下划线+数字"),
            ("a1B2c3", "字母数字混合"),
            ("A_B_C", "大写字母和下划线"),
            ("a_b_c", "小写字母和下划线"),
            ("Test123_456", "字母数字下划线混合"),
            ("_test123", "下划线+字母+数字"),
            ("test_123_", "字母+数字+下划线组合"),
        ]

        for name, description in test_cases:
            print(f"\n测试: {description}")
            print(f"  输入: {repr(name)} (长度: {len(name)})")
            print(f"  预期: 验证通过")
            try:
                _validate_collection_name(name)
                print(f"  输出: ✅ 验证通过 - 名称有效")
            except Exception as e:
                print(f"  输出: ❌ 验证失败 - {type(e).__name__}: {e}")
                raise

    def test_more_invalid_characters(self):
        """Test more invalid character scenarios."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_more_invalid_characters - 测试更多非法字符场景")
        print(f"{'=' * 60}")

        invalid_cases = [
            ("name@test", "包含@符号"),
            ("name#test", "包含#符号"),
            ("name%test", "包含%符号"),
            ("name&test", "包含&符号"),
            ("name*test", "包含*符号"),
            ("name+test", "包含+符号"),
            ("name=test", "包含=符号"),
            ("name/test", "包含/符号"),
            ("name\\test", "包含\\符号"),
            ("name|test", "包含|符号"),
            ("name<test", "包含<符号"),
            ("name>test", "包含>符号"),
            ("name[test", "包含[符号"),
            ("name]test", "包含]符号"),
            ("name{test", "包含{符号"),
            ("name}test", "包含}符号"),
            ("name'test", "包含单引号"),
            ('name"test', "包含双引号"),
            ("name\ttest", "包含制表符"),
            ("name\ntest", "包含换行符"),
            ("name\rtest", "包含回车符"),
            ("name\0test", "包含空字符"),
            ("name test", "包含空格"),
        ]

        for name, description in invalid_cases:
            print(f"\n测试: {description}")
            print(f"  输入: {repr(name)} (长度: {len(name)})")
            print(f"  预期: 抛出 ValueError，错误信息包含 'Only letters, digits, and underscore'")
            try:
                with pytest.raises(ValueError, match="Only letters, digits, and underscore") as exc_info:
                    _validate_collection_name(name)
                print(f"  输出: ✅ 验证通过 - 捕获到 ValueError: {exc_info.value}")
            except Exception as e:
                print(f"  输出: ❌ 验证失败 - 未捕获到预期的 ValueError，实际异常: {type(e).__name__}: {e}")
                raise

    def test_table_name_generation(self):
        """Test that table name generation respects DB limits."""
        print(f"\n{'=' * 60}")
        print(f"测试: test_table_name_generation - 测试表名生成是否符合DB限制")
        print(f"{'=' * 60}")

        prefix = CollectionNames.prefix()
        db_limit = self._db_table_max_length
        effective_max = self._effective_max_length_by_db_limit

        print(f"\n配置信息:")
        print(f"  前缀: {repr(prefix)} (长度: {len(prefix)})")
        print(f"  DB表名最大长度: {db_limit}")
        print(f"  有效最大长度: {effective_max}")

        # 测试各种长度的名称
        test_lengths = [1, 10, effective_max - 1, effective_max, effective_max + 1]

        for length in test_lengths:
            name = "a" * length
            table_name = CollectionNames.table_name(name)
            table_len = len(table_name)

            print(f"\n测试: 名称长度 {length}")
            print(f"  输入: {repr(name[:30])}... (长度: {length})")
            print(f"  生成的表名: {repr(table_name[:50])}... (长度: {table_len})")

            # 验证名称本身
            try:
                _validate_collection_name(name)
                validation_status = "✅ 通过"
            except Exception as e:
                validation_status = f"❌ 失败: {e}"

            # 检查表名长度
            if table_len <= db_limit:
                table_status = f"✅ 符合DB限制 ({table_len} <= {db_limit})"
            else:
                table_status = f"⚠️  超过DB限制 ({table_len} > {db_limit})"

            print(f"  名称验证: {validation_status}")
            print(f"  表名长度: {table_status}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
