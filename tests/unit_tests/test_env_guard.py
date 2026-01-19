"""
Example tests demonstrating how to use EnvGuard for environment variable management.

This file shows various usage patterns for the EnvGuard class.
"""

import os
import pytest

from .test_utils import EnvGuard, env_guard


class TestEnvGuard:
    """Examples of using EnvGuard in tests"""

    def test_basic_context_manager_usage(self):
        """Example: Using EnvGuard as a context manager"""
        original_value = os.environ.get("TEST_VAR")

        with EnvGuard(TEST_VAR="test-value"):
            assert os.environ["TEST_VAR"] == "test-value"

        # Value is restored automatically
        assert os.environ.get("TEST_VAR") == original_value

    def test_setting_multiple_variables(self):
        """Example: Setting multiple environment variables at once"""
        original_var1 = os.environ.get("VAR1")
        original_var2 = os.environ.get("VAR2")

        with EnvGuard(VAR1="value1", VAR2="value2"):
            assert os.environ["VAR1"] == "value1"
            assert os.environ["VAR2"] == "value2"

        # Values are restored
        assert os.environ.get("VAR1") == original_var1
        assert os.environ.get("VAR2") == original_var2

    def test_unsetting_variables(self):
        """Example: Unsetting environment variables (setting to None)"""
        # Set a variable first
        os.environ["TEMP_VAR"] = "original"

        with EnvGuard(TEMP_VAR=None):
            assert "TEMP_VAR" not in os.environ

        # Variable is restored
        assert os.environ["TEMP_VAR"] == "original"
        os.environ.pop("TEMP_VAR")  # Clean up

    def test_explicit_save_and_restore(self):
        """Example: Using explicit save/restore methods"""
        os.environ["MY_VAR"] = "original"

        guard = EnvGuard()
        guard.save("MY_VAR")
        os.environ["MY_VAR"] = "modified"
        assert os.environ["MY_VAR"] == "modified"

        guard.restore()
        assert os.environ["MY_VAR"] == "original"
        os.environ.pop("MY_VAR")  # Clean up

    def test_method_chaining(self):
        """Example: Method chaining with EnvGuard"""
        guard = EnvGuard()
        guard.save("VAR1", "VAR2").set(VAR1="new1", VAR2="new2")

        assert os.environ["VAR1"] == "new1"
        assert os.environ["VAR2"] == "new2"

        guard.restore()

    def test_nested_context_managers(self):
        """Example: Nested EnvGuard context managers"""
        original = os.environ.get("NESTED_VAR")

        with EnvGuard(NESTED_VAR="outer"):
            assert os.environ["NESTED_VAR"] == "outer"

            with EnvGuard(NESTED_VAR="inner"):
                assert os.environ["NESTED_VAR"] == "inner"

            # Back to outer value
            assert os.environ["NESTED_VAR"] == "outer"

        # Back to original
        assert os.environ.get("NESTED_VAR") == original

    def test_env_guard_convenience_function(self):
        """Example: Using the env_guard convenience function"""
        original = os.environ.get("CONVENIENCE_VAR")

        with env_guard(CONVENIENCE_VAR="test"):
            assert os.environ["CONVENIENCE_VAR"] == "test"

        assert os.environ.get("CONVENIENCE_VAR") == original

    def test_real_world_api_key_example(self):
        """Example: Real-world usage with API keys (like in embedding function tests)"""
        original_key = os.environ.get("OPENAI_API_KEY")
        original_custom = os.environ.get("CUSTOM_OPENAI_KEY")

        # Simulate a test that needs a custom API key
        with EnvGuard(CUSTOM_OPENAI_KEY=original_key or "test-key"):
            # Test code that uses CUSTOM_OPENAI_KEY
            assert os.environ.get("CUSTOM_OPENAI_KEY") is not None

        # Original values restored
        assert os.environ.get("OPENAI_API_KEY") == original_key
        assert os.environ.get("CUSTOM_OPENAI_KEY") == original_custom


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
