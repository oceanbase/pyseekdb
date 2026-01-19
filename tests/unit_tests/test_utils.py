"""
Test utilities for pyseekdb unit tests.

Provides helper classes and functions for testing, including environment variable management.
"""

import os
from contextlib import contextmanager
from typing import Dict, Optional


class EnvGuard:
    """
    Environment variable guard for safe testing.

    Saves current environment variables, allows temporary modification,
    and automatically restores them when done. Can be used as a context manager
    or with explicit save/restore methods.

    Example usage as context manager:
        with EnvGuard(OPENAI_API_KEY="test-key", CUSTOM_VAR="value"):
            # Environment variables are set here
            assert os.environ["OPENAI_API_KEY"] == "test-key"
        # Original values are restored automatically

    Example usage with explicit methods:
        guard = EnvGuard()
        guard.set(OPENAI_API_KEY="test-key")
        # ... test code ...
        guard.restore()

    Example usage with save/restore:
        guard = EnvGuard()
        guard.save("OPENAI_API_KEY", "CUSTOM_VAR")
        os.environ["OPENAI_API_KEY"] = "test-key"
        # ... test code ...
        guard.restore()
    """

    def __init__(self, **env_vars: Optional[str]):
        """
        Initialize EnvGuard with optional environment variables to set.

        Args:
            **env_vars: Environment variables to set immediately.
                       Use None to unset a variable.
        """
        self._saved: Dict[str, Optional[str]] = {}
        if env_vars:
            self.set(**env_vars)

    def save(self, *var_names: str) -> "EnvGuard":
        """
        Save current values of specified environment variables.

        Args:
            *var_names: Names of environment variables to save.

        Returns:
            Self for method chaining.

        Example:
            guard = EnvGuard()
            guard.save("OPENAI_API_KEY", "CUSTOM_VAR")
        """
        for var_name in var_names:
            self._saved[var_name] = os.environ.get(var_name)
        return self

    def set(self, **env_vars: Optional[str]) -> "EnvGuard":
        """
        Set environment variables, saving their current values.

        Args:
            **env_vars: Environment variables to set.
                       Use None to unset a variable.

        Returns:
            Self for method chaining.

        Example:
            guard = EnvGuard()
            guard.set(OPENAI_API_KEY="test-key", CUSTOM_VAR=None)  # None unsets
        """
        for var_name, value in env_vars.items():
            # Save current value if not already saved
            if var_name not in self._saved:
                self._saved[var_name] = os.environ.get(var_name)

            # Set or unset the variable
            if value is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = value
        return self

    def restore(self) -> "EnvGuard":
        """
        Restore all saved environment variables to their original values.

        Returns:
            Self for method chaining.

        Example:
            guard = EnvGuard()
            guard.set(OPENAI_API_KEY="test-key")
            # ... test code ...
            guard.restore()
        """
        for var_name, original_value in self._saved.items():
            if original_value is None:
                # Variable didn't exist, so remove it
                os.environ.pop(var_name, None)
            else:
                # Restore original value
                os.environ[var_name] = original_value
        self._saved.clear()
        return self

    def __enter__(self) -> "EnvGuard":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, automatically restoring environment variables."""
        self.restore()

    def __repr__(self) -> str:
        """String representation of EnvGuard."""
        saved_vars = ", ".join(self._saved.keys())
        return f"EnvGuard(saved={saved_vars})"


@contextmanager
def env_guard(**env_vars: Optional[str]):
    """
    Context manager for temporarily setting environment variables.

    This is a convenience function that creates an EnvGuard and uses it
    as a context manager.

    Args:
        **env_vars: Environment variables to set temporarily.
                   Use None to unset a variable.

    Example:
        with env_guard(OPENAI_API_KEY="test-key", CUSTOM_VAR="value"):
            # Environment variables are set here
            assert os.environ["OPENAI_API_KEY"] == "test-key"
        # Original values are restored automatically
    """
    with EnvGuard(**env_vars):
        yield
