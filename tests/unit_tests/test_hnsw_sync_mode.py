"""Unit tests for HNSW sync_mode gating and the removed write-time sync flag."""

import inspect

import pytest

from pyseekdb import Collection, Namespace, Version
from pyseekdb.client.client_base import HNSW_SYNC_MODE_MIN_VERSION, BaseClient, _hnsw_sync_mode_supported


class _VersionClient:
    def __init__(self, db_type: str, version: str):
        self._info = (db_type, Version(version))

    def detect_db_type_and_version(self):
        return self._info


@pytest.mark.parametrize(
    ("db_type", "version", "supported"),
    [
        ("seekdb", "1.3.0.0", True),
        ("seekdb", "1.3.1.0", True),
        ("SeekDB", "1.2.0.0", False),
        ("oceanbase", "4.6.1.0", False),
        ("oceanbase", "4.5.0.0", False),
    ],
)
def test_hnsw_sync_mode_support_matrix(db_type, version, supported):
    assert _hnsw_sync_mode_supported(db_type, Version(version)) is supported


def test_assert_rejects_backends_that_do_not_accept_sync_mode():
    client = _VersionClient("oceanbase", "4.6.1.0")
    with pytest.raises(ValueError, match=r"oceanbase 4\.6\.1\.0") as exc_info:
        BaseClient._assert_hnsw_sync_mode_supported(client, "immediate")
    message = str(exc_info.value)
    assert "seekdb >= 1.3.0.0" in message
    assert "collection was not created" in message
    assert Version("1.3.0.0") == HNSW_SYNC_MODE_MIN_VERSION


def test_assert_allows_seekdb_130():
    client = _VersionClient("seekdb", "1.3.0.0")
    BaseClient._assert_hnsw_sync_mode_supported(client, "async")


def test_old_seekdb_rejects_sync_mode():
    client = _VersionClient("seekdb", "1.2.0.0")
    with pytest.raises(ValueError, match=r"seekdb 1\.2\.0\.0"):
        BaseClient._assert_hnsw_sync_mode_supported(client, "immediate")


@pytest.mark.parametrize("method_name", ["add", "update", "upsert", "delete"])
@pytest.mark.parametrize("api", [Collection, Namespace])
def test_writes_do_not_accept_sync_keyword(api, method_name):
    assert "sync" not in inspect.signature(getattr(api, method_name)).parameters
