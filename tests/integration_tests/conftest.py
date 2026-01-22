"""
Pytest configuration and shared fixtures for pyseekdb tests.
Provides parameterized client fixtures for testing across embedded, server, and oceanbase modes.
"""

import contextlib
import os
import sys
from pathlib import Path

import pytest

# Add project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb  # noqa: E402

# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get("SEEKDB_PATH", os.path.join(project_root, "seekdb.db"))
SEEKDB_DATABASE = os.environ.get("SEEKDB_DATABASE", "test")

# Server mode
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "2881"))
SERVER_DATABASE = os.environ.get("SERVER_DATABASE", "test")
SERVER_USER = os.environ.get("SERVER_USER", "root")
SERVER_PASSWORD = os.environ.get("SERVER_PASSWORD", "")

# OceanBase mode
OB_HOST = os.environ.get("OB_HOST", "localhost")
OB_PORT = int(os.environ.get("OB_PORT", "11202"))
OB_TENANT = os.environ.get("OB_TENANT", "mysql")
OB_DATABASE = os.environ.get("OB_DATABASE", "test")
OB_USER = os.environ.get("OB_USER", "root")
OB_PASSWORD = os.environ.get("OB_PASSWORD", "")


# ==================== Client Factory Functions ====================
def create_embedded_client():
    """Create an embedded client instance."""
    try:
        import pylibseekdb  # noqa: F401
    except ImportError:
        pytest.skip("seekdb embedded package is not installed")

    return pyseekdb.Client(path=SEEKDB_PATH, database=SEEKDB_DATABASE)


def create_server_client():
    """Create a server client instance."""
    client = pyseekdb.Client(
        host=SERVER_HOST,
        port=SERVER_PORT,
        tenant="sys",
        database=SERVER_DATABASE,
        user=SERVER_USER,
        password=SERVER_PASSWORD,
    )

    # Test connection
    try:
        result = client._server._execute("SELECT 1 as test")
        assert result and result[0].get("test") == 1
    except Exception as exc:
        pytest.fail(f"seekdb server connection failed ({SERVER_HOST}:{SERVER_PORT}): {exc}")

    return client


def create_oceanbase_client():
    """Create an OceanBase client instance."""
    client = pyseekdb.Client(
        host=OB_HOST,
        port=OB_PORT,
        tenant=OB_TENANT,
        database=OB_DATABASE,
        user=OB_USER,
        password=OB_PASSWORD,
    )

    # Test connection
    try:
        result = client._server._execute("SELECT 1 as test")
        assert result and result[0].get("test") == 1
    except Exception as exc:
        pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {exc}")

    return client


# ==================== AdminClient Factory Functions ====================
def create_embedded_admin_client():
    """Create an embedded admin client instance."""
    try:
        import pylibseekdb  # noqa: F401
    except ImportError:
        pytest.skip("seekdb embedded package is not installed")

    return pyseekdb.AdminClient(path=SEEKDB_PATH)


def create_server_admin_client():
    """Create a server admin client instance."""
    admin = pyseekdb.AdminClient(
        host=SERVER_HOST,
        port=SERVER_PORT,
        tenant="sys",
        user=SERVER_USER,
        password=SERVER_PASSWORD,
    )

    # Test connection
    try:
        result = admin._server._execute("SELECT 1 as test")
        assert result and result[0].get("test") == 1
    except Exception as exc:
        pytest.fail(f"seekdb server connection failed ({SERVER_HOST}:{SERVER_PORT}): {exc}")

    return admin


def create_oceanbase_admin_client():
    """Create an OceanBase admin client instance."""
    admin = pyseekdb.AdminClient(host=OB_HOST, port=OB_PORT, tenant=OB_TENANT, user=OB_USER, password=OB_PASSWORD)

    # Test connection
    try:
        result = admin._server._execute("SELECT 1 as test")
        assert result and result[0].get("test") == 1
    except Exception as exc:
        pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {exc}")

    return admin


# ==================== Parameterized Client Fixtures ====================
@pytest.fixture(params=["embedded", "server", "oceanbase"])
def db_client(request):
    """
    Parameterized fixture that provides clients for all three modes.

    This fixture automatically creates test variants for embedded, server, and oceanbase modes.

    Usage:
        def test_my_feature(db_client):
            collection = db_client.get_or_create_collection(...)
            # test logic here

    This will automatically run 3 times: once for each client mode.
    Generated test names will be:
        - test_my_feature[embedded]
        - test_my_feature[server]
        - test_my_feature[oceanbase]
    """
    mode = request.param

    if mode == "embedded":
        client = create_embedded_client()
    elif mode == "server":
        client = create_server_client()
    elif mode == "oceanbase":
        client = create_oceanbase_client()
    else:
        raise ValueError(f"Unknown client mode: {mode}")

    yield client

    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def embedded_client():
    """Fixture for embedded client only."""
    client = create_embedded_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def server_client():
    """Fixture for server client only."""
    client = create_server_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def oceanbase_client():
    """Fixture for OceanBase client only."""
    client = create_oceanbase_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


# ==================== Parameterized AdminClient Fixtures ====================
@pytest.fixture(params=["embedded", "server", "oceanbase"])
def admin_client(request):
    """
    Parameterized fixture that provides admin clients for all three modes.

    This fixture automatically creates test variants for embedded, server, and oceanbase modes.

    Usage:
        def test_my_admin_feature(admin_client):
            admin_client.create_database("test_db")
            # test logic here

    This will automatically run 3 times: once for each client mode.
    Generated test names will be:
        - test_my_admin_feature[embedded]
        - test_my_admin_feature[server]
        - test_my_admin_feature[oceanbase]
    """
    mode = request.param

    if mode == "embedded":
        client = create_embedded_admin_client()
    elif mode == "server":
        client = create_server_admin_client()
    elif mode == "oceanbase":
        client = create_oceanbase_admin_client()
    else:
        raise ValueError(f"Unknown admin client mode: {mode}")

    yield client

    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def embedded_admin_client():
    """Fixture for embedded admin client only."""
    client = create_embedded_admin_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def server_admin_client():
    """Fixture for server admin client only."""
    client = create_server_admin_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()


@pytest.fixture
def oceanbase_admin_client():
    """Fixture for OceanBase admin client only."""
    client = create_oceanbase_admin_client()
    yield client
    with contextlib.suppress(Exception):
        if hasattr(client, "close"):
            client.close()
