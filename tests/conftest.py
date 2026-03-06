"""
Pytest Fixtures for Traot
=================================

Shared test fixtures used across all test files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile

from src.core.database import Database


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    yield temp_file.name
    Path(temp_file.name).unlink(missing_ok=True)


@pytest.fixture
def temp_db(temp_db_path):
    """Create a temporary database instance."""
    db = Database(temp_db_path)
    yield db
    db.close()
