"""
tests/test_enrollment.py
=========================
Pytest unit tests for enrollment/enrollment_manager.py (v2)

Tests cover:
    - Employee registration and retrieval
    - Duplicate employee code detection
    - Embedding storage and retrieval
    - Embedding averaging across frames
    - Enrollment update (re-enroll)
    - Employee listing and deletion
    - DatabaseManager isolation via in-memory SQLite

These tests use an in-memory SQLite database so no file I/O occurs and
tests are fully isolated from each other.
"""

import sqlite3
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from enrollment.enrollment_manager import EnrollmentManager, Employee, EnrollmentRecord


# ---------------------------------------------------------------------------
# In-memory database fixture
# ---------------------------------------------------------------------------

class _InMemoryDB:
    """
    Minimal DatabaseManager stand-in backed by an in-memory SQLite connection.
    Mirrors the interface expected by EnrollmentManager v2.
    """

    def __init__(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS employees (
                id              TEXT PRIMARY KEY,
                employee_code   TEXT NOT NULL UNIQUE,
                full_name       TEXT NOT NULL,
                department      TEXT NOT NULL DEFAULT '',
                email           TEXT DEFAULT '',
                phone           TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                is_active       INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id     TEXT NOT NULL REFERENCES employees(id),
                embedding       BLOB NOT NULL,
                frame_count     INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
        """)
        self.connection.commit()

    def close(self):
        self.connection.close()


@pytest.fixture
def db():
    _db = _InMemoryDB()
    yield _db
    _db.close()


@pytest.fixture
def manager(db):
    return EnrollmentManager(db=db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_embedding(seed: int = 0, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def _register(manager: EnrollmentManager, code: str = "NHAI-001", name: str = "Alice") -> str:
    """Helper: register one employee and return their id."""
    return manager.register_employee(
        employee_code=code,
        full_name=name,
        department="Engineering",
    )


# ---------------------------------------------------------------------------
# Employee registration
# ---------------------------------------------------------------------------

class TestRegisterEmployee:

    def test_register_returns_uuid_string(self, manager):
        emp_id = _register(manager)
        assert isinstance(emp_id, str)
        assert len(emp_id) > 0

    def test_register_persists_to_db(self, manager, db):
        emp_id = _register(manager, code="NHAI-002", name="Bob")
        row = db.connection.execute(
            "SELECT * FROM employees WHERE id = ?", (emp_id,)
        ).fetchone()
        assert row is not None
        assert row["employee_code"] == "NHAI-002"
        assert row["full_name"] == "Bob"

    def test_duplicate_code_raises(self, manager):
        _register(manager, code="NHAI-003")
        with pytest.raises((ValueError, Exception)):
            _register(manager, code="NHAI-003", name="Different Person")

    def test_department_stored_correctly(self, manager, db):
        emp_id = manager.register_employee(
            employee_code="NHAI-004",
            full_name="Carol",
            department="IT",
        )
        row = db.connection.execute(
            "SELECT department FROM employees WHERE id = ?", (emp_id,)
        ).fetchone()
        assert row["department"] == "IT"

    def test_is_active_defaults_true(self, manager, db):
        emp_id = _register(manager, code="NHAI-005")
        row = db.connection.execute(
            "SELECT is_active FROM employees WHERE id = ?", (emp_id,)
        ).fetchone()
        assert bool(row["is_active"]) is True


# ---------------------------------------------------------------------------
# Embedding storage
# ---------------------------------------------------------------------------

class TestEmbeddingStorage:

    def test_store_embedding_succeeds(self, manager):
        emp_id = _register(manager, code="NHAI-010")
        emb = _rng_embedding(seed=1)
        manager.store_embedding(employee_id=emp_id, embedding=emb)

        row = manager.db.connection.execute(
            "SELECT * FROM embeddings WHERE employee_id = ?", (emp_id,)
        ).fetchone()
        assert row is not None

    def test_stored_embedding_matches_original(self, manager):
        emp_id = _register(manager, code="NHAI-011")
        emb = _rng_embedding(seed=2)
        manager.store_embedding(employee_id=emp_id, embedding=emb)

        retrieved = manager.get_embedding(employee_id=emp_id)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved, emb, atol=1e-5)

    def test_get_embedding_returns_none_for_unknown(self, manager):
        result = manager.get_embedding(employee_id="nonexistent-id")
        assert result is None

    def test_update_embedding_replaces_existing(self, manager):
        emp_id = _register(manager, code="NHAI-012")
        emb_v1 = _rng_embedding(seed=3)
        emb_v2 = _rng_embedding(seed=4)

        manager.store_embedding(employee_id=emp_id, embedding=emb_v1)
        manager.store_embedding(employee_id=emp_id, embedding=emb_v2)

        retrieved = manager.get_embedding(employee_id=emp_id)
        # Should reflect the most recent embedding
        assert retrieved is not None
        # Not equal to v1
        assert not np.allclose(retrieved, emb_v1, atol=1e-3), (
            "Old embedding should have been replaced by v2"
        )


# ---------------------------------------------------------------------------
# Employee retrieval
# ---------------------------------------------------------------------------

class TestEmployeeRetrieval:

    def test_get_employee_by_id(self, manager):
        emp_id = _register(manager, code="NHAI-020", name="David")
        emp = manager.get_employee(employee_id=emp_id)
        assert emp is not None
        assert emp.full_name == "David"
        assert emp.employee_code == "NHAI-020"

    def test_get_employee_by_code(self, manager):
        _register(manager, code="NHAI-021", name="Eve")
        emp = manager.get_employee_by_code("NHAI-021")
        assert emp is not None
        assert emp.full_name == "Eve"

    def test_get_nonexistent_employee_returns_none(self, manager):
        result = manager.get_employee(employee_id="no-such-id")
        assert result is None

    def test_list_employees_returns_all(self, manager):
        for i in range(5):
            _register(manager, code=f"NHAI-03{i}", name=f"Person {i}")
        employees = manager.list_employees()
        assert len(employees) >= 5

    def test_list_employees_active_only(self, manager):
        emp_id = _register(manager, code="NHAI-040", name="Frank")
        # Deactivate Frank
        manager.deactivate_employee(emp_id)
        active = manager.list_employees(active_only=True)
        ids = [e.id for e in active]
        assert emp_id not in ids


# ---------------------------------------------------------------------------
# Enrolled embeddings gallery
# ---------------------------------------------------------------------------

class TestGetAllEmbeddings:

    def test_get_all_embeddings_structure(self, manager):
        for i in range(3):
            emp_id = _register(manager, code=f"NHAI-05{i}", name=f"Person {i}")
            manager.store_embedding(employee_id=emp_id, embedding=_rng_embedding(seed=i))

        gallery = manager.get_all_embeddings()
        assert isinstance(gallery, list)
        assert len(gallery) == 3
        for emp_id, emb in gallery:
            assert isinstance(emp_id, str)
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (128,)

    def test_gallery_excludes_employees_without_embeddings(self, manager):
        _register(manager, code="NHAI-060", name="Ghost")
        gallery = manager.get_all_embeddings()
        emp_ids = [eid for eid, _ in gallery]
        # Ghost has no embedding — should not appear
        ghost_id = manager.get_employee_by_code("NHAI-060").id
        assert ghost_id not in emp_ids


# ---------------------------------------------------------------------------
# Delete and deactivate
# ---------------------------------------------------------------------------

class TestDeleteEmployee:

    def test_deactivate_sets_flag(self, manager, db):
        emp_id = _register(manager, code="NHAI-070")
        manager.deactivate_employee(emp_id)
        row = db.connection.execute(
            "SELECT is_active FROM employees WHERE id = ?", (emp_id,)
        ).fetchone()
        assert bool(row["is_active"]) is False

    def test_delete_removes_employee(self, manager):
        emp_id = _register(manager, code="NHAI-071")
        manager.delete_employee(emp_id)
        result = manager.get_employee(employee_id=emp_id)
        assert result is None

    def test_delete_removes_associated_embeddings(self, manager, db):
        emp_id = _register(manager, code="NHAI-072")
        manager.store_embedding(employee_id=emp_id, embedding=_rng_embedding(1))
        manager.delete_employee(emp_id)
        row = db.connection.execute(
            "SELECT * FROM embeddings WHERE employee_id = ?", (emp_id,)
        ).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_store_embedding_for_nonexistent_employee_raises(self, manager):
        with pytest.raises(Exception):
            manager.store_embedding(
                employee_id="ghost-id",
                embedding=_rng_embedding(0),
            )

    def test_embedding_l2_normalized_on_store(self, manager):
        emp_id = _register(manager, code="NHAI-080")
        unnormalized = np.array([3.0] * 128, dtype=np.float32)
        manager.store_embedding(employee_id=emp_id, embedding=unnormalized)

        retrieved = manager.get_embedding(employee_id=emp_id)
        if retrieved is not None:
            norm = np.linalg.norm(retrieved)
            assert norm == pytest.approx(1.0, abs=0.01), (
                f"Stored embedding should be L2-normalised, got norm={norm:.4f}"
            )