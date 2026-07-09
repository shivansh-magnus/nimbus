"""
Day-10 MCP server unit tests.

These test the server's helper functions and tool handlers *directly* as
Python callables — NOT over the wire.  No actual MCP transport is started,
no HTTP port is bound.

Covers:
  - _validate_dataset_path allow-list enforcement
  - list_local_datasets returns expected entries
  - get_run_report handles missing/valid run_ids correctly
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

# Import the server module and its helpers
from automl_agents.mcp_server import (
    _validate_dataset_path,
    ALLOWED_DATA_DIR,
    ROOT,
)


# ---------------------------------------------------------------------------
# _validate_dataset_path
# ---------------------------------------------------------------------------

class TestValidateDatasetPath:
    """Tests for the path allow-list enforcer."""

    def test_rejects_absolute_path_outside_allowed_dir(self):
        """An absolute path that doesn't live under data/raw/ must be rejected."""
        with pytest.raises(ValueError, match="must live under"):
            _validate_dataset_path("C:\\Windows\\System32\\evil.csv")

    def test_rejects_relative_path_traversal(self):
        """Path traversal (../../etc/passwd) must be rejected."""
        with pytest.raises(ValueError, match="must live under"):
            _validate_dataset_path("data/raw/../../etc/passwd")

    def test_rejects_parent_directory_traversal(self):
        """../secrets.csv relative to project root must be rejected."""
        with pytest.raises(ValueError, match="must live under"):
            _validate_dataset_path("../secrets.csv")

    def test_accepts_valid_path_under_data_raw(self):
        """A path that actually exists under data/raw/ must be accepted."""
        synthetic = ALLOWED_DATA_DIR / "synthetic_ground_truth.csv"
        if not synthetic.exists():
            pytest.skip("synthetic_ground_truth.csv not present; run generate_synthetic first")
        result = _validate_dataset_path(str(synthetic))
        assert result == synthetic

    def test_rejects_nonexistent_file_under_allowed_dir(self):
        """A path under data/raw/ that doesn't exist must still be rejected."""
        with pytest.raises(ValueError, match="not found"):
            _validate_dataset_path("data/raw/does_not_exist_12345.csv")


# ---------------------------------------------------------------------------
# list_local_datasets (called directly, not over MCP wire)
# ---------------------------------------------------------------------------

class TestListLocalDatasets:
    """Tests for the list_local_datasets tool handler."""

    def test_returns_list_of_dicts(self):
        """list_local_datasets should return a list, each entry a dict with 'id'."""
        from automl_agents.mcp_server import list_local_datasets

        datasets = asyncio.run(list_local_datasets())
        assert isinstance(datasets, list)
        for ds in datasets:
            assert "id" in ds
            assert "path" in ds
            assert "target_column" in ds

    def test_includes_synthetic_if_present(self):
        """If synthetic_ground_truth.csv exists, it should appear in the list."""
        from automl_agents.mcp_server import list_local_datasets

        synthetic = ALLOWED_DATA_DIR / "synthetic_ground_truth.csv"
        if not synthetic.exists():
            pytest.skip("synthetic_ground_truth.csv not present")

        datasets = asyncio.run(list_local_datasets())
        ids = [d["id"] for d in datasets]
        assert any("synthetic" in id_str for id_str in ids)


# ---------------------------------------------------------------------------
# get_run_report
# ---------------------------------------------------------------------------

class TestGetRunReport:
    """Tests for the get_run_report tool handler."""

    def test_returns_error_for_unknown_run_id(self):
        """An unknown run_id should return an error message, not raise."""
        from automl_agents.mcp_server import get_run_report

        result = asyncio.run(get_run_report("nonexistent_run_id_12345"))
        assert "Error" in result

    def test_rejects_path_separator_in_run_id(self):
        """run_id with path separators should be rejected."""
        from automl_agents.mcp_server import get_run_report

        result = asyncio.run(get_run_report("../../etc/passwd"))
        assert "invalid characters" in result

    def test_returns_contents_for_existing_run(self):
        """If a run with a report.md exists, its contents should be returned."""
        from automl_agents.mcp_server import get_run_report

        # Create a temporary run directory with a fake report
        test_run_id = "test_run_for_mcp_unit_test"
        test_dir = ROOT / "runs" / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)
        report_file = test_dir / "report.md"
        report_content = "# Test Report\nThis is a test."

        try:
            report_file.write_text(report_content, encoding="utf-8")
            result = asyncio.run(get_run_report(test_run_id))
            assert result == report_content
        finally:
            # Clean up
            report_file.unlink(missing_ok=True)
            test_dir.rmdir()
