"""
Day-10 Typer CLI unit tests.

Uses Typer's CliRunner to invoke subcommands without starting a subprocess.
These tests verify the CLI wiring — they don't run full pipeline end-to-end
(that's what test_pipeline.py and the stress test are for).
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from automl_agents.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

class TestCLIHelp:
    """Verify the CLI entrypoint and subcommands show up in --help."""

    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "nimbus" in result.output.lower() or "automl" in result.output.lower()

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--csv" in result.output
        assert "--target" in result.output
        assert "--provider" in result.output

    def test_stress_test_help(self):
        result = runner.invoke(app, ["stress-test", "--help"])
        assert result.exit_code == 0

    def test_verify_providers_help(self):
        result = runner.invoke(app, ["verify-providers", "--help"])
        assert result.exit_code == 0

    def test_generate_data_help(self):
        result = runner.invoke(app, ["generate-data", "--help"])
        assert result.exit_code == 0
        assert "--n-rows" in result.output
        assert "--seed" in result.output

    def test_download_data_help(self):
        result = runner.invoke(app, ["download-data", "--help"])
        assert result.exit_code == 0

    def test_serve_mcp_help(self):
        result = runner.invoke(app, ["serve-mcp", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "--port" in result.output


# ---------------------------------------------------------------------------
# Subcommands that can be tested without hitting an LLM
# ---------------------------------------------------------------------------

class TestCLISubcommands:
    """Tests that run actual subcommands (no LLM calls)."""

    def test_run_with_nonexistent_csv_fails(self):
        """nimbus run --csv /nonexistent/path.csv should exit code 1."""
        result = runner.invoke(app, [
            "run",
            "--csv", "/nonexistent/path/to/data.csv",
            "--target", "churn",
        ])
        assert result.exit_code == 1

    def test_generate_data(self, tmp_path):
        """nimbus generate-data should create a synthetic CSV."""
        result = runner.invoke(app, [
            "generate-data",
            "--output-dir", str(tmp_path),
            "--n-rows", "50",
            "--seed", "123",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "synthetic_ground_truth.csv").exists()
