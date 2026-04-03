from typer.testing import CliRunner

from feedback_intelligence.cli import app


def test_status_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "configured" in result.stdout.lower()
