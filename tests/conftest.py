import pytest


@pytest.fixture(autouse=True)
def no_repo_dotenv(tmp_path, monkeypatch):
    """Tests never read the developer's real .env (credentials, data paths)."""
    monkeypatch.setenv("AUTOTRADE_DOTENV", str(tmp_path / "no-such.env"))
