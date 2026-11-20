import pytest


@pytest.fixture(autouse=True)
def hermetic_paths(tmp_path, monkeypatch):
    """Tests never read the developer's real .env (credentials, data paths) and never write
    into the repo's data/ or research/ (tests that need other locations override these)."""
    monkeypatch.setenv("AUTOTRADE_DOTENV", str(tmp_path / "no-such.env"))
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "hermetic-data"))
    monkeypatch.setenv("AUTOTRADE_RESEARCH", str(tmp_path / "hermetic-research"))
