import pytest


@pytest.fixture(autouse=True)
def hermetic_paths(tmp_path, monkeypatch):
    """Tests never read the developer's real .env (credentials, data paths) and never write
    into the repo's data/ or research/ (tests that need other locations override these)."""
    monkeypatch.setenv("AUTOTRADE_DOTENV", str(tmp_path / "no-such.env"))
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "hermetic-data"))
    monkeypatch.setenv("AUTOTRADE_RESEARCH", str(tmp_path / "hermetic-research"))


@pytest.fixture(autouse=True)
def no_real_pykrx(monkeypatch):
    """No test reaches KRX over the network; tests that need pykrx data patch these again."""
    from autotrade.data import delisted_kr, krx_overlay

    def blocked(*a, **k):
        raise AssertionError("a test called the real pykrx")
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", blocked)
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", blocked)


@pytest.fixture(autouse=True)
def no_real_kind(monkeypatch):
    """No test reaches KIND; `lab collect` tests see an empty delisting list unless they
    patch pipeline.kind_delisted themselves."""
    import pandas as pd
    from autotrade import pipeline
    from autotrade.data import delisted_kr

    def blocked(*a, **k):
        raise AssertionError("a test called the real KIND site")
    monkeypatch.setattr(delisted_kr, "_kind_html", blocked)
    empty = pd.DataFrame(columns=["name", "market_board", "delisted_on"],
                         index=pd.Index([], name="symbol"))
    monkeypatch.setattr(pipeline, "kind_delisted", lambda since, until=None: empty)
