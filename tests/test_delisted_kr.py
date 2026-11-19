from datetime import date

import pandas as pd
import pytest

from autotrade.data import delisted_kr
from autotrade.data.delisted_kr import fetch_delisted_daily, supplement_delisted_kr
from autotrade.data.store import load_daily, save_meta


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(delisted_kr.time, "sleep", lambda s: None)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path))
    return tmp_path


def korean_cols_df(n=5, high=12):
    idx = pd.DatetimeIndex(pd.bdate_range("2018-01-02", periods=n), name="날짜")
    return pd.DataFrame({"시가": 10, "고가": high, "저가": 9, "종가": 11,
                         "거래량": 1000, "등락률": 0.5}, index=idx)


def save_kr_meta(delisted: dict[str, bool]):
    save_meta("kr", pd.DataFrame({"delisted": delisted},
                                 index=pd.Index(list(delisted), name="symbol")))


def test_columns_renamed_and_source_tagged(monkeypatch):
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: korean_cols_df())
    df = fetch_delisted_daily("000000", date(2018, 1, 1), date(2018, 12, 31))
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "source"]
    assert (df["source"] == "pykrx").all()
    assert df.index.tz is None


def test_requests_adjusted_prices(monkeypatch):
    seen = {}

    def fake(s, e, sym, adjusted):
        seen.update(s=s, e=e, sym=sym, adj=adjusted)
        return korean_cols_df()

    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", fake)
    fetch_delisted_daily("000000", date(2018, 1, 1), date(2018, 12, 31))
    assert seen == {"s": "20180101", "e": "20181231", "sym": "000000", "adj": True}


def test_empty_result_reported_not_saved(monkeypatch, tmp_data):
    save_kr_meta({"111111": True})
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: pd.DataFrame())
    rep = supplement_delisted_kr(date(2018, 1, 1))
    assert rep.empty == ["111111"] and rep.fetched == 0 and rep.failed == []
    assert load_daily("kr", "111111") is None


def test_only_delisted_saved_and_bad_data_failed(monkeypatch, tmp_data):
    save_kr_meta({"111111": True, "222222": True, "333333": False, "444444": True})

    def fake(s, e, sym, adjusted):
        if sym == "222222":
            return korean_cols_df(high=5)  # high < close -> ohlc inconsistent
        if sym == "444444":
            raise RuntimeError("boom")
        return korean_cols_df()

    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", fake)
    rep = supplement_delisted_kr(date(2018, 1, 1))
    assert rep.fetched == 1
    assert sorted(rep.failed) == ["222222", "444444"]
    assert load_daily("kr", "111111")["source"].eq("pykrx").all()
    assert load_daily("kr", "222222") is None
    assert load_daily("kr", "333333") is None
