from datetime import date

import pandas as pd
import pytest

from autotrade.data import delisted_kr
from autotrade.data.delisted_kr import fetch_delisted_daily, supplement_delisted_kr
from autotrade.data.store import load_daily, save_daily, save_meta


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


def test_suspension_gap_is_warning_not_failure(monkeypatch, tmp_data):
    save_kr_meta({"111111": True})
    df = pd.concat([korean_cols_df(5), korean_cols_df(5)])
    df.index = pd.DatetimeIndex(list(pd.bdate_range("2018-01-02", periods=5))
                                + list(pd.bdate_range("2018-02-12", periods=5)), name="날짜")
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: df)
    rep = supplement_delisted_kr(date(2018, 1, 1))
    assert rep.fetched == 1 and rep.failed == []
    assert len(load_daily("kr", "111111")) == 10


def test_zero_price_suspension_rows_dropped(monkeypatch):
    raw = korean_cols_df(5)
    raw.iloc[1:3, raw.columns.get_indexer(["시가", "고가", "저가", "거래량"])] = 0
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: raw)
    df = fetch_delisted_daily("000000", date(2018, 1, 1), date(2018, 12, 31))
    assert len(df) == 3 and (df[["open", "high", "low", "close"]] > 0).all().all()


def test_existing_other_source_skipped_and_untouched(monkeypatch, tmp_data):
    from tests.test_data import good_df
    save_kr_meta({"111111": True})
    save_daily("kr", "111111", good_df())  # source defaults to toss
    called = []
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: called.append(1))
    rep = supplement_delisted_kr(date(2018, 1, 1))
    assert rep.skipped == ["111111"] and rep.fetched == 0 and not called
    assert load_daily("kr", "111111")["source"].eq("toss").all()


def test_pykrx_uses_shared_clean_daily(monkeypatch):
    seen = []

    def spy(df):
        seen.append(len(df))
        return df
    monkeypatch.setattr(delisted_kr, "clean_daily", spy)
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", lambda *a, **k: korean_cols_df(5))
    fetch_delisted_daily("000000", date(2018, 1, 1), date(2018, 12, 31))
    assert seen == [5]


def test_failure_is_logged_with_type_and_message(monkeypatch, tmp_data, caplog):
    save_kr_meta({"444444": True})

    def boom(*a, **k):
        raise ConnectionError("KRX server closed the connection")
    monkeypatch.setattr(delisted_kr, "_pykrx_ohlcv", boom)
    rep = supplement_delisted_kr(date(2018, 1, 1))
    assert rep.failed == ["444444"]
    assert "444444" in caplog.text and "ConnectionError" in caplog.text
    assert "KRX server closed the connection" in caplog.text


def kind_row(n, board, code, name, day):
    return (f'<tr class=""><td class="first txc" scope="row">{n}</td>'
            f"<td><img src='/i.gif' class='vmiddle legend' alt='{board}'> <a id=\"companysum\" "
            f"href=\"#companysum\" onclick=\"companysummary_open('{code}'); return false;\" "
            f"title='{name}'> {name}</a> <img src='/p.gif' class='vmiddle legend' "
            f"alt='상장폐지'> </td>\n<td class=\"txc\">{day}</td>\n<td>사유</td><td></td></tr>")


KIND_PAGE = "<table><tbody>" + "".join([
    kind_row(4, "유가증권", "41731", "코람코더원리츠", "2026-11-20"),
    kind_row(3, "코넥스", "12345", "코넥스회사", "2026-10-01"),
    kind_row(2, "코스닥", "0000D", "신규코드", "2025-05-16"),
    kind_row(1, "코스닥", "05381", "아이팩토리", "2016-09-19")]) + "</tbody></table>"


def test_parse_kind_maps_codes_to_tickers_and_drops_konex():
    df = delisted_kr.parse_kind(KIND_PAGE)
    assert list(df.index) == ["417310", "0000D0", "053810"]
    assert df.loc["417310", "market_board"] == "KOSPI"
    assert df.loc["053810", "name"] == "아이팩토리"
    assert df.loc["0000D0", "delisted_on"] == "2025-05-16"


def test_parse_kind_refuses_a_truncated_page():
    cut = KIND_PAGE.replace(kind_row(2, "코스닥", "0000D", "신규코드", "2025-05-16"), "")
    with pytest.raises(delisted_kr.KindError):
        delisted_kr.parse_kind(cut)
    with pytest.raises(delisted_kr.KindError):
        delisted_kr.parse_kind("<html>maintenance</html>")


def test_add_delisted_meta_appends_only_unknown_symbols():
    meta = pd.DataFrame({"name": ["Live"], "security_type": ["STOCK"], "product": ["stock"],
                         "is_common_share": [True], "delisted": [False], "source": ["toss"],
                         "market_board": ["KOSPI"], "isin": ["KR1"]},
                        index=pd.Index(["417310"], name="symbol"))
    out = delisted_kr.add_delisted_meta(meta, delisted_kr.parse_kind(KIND_PAGE))
    assert out.loc["417310", "source"] == "toss" and not out.loc["417310", "delisted"]
    assert bool(out.loc["053810", "delisted"]) and out.loc["053810", "product"] == "stock"
    assert out.loc["053810", "source"] == "kind" and bool(out.loc["053810", "is_common_share"])
    assert list(out.columns) == list(meta.columns)
