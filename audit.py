"""Audit what can be joined on a historical stock and exchange date."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pandas as pd
import requests

KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
VALUE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
TENCENT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
BAR_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount",
               "amplitude", "pct_change", "change", "turnover_rate"]


def get_json(endpoint, **params):
    for attempt in range(3):
        try:
            response = requests.get(endpoint, params=params, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(1.5 ** attempt)


def bars(code, start, end, adjust):
    market, symbol = code.split(".")
    payload = get_json(
        KLINE, secid=f"{int(market == 'sh')}.{symbol}", klt=101, fqt=adjust,
        beg=start.replace("-", ""), end=end.replace("-", ""),
        fields1="f1,f2,f3,f4,f5,f6", fields2=",".join(f"f{i}" for i in range(51, 62)),
    )
    rows = (payload.get("data") or {}).get("klines") or []
    return pd.DataFrame([row.split(",") for row in rows], columns=BAR_COLUMNS).assign(provider="eastmoney")


def tencent_prices(code, start, end, adjust):
    # Use a single provider for the entire history of each price series.
    symbol = code.replace(".", "")
    series = "hfqday" if adjust == 2 else "day"
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    while first <= last:
        until = min(first + pd.DateOffset(years=2) - pd.Timedelta(days=1), last)
        query = f"{symbol},day,{first.date()},{until.date()},640,{'hfq' if adjust == 2 else ''}"
        payload = get_json(TENCENT, param=query)
        if payload.get("code") != 0:
            raise ValueError(f"Tencent price request failed: {symbol}, {first.date()}")
        quotes = ((payload.get("data") or {}).get(symbol) or {}).get(series) or []
        rows.extend(row[:5] for row in quotes if str(first.date()) <= row[0] <= str(until.date()))
        first = until + pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["date", "open", "close", "high", "low"]).assign(provider="tencent")


def prices(code, start, end, adjust):
    try:
        frame = bars(code, start, end, adjust)
        if not frame.empty:
            return frame
    except requests.RequestException:
        pass
    return tencent_prices(code, start, end, adjust)


def valuations(code, start, end):
    columns = {"TRADE_DATE": "date", "PB_MRQ": "pb", "TOTAL_MARKET_CAP": "market_cap"}
    symbol = code.split(".")[1]
    frames = []
    page = 1
    while True:
        payload = get_json(
            VALUE, reportName="RPT_VALUEANALYSIS_DET", columns=",".join(columns),
            filter=f'(SECURITY_CODE="{symbol}")(TRADE_DATE>=\'{start}\')(TRADE_DATE<=\'{end}\')',
            pageNumber=page, pageSize=500, sortColumns="TRADE_DATE", sortTypes=1,
        )
        result = payload.get("result") or {}
        rows = result.get("data") or []
        if rows:
            frames.append(pd.DataFrame.from_records(rows, columns=list(columns)))
        if page >= result.get("pages", 0) or not rows:
            break
        page += 1
    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=list(columns)))
    frame = frame.rename(columns=columns)
    frame["date"] = frame["date"].astype("string").str.slice(0, 10)
    return frame


def cached(path, loader):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, dtype={"date": str})
    frame = loader()
    if frame["date"].duplicated().any():
        raise ValueError(f"duplicate exchange dates: {path}")
    if not frame.empty:
        pending = path.with_suffix(".tmp")
        frame.to_csv(pending, index=False)
        pending.replace(path)
    return frame


def sample_codes(panel, per_year):
    picked = {"sh.600068"}
    for _, year in panel.groupby(panel["date"].str[:4]):
        eligible = (year[year["date"].eq(year["date"].max())]
                    .sort_values(["market_cap", "code"]))
        if len(eligible) > 0:
            for i in range(per_year):
                picked.add(eligible.iloc[round(i * (len(eligible) - 1) / max(1, per_year - 1))]["code"])
    return sorted(picked)


def audit(code, dates, cache, start, end, next_sessions):
    loads = {"raw": lambda: prices(code, start, end, 0),
             "adjusted": lambda: prices(code, start, end, 2),
             "valuation": lambda: valuations(code, start, end)}
    checked = pd.DataFrame({"date": sorted(dates), "code": code})
    checked["next_session"] = checked["date"].map(next_sessions)
    checked["next_session_open"] = False
    checked["next_session_amount"] = False
    errors = []
    for label, loader in loads.items():
        try:
            frame = cached(cache / f"{code}_{label}.csv", loader)
            valid = frame["pb"].gt(0) & frame["market_cap"].gt(0) if label == "valuation" else (
                pd.to_numeric(frame["close"], errors="coerce").gt(0))
            checked[label] = checked["date"].isin(frame.loc[valid, "date"])
            if label == "raw":
                valid_open = pd.to_numeric(frame["open"], errors="coerce").gt(0)
                checked["next_session_open"] = checked["next_session"].isin(frame.loc[valid_open, "date"])
                if "amount" in frame:
                    liquid = pd.to_numeric(frame["amount"], errors="coerce").gt(0)
                    checked["positive_amount"] = checked["date"].isin(frame.loc[liquid, "date"])
                    checked["next_session_amount"] = checked["next_session"].isin(frame.loc[liquid, "date"])
                else:
                    checked["positive_amount"] = False
            if label != "valuation":
                checked[f"{label}_provider"] = frame["provider"].iloc[0] if "provider" in frame and len(frame) else (
                    "eastmoney" if len(frame) else None)
            if frame.empty:
                errors.append((code, label, "no records"))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            checked[label] = False
            if label == "raw":
                checked["positive_amount"] = False
            errors.append((code, label, str(exc)))
    return checked, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/course_panel.csv"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-11-22")
    parser.add_argument("--per-year", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--universe", choices=["sample", "panel"], default="sample")
    parser.add_argument("--batch", type=int, help="Zero-based batch index in the sorted panel universe")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.batch is not None and (args.universe != "panel" or args.batch < 0 or args.batch_size < 1):
        parser.error("--batch requires the panel universe, a nonnegative index, and a positive batch size")
    panel = pd.read_csv(args.panel, dtype={"date": str, "code": str})
    eligible = panel.loc[
        panel["date"].between(args.start, args.end) & panel["tradable"].eq(1)
        & panel["is_st"].eq(0) & panel["listed_days"].ge(120) & panel["market_cap"].gt(0)
    ]
    if eligible.empty:
        raise SystemExit("No eligible rows in the requested date range")
    cache = args.panel.parent / "cache" / f"{args.start}_{args.end}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        benchmark = cached(cache / "benchmark.csv", lambda: bars("sh.000300", args.start, args.end, 0))
    except (requests.RequestException, ValueError) as exc:
        print(f"Benchmark request failed; continuing stock coverage audit: {exc}")
        benchmark = pd.DataFrame(columns=["date"])
    calendar = sorted(benchmark["date"].dropna().unique())
    next_sessions = dict(zip(calendar, calendar[1:]))
    dates_by_code = eligible.groupby("code")["date"].agg(list).to_dict()
    if args.universe == "sample":
        # Audit the historical exit even if it does not appear in the course panel.
        dates_by_code.setdefault("sh.600068", ["2021-08-31"])
        codes = sample_codes(eligible, args.per_year)
    else:
        codes = sorted(dates_by_code)
        if args.batch is not None:
            offset = args.batch * args.batch_size
            codes = codes[offset:offset + args.batch_size]
    if not codes:
        raise SystemExit("No codes in the requested batch")
    out = args.out
    if args.universe == "panel":
        out = out / f"panel_{args.start}_{args.end}"
        if args.batch is not None:
            out = out / f"batch_{args.batch_size}_{args.batch:03d}"
    print(f"Auditing {len(codes)} distinct codes; writing coverage to {out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(
            lambda code: audit(code, dates_by_code[code], cache, args.start, args.end, next_sessions), codes,
        ))
    details = pd.concat([rows for rows, _ in completed], ignore_index=True)
    details["all_sources"] = details[["raw", "adjusted", "valuation"]].all(axis=1)
    details["calendar_matched"] = details["next_session"].notna()
    result = details.groupby("date").agg(
        rows=("code", "size"), raw=("raw", "sum"), adjusted=("adjusted", "sum"),
        valuation=("valuation", "sum"), all_sources=("all_sources", "sum"),
        positive_amount=("positive_amount", "sum"),
        calendar_matched=("calendar_matched", "sum"),
        next_session_open=("next_session_open", "sum"),
        next_session_amount=("next_session_amount", "sum"),
    )
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "coverage.csv")
    details.to_csv(out / "joined_dates.csv", index=False)
    pd.DataFrame([error for _, errors in completed for error in errors],
                 columns=["code", "source", "issue"]).to_csv(out / "source_errors.csv", index=False)
    print(f"{len(codes)} codes; {len(result)} dates; {int(details['all_sources'].sum())}/{len(details)} stock-months matched")
    print(f"Benchmark: {len(benchmark)} daily records, {benchmark.date.min()} to {benchmark.date.max()}")


if __name__ == "__main__":
    main()
