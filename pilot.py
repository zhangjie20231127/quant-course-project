"""Run a small process check, not a reportable strategy backtest."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = ["bp", "small"]
EXTRAS = ["momentum", "reversal", "low_vol"]
FACTORS = BASE + EXTRAS


def zscore(s):
    median = s.median()
    mad = (s - median).abs().median()
    if mad > 0:
        s = s.clip(median - 4.4478 * mad, median + 4.4478 * mad)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def exposures(adjusted, valuation, month_ends):
    prices = adjusted.set_index("date")["close"].sort_index().astype(float)
    prices = prices.where(prices.gt(0))
    month_prices = prices.reindex(month_ends)
    out = pd.DataFrame({
        "momentum": month_prices.shift(1) / month_prices.shift(12) - 1,
        "reversal": -(prices / prices.shift(20) - 1),
        "low_vol": -prices.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252),
    }).reindex(month_ends)
    value = valuation.set_index("date").reindex(month_ends)
    out["bp"] = 1 / value["pb"].where(value["pb"].gt(0))
    out["small"] = -np.log(value["market_cap"].where(value["market_cap"].gt(0)))
    return out.rename_axis("date").reset_index()


def target_weights(frame, factors, holdings):
    ranked = (frame.assign(score=frame[factors].mean(axis=1))
              .sort_values(["score", "code"], ascending=[False, True]).head(holdings))
    return pd.Series(1 / len(ranked), index=ranked["code"])


def trade(weights, before, per_side_fee):
    delta = weights.subtract(before, fill_value=0)
    traded = float(delta.abs().sum())
    return traded, traded * per_side_fee


def fit_attribution(frame, weights):
    y = frame.set_index("code")["next_ret"]
    x = frame.set_index("code")[FACTORS].assign(intercept=1.0)
    coef, _, rank, _ = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)
    if rank != len(x.columns):
        raise ValueError("attribution regression is rank deficient")
    residual = y - x @ coef
    contributions = x.loc[weights.index].mul(weights, axis=0).sum() * coef
    contributions["residual"] = float(residual.loc[weights.index] @ weights)
    if not np.isclose(contributions.sum(), y.loc[weights.index] @ weights, atol=1e-12):
        raise ValueError("attribution does not reconcile")
    return contributions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/cache/2019-01-01_2026-11-22"))
    p.add_argument("--panel", type=Path, default=Path("data/course_panel.csv"))
    p.add_argument("--holdings", type=int, default=3)
    opt = p.parse_args()
    panel = pd.read_csv(opt.panel, dtype={"code": str, "date": str})
    dates = sorted(panel["date"].unique())
    frames = []
    for price_file in sorted(opt.cache.glob("*_adjusted.csv")):
        code = price_file.name.removesuffix("_adjusted.csv")
        value_file = opt.cache / f"{code}_valuation.csv"
        if value_file.exists():
            frames.append(exposures(pd.read_csv(price_file), pd.read_csv(value_file), dates)
                          .assign(code=code))
    if not frames:
        raise SystemExit("Run the audit to cache market and valuation inputs first.")
    matched = panel.merge(pd.concat(frames), on=["date", "code"], validate="one_to_one")
    matched = matched.loc[matched["tradable"].eq(1) & matched["is_st"].eq(0)
                          & matched["listed_days"].ge(120)].dropna(subset=FACTORS)
    min_size = max(2 * opt.holdings, len(FACTORS) + 2)
    sizes = matched.groupby("date").size()
    usable = [d for d in dates if sizes.get(d, 0) >= min_size]
    periods, ics, attribution, holdings_rows = [], [], [], []
    previous = {m: pd.Series(dtype=float) for m in ("baseline", "research")}
    last_position = None
    date_positions = {date: i for i, date in enumerate(dates)}
    for date in usable:
        cross = matched.loc[matched["date"].eq(date)].copy()
        cross[FACTORS] = cross[FACTORS].apply(zscore)
        known_outcomes = cross.dropna(subset=["next_ret"])
        rank_ic = known_outcomes[FACTORS].corrwith(known_outcomes["next_ret"], method="spearman")
        ics.append(rank_ic.rename(date))
        pos = date_positions[date]
        w = {m: target_weights(cross, f, opt.holdings)
             for m, f in (("baseline", BASE), ("research", FACTORS))}
        y = cross.set_index("code")["next_ret"]
        benchmark = cross["benchmark_ret"].dropna().unique()
        if len(benchmark) != 1 or any(y.reindex(weights.index).isna().any() for weights in w.values()):
            continue
        if last_position is not None and pos != last_position + 1:
            raise SystemExit(f"Discontinuous marked outcomes at {date}; do not compound these months.")
        row = {"signal_date": date, "return_end": dates[pos + 1], "csi300": benchmark[0]}
        for m, weights in w.items():
            held = y.loc[weights.index]
            traded, cost = trade(weights, previous[m], 0.0015)
            gross = float(held @ weights)
            row[f"{m}_gross"] = gross
            row[f"{m}_turnover"] = traded
            row[f"{m}_cost"] = cost
            row[f"{m}_net"] = (1 + gross) * (1 - cost) - 1
            previous[m] = weights * (1 + held) / (1 + gross)
            holdings_rows.extend((date, m, code, weight) for code, weight in weights.items())
        periods.append(row)
        last_position = pos
        try:
            part = fit_attribution(known_outcomes, w["research"])
            part["trading_cost_effect"] = row["research_net"] - row["research_gross"]
            attribution.append(part.rename(date))
        except ValueError as exc:
            if str(exc) != "attribution regression is rank deficient":
                raise
            print(f"{date}: {exc}")
    if not periods:
        raise SystemExit("No complete marked rounds in the download sample.")
    output = Path("results/pilot")
    output.mkdir(parents=True, exist_ok=True)
    returns = pd.DataFrame(periods)
    returns.to_csv(output / "returns.csv", index=False)
    returns.set_index("return_end")[["baseline_net", "research_net", "csi300"]].add(1).cumprod().to_csv(output / "nav.csv")
    pd.DataFrame(ics).rename_axis("signal_date").to_csv(output / "rank_ic.csv")
    pd.DataFrame(attribution).rename_axis("signal_date").to_csv(output / "attribution.csv")
    pd.DataFrame(holdings_rows, columns=["signal_date", "model", "code", "weight"]).to_csv(output / "holdings.csv", index=False)
    print(f"Process check: {len(returns)} observed rounds; {len(attribution)} reconciled regressions")


if __name__ == "__main__":
    main()
