"""Fetch the fixed, historically selected research cohort; reuse the audit cache."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import pandas as pd
from audit import audit, bars, cached

START, END, FORMATION, FINAL = '2019-01-01', '2026-11-22', '2020-01-23', '2025-09-30'
PANEL = Path('data/course_panel.csv')
CACHE = PANEL.parent / 'cache' / f'{START}_{END}'
OUT = Path('results/study')


def source():
    return pd.read_csv(PANEL, dtype={'date':str,'code':str})


def eligible(p):
    return p.tradable.eq(1) & p.is_st.eq(0) & p.listed_days.ge(120) & p.market_cap.gt(0)


def cohort(p):
    s = p[p.date.eq(FORMATION) & eligible(p)]
    assert len(s) >= 120
    return s.sort_values(['market_cap','code'], ascending=[False,True]).head(120).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    p = source()
    members = cohort(p)
    OUT.mkdir(parents=True, exist_ok=True)
    members[['date','code','market_cap']].to_csv(OUT/'members.csv', index=False)
    bench = cached(CACHE/'benchmark.csv', lambda: bars('sh.000300',START,END,0))
    cal = sorted(bench.date)
    nxt = dict(zip(cal,cal[1:]))
    p = p[p.code.isin(members.code) & p.date.between(FORMATION, FINAL)]
    groups = p[eligible(p)].groupby('code').date.agg(list).to_dict()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        answers = []
        for i, item in enumerate(pool.map(lambda code: audit(code, groups[code], CACHE, START, END, nxt), members.code), 1):
            answers.append(item)
            if i % 20 == 0:
                print(f'{i}/{len(members)}', flush=True)
    d = pd.concat([df for df,_ in answers], ignore_index=True)
    d['all_sources'] = d[['raw','adjusted','valuation']].all(axis=1)
    d.to_csv(OUT/'coverage.csv', index=False)
    pd.DataFrame([row for _,err in answers for row in err], columns=['code','source','issue']).to_csv(OUT/'source_errors.csv',index=False)
    print(f'{len(members)} codes; {d.all_sources.sum()}/{len(d)} match all sources')

if __name__ == '__main__':
    main()
