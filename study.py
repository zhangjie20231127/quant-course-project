"""Reproduce the paper's cohort study, execution simulation, and factor tests."""
import json
import numpy as np
import pandas as pd
from acquire import CACHE, OUT, FORMATION, FINAL, source, cohort, eligible
from pilot import FACTORS, BASE, EXTRAS, exposures, zscore

TAU = 0.0015
HOLD = 20
CUT = '2023-01-01'
VARS = {'baseline': BASE, 'research': FACTORS, **{f'without_{f}':[g for g in FACTORS if g!=f] for f in EXTRAS}}


def load():
    p = source()
    codes = sorted(cohort(p).code)
    bench = pd.read_csv(CACHE/'benchmark.csv').set_index('date').sort_index().close.astype(float)
    calendar = bench.index
    nx = dict(zip(calendar, calendar[1:]))
    dates = sorted(p.date.unique())
    signals = [d for d in dates if FORMATION <= d <= FINAL]
    assert all(d in nx for d in signals)
    marked = pd.DataFrame(index=calendar, columns=codes, dtype=float)
    tradeable = marked.copy().fillna(0).astype(bool)
    # Course amount is month-end day turnover in thousands of yuan, verified in overlap.
    amount_panel = p[p.code.isin(codes)].pivot(index="date",columns="code",values="amount").sort_index()
    adv = (1000 * amount_panel.where(amount_panel.gt(0))).rolling(6,min_periods=3).median()
    overlaps=[]
    frames = []
    p = p[eligible(p) & p.code.isin(codes) & p.date.isin(signals)]
    def read(code, part):
        path = CACHE/f'{code}_{part}.csv'
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    for c in codes:
        a, r, v = [read(c, k) for k in ('adjusted','raw','valuation')]
        if not a.empty:
            a = a.sort_values('date')
            q = a.set_index('date').close.astype(float).reindex(calendar)
            marked[c] = q.where(q.gt(0))
            if not v.empty:
                frames.append(exposures(a,v,dates).assign(code=c))
        if not r.empty and not a.empty:
            rr = r.set_index('date').reindex(calendar)
            amount = pd.to_numeric(rr['amount'], errors='coerce') if 'amount' in rr else pd.Series(np.nan, index=calendar)
            actual = amount.reindex(amount_panel.index)
            implied = amount_panel[c]*1000
            ok = actual.gt(0) & implied.gt(0) & implied.index.to_series().between(FORMATION,FINAL)
            overlaps.extend((actual[ok]/implied[ok]-1).abs().tolist())
            raw_active = rr.high.gt(rr.low) & rr.close.gt(0)
            if 'amount' in rr:
                raw_active &= amount.gt(0)
            tradeable[c] = raw_active & marked[c].notna()
    assert len(overlaps)>100 and np.median(overlaps) < 0.001, "month-end amount unit validation failed"
    pd.DataFrame({"relative_abs_error":overlaps}).to_csv(OUT/"amount_overlap.csv",index=False)
    matched = p.merge(pd.concat(frames),on=['date','code'],validate='one_to_one').dropna(subset=FACTORS)
    out = {}
    for d,g in matched.groupby('date'):
        z = g.set_index('code')[FACTORS].apply(zscore)
        assert len(z) >= HOLD * 2 and np.isfinite(z).all().all()
        out[d] = z
    assert set(out) == set(signals)
    return codes, bench, nx, signals, marked, tradeable, adv, out, p


def rebalance(old, cash, score, allowed, fee):
    pre = old.sum()+cash
    desired = pd.Series(0.0,index=old.index)
    pick = score.sort_values(ascending=False,kind='stable').head(HOLD).index
    desired.loc[pick] = pre/HOLD
    target = desired-old
    delta = target.where(allowed,0.)
    sold = -delta.clip(upper=0)
    want = delta.clip(lower=0)
    cash += sold.sum()*(1-fee)
    bought = want * min(1.,cash/(want.sum()*(1+fee))) if want.sum()>0 else want
    actual = bought-sold
    cash -= bought.sum()*(1+fee)
    if abs(cash) < 1e-12:
        cash = 0.
    new = old+actual
    cost = actual.abs().sum()*fee
    assert cash>=0 and new.min()>=-1e-12 and np.isclose(new.sum()+cash,pre-cost,atol=1e-12)
    return new,cash,actual,target,cost


def simulate(model, factors, fee, cfg, attribution=False):
    codes, bench, nx, signals, marked, tradeable, adv, xs, _ = cfg
    px = marked.ffill()
    sessions = [nx[d] for d in signals]
    cal = bench.loc[sessions[0]:sessions[-1]].index
    active = dict(zip(sessions[:-1],signals[:-1]))
    stock = pd.Series(0.,index=codes)
    cash = 1.
    nav=[]; periods=[]; parts=[]; holdings=[]; caps=[]
    state = None
    for day in cal:
        price=px.loc[day]
        assert not (stock.gt(0) & price.isna()).any()
        pre_stock=stock.mul(price).fillna(0)
        pre=pre_stock.sum()+cash
        if day in sessions and state:
            d,begin,post,w,turn,cost,failed,stale = state
            net=pre/begin-1
            gross=pre/post-1
            periods.append((d,nx[d],day,net,gross,turn,cost/begin,failed,stale,1-w.sum()))
            if attribution:
                r = (price / px.loc[nx[d]] - 1).replace([np.inf,-np.inf],np.nan)
                x = xs[d].assign(intercept=1.)
                observed = marked.loc[nx[d]].notna() & marked.loc[day].notna()
                fit = x.join(r.where(observed).rename('return')).dropna()
                beta,_,rank,_ = np.linalg.lstsq(fit[x.columns],fit['return'],rcond=None)
                assert rank==len(x.columns)
                ww=w.reindex(x.index)
                residual=r.reindex(x.index)-x@beta
                c = (x.mul(ww,axis=0).sum()*beta).to_dict()
                c['residual']=float(residual.fillna(0)@ww)
                outside=w.index.difference(x.index)
                c['legacy_holdings']=float(r.reindex(outside).fillna(0)@w.reindex(outside))
                c['fee_effect']=net-gross
                assert abs(sum(c.values())-net)<1e-10
                parts.append(dict(signal=d,**c))
        if day in active:
            d=active[day]
            score=xs[d][factors].mean(axis=1)
            new,cash,actual,target,cost=rebalance(pre_stock,cash,score,tradeable.loc[day],fee)
            traded=actual.abs(); turn=traded.sum()/pre
            failed=float((target-actual).abs().sum()/pre)
            stale=float(new.where(marked.loc[day].isna(),0.).sum()/pre)
            stock=new.div(price.where(price.gt(0))).fillna(0.)
            post=new.sum()+cash
            w=new/post
            state=(d,pre,post,w,turn,cost,failed,stale)
            if attribution:
                for c in w[w>0].index:
                    holdings.append((d,day,c,w[c],not tradeable.loc[day,c]))
                for c in traded[traded>1e-12].index:
                    crowd=adv.loc[d,c]
                    caps.append((d,c,traded[c]/pre,crowd,0.10*crowd/(traded[c]/pre)))
            nav.append((day,post))
        else:
            nav.append((day,pre))
    daily=pd.Series(dict(nav),name=model)
    monthly=pd.DataFrame(periods,columns=['signal','begin','end','net','gross','turnover','fee','unfilled','stale','cash'])
    assert np.isclose((1+monthly.net).prod(),daily.iloc[-1],atol=1e-10)
    return daily,monthly,pd.DataFrame(parts),pd.DataFrame(holdings,columns=['signal','execution','code','weight','blocked']),pd.DataFrame(caps,columns=['signal','code','trade_weight','adv','capacity_yuan'])


def diagnostics(cfg):
    _,_,nx,signals,marked,_,_,xs,p = cfg
    ic=[]; groups=[]; cross=[]
    for d,nd in zip(signals,signals[1:]):
        x=xs[d]; y=marked.loc[nx[nd]]/marked.loc[nx[d]]-1
        xy=x.join(y.rename('outcome')).dropna()
        ic.append(xy[FACTORS].corrwith(xy.outcome,method='spearman').rename(d))
        for f in FACTORS:
            q=pd.qcut(xy[f].rank(method='first'),5,labels=False)
            groups.extend((d,f,i+1,r) for i,r in xy.groupby(q).outcome.mean().items())
        cross.append((d,int(p.date.eq(d).sum()),len(x),len(xy)))
    pd.DataFrame(ic).rename_axis('signal').to_csv(OUT/'ic.csv')
    pd.DataFrame(groups,columns=['signal','factor','group','gross']).to_csv(OUT/'groups.csv',index=False)
    pd.DataFrame(cross,columns=['signal','eligible','complete','outcomes']).to_csv(OUT/'cross.csv',index=False)


def stats(month,daily):
    n=len(month); m=month.net; b=month.benchmark; e=m-b
    wealth=np.r_[1.,daily.values]
    dd=wealth/np.maximum.accumulate(wealth)-1
    return dict(months=n, annual=(1+m).prod()**(12/n)-1, benchmark=(1+b).prod()**(12/n)-1,
        vol=m.std()*np.sqrt(12), sharpe=m.mean()/m.std()*np.sqrt(12), mdd=dd.min(),
        excess=((1+m).prod()/(1+b).prod())**(12/n)-1, tracking=e.std()*np.sqrt(12),
        ir=e.mean()/e.std()*np.sqrt(12), turnover=month.turnover.mean(), terminal=(1+m).prod(),
        unfilled=month.unfilled.mean(), cash=month.cash.mean(), fee=month.fee.mean())


def main():
    cfg=load(); _, bench,nx,signals,*_=cfg
    diagnostics(cfg)
    navs={}; mons={}; summary=[]
    for name,factors,fee in [(n,f,TAU) for n,f in VARS.items()] + [(f'{m}_{bps}bp',VARS[m],bps/10000) for m in ('baseline','research') for bps in (0,30,50)]:
        daily,m,a,h,c=simulate(name,factors,fee,cfg,name=='research')
        m['benchmark']=[bench[end]/bench[begin]-1 for begin,end in zip(m.begin,m.end)]
        navs[name]=daily; mons[name]=m
        if name in ('baseline','research'):
            m.to_csv(OUT/f'{name}_monthly.csv',index=False)
        if name=='research':
            a.to_csv(OUT/'attribution.csv',index=False); h.to_csv(OUT/'holdings.csv',index=False); c.to_csv(OUT/'capacity.csv',index=False)
        for label,selected in [('full',m),('early',m[m.signal.lt(CUT)]),('holdout',m[m.signal.ge(CUT)])]:
            rebased=daily.loc[selected.begin.iloc[0]:selected.end.iloc[-1]].copy()
            first_pre=rebased.iloc[0]/(1-selected.fee.iloc[0])
            rebased.iloc[-1]=first_pre*(1+selected.net).prod()
            summary.append(dict(model=name,segment=label,**stats(selected,rebased/first_pre)))
    dates=navs['research'].index
    navs['csi300']=bench.reindex(dates)/bench.loc[dates[0]]
    pd.DataFrame(navs).rename_axis('date').to_csv(OUT/'daily.csv')
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    meta={'first_signal': signals[0], 'first_execution': nx[signals[0]], 'last_signal': signals[-2], 'last_mark': nx[signals[-1]],'holdout_cut':CUT}
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(pd.DataFrame(summary).query("model in ['research','baseline']").round(4).to_string(index=False))

if __name__=='__main__':
    main()
