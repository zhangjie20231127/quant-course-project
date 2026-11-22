"""Build paper exhibit tables and charts directly from study CSV outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from acquire import OUT

D=OUT/'exhibits'; D.mkdir(exist_ok=True)
C=['#23466d','#ae5d35','#64737d','#4a785d','#815180']
L={'baseline':'Baseline','research':'Research','csi300':'CSI 300','bp':'B/P','small':'Small size','momentum':'Momentum','reversal':'Reversal','low_vol':'Low volatility'}
CN={'baseline':'基准模型','research':'研究模型','csi300':'沪深300','bp':'账面市值比','small':'小规模','momentum':'中期动量','reversal':'短期反转','low_vol':'低波动','intercept':'截距','residual':'个股残差','legacy_holdings':'未退出旧仓','fee_effect':'交易成本影响'}


def pct(x, digits=2):return f'{x:.{digits}%}'
def num(x, digits=2):return f'{x:.{digits}f}'
def csv(name):return pd.read_csv(OUT/f'{name}.csv')

def hac(x, lag=3):
    x=np.asarray(x,dtype=float); n=len(x); mu=x.mean(); z=x-mu
    v=(z@z)/n
    for j in range(1,min(lag,n-1)+1):v+=2*(1-j/(lag+1))*(z[j:]@z[:-j])/n
    se=np.sqrt(max(v,0)/n)
    return mu,se,mu-1.96*se,mu+1.96*se


def figure(key, size=(7.7,3.35)):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    f,a=plt.subplots(figsize=size); a.grid(axis='y',color='#dbe3ec',linewidth=.7,alpha=.85); a.set_axisbelow(True)
    return f,a


def save(f,key):
    f.tight_layout(); f.savefig(D/f'{key}.png',dpi=230,bbox_inches='tight',pad_inches=.12);plt.close(f)


def main():
    n=csv('daily').set_index('date'); n.index=pd.to_datetime(n.index)
    m={k:csv(f'{k}_monthly') for k in ('baseline','research')}; s=csv('summary').set_index(['segment','model']); a=csv('attribution').set_index('signal'); c=csv('cross'); cov=csv('coverage'); ic=csv('ic').set_index('signal'); gr=csv('groups'); cap=csv('capacity'); overlap=csv('amount_overlap'); holdings=csv('holdings'); meta=json.loads((OUT/'meta.json').read_text())
    assert len(a)==len(m['research'])==len(ic) and np.allclose(a.sum(axis=1),m['research'].net,atol=1e-10,rtol=0)
    tbl={}; text={**meta}
    for key,ylabel in [('nav','Net asset value (start = 1)'),('drawdown','Drawdown')]:
        f,ax=figure(key)
        for i,k in enumerate(('baseline','research','csi300')):
            y=n[k] if key=='nav' else n[k]/n[k].cummax().clip(lower=1)-1
            ax.plot(n.index,y,label=L[k],color=C[i],linewidth=1.7)
        ax.axvline(pd.Timestamp('2023-02-01'),color='#98a4b1',linewidth=.8,linestyle='--');ax.set_ylabel(ylabel);ax.legend(frameon=False,ncol=3,loc='best')
        if key=='drawdown':ax.yaxis.set_major_formatter(PercentFormatter(1))
        save(f,key)
    f,ax=figure('coverage');tt=pd.to_datetime(c.signal)
    for i,(key,label) in enumerate([('eligible','Eligibility'),('complete','Complete signals'),('outcomes','Observed endpoints')]): ax.plot(tt,c[key],label=label,color=C[i],linewidth=1.5)
    ax.set_ylabel('Stock count');ax.set_ylim(min(c.outcomes)-7,125);ax.legend(frameon=False,ncol=3);save(f,'coverage')
    f,ax=figure('groups'); mm=gr.groupby(['factor','group']).gross.mean()
    for i,k in enumerate(ic.columns): ax.plot(range(1,6),mm[k],marker='o',label=L[k],color=C[i])
    ax.axhline(0,color='#89929b',linewidth=.8);ax.set_xticks(range(1,6));ax.set_xlabel('Factor-score quintile (1 low; 5 high)');ax.set_ylabel('Mean monthly return');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,ncol=3);save(f,'groups')
    f,ax=figure('ic'); q=ic.set_axis(pd.to_datetime(ic.index)).rolling(12).mean()
    for i,k in enumerate(ic.columns):ax.plot(q.index,q[k],label=L[k],color=C[i],linewidth=1.45)
    ax.axhline(0,color='#89929b',linewidth=.8);ax.set_ylabel('12-period mean Rank IC');ax.legend(frameon=False,ncol=3);save(f,'ic')
    f,ax=figure('excess'); r=m['research']; b=m['baseline']; rolling=(1+r.net).rolling(12).apply(np.prod,raw=True)/(1+b.net).rolling(12).apply(np.prod,raw=True)-1
    ax.plot(pd.to_datetime(r.end),rolling,color=C[1],linewidth=1.8);ax.axhline(0,color='#89929b',linewidth=.8);ax.set_ylabel('12-period relative return');ax.yaxis.set_major_formatter(PercentFormatter(1));save(f,'excess')
    f,ax=figure('attr'); vals=a.mean(); col=[C[0] if v>=0 else C[1] for v in vals]; labels=[L[k] if k in L else {'intercept':'Intercept','residual':'Residual','legacy_holdings':'Legacy','fee_effect':'Costs'}[k] for k in vals.index]
    ax.barh(labels,vals,color=col);ax.axvline(0,color='#89929b',linewidth=.8);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Mean monthly contribution');save(f,'attr')
    f,ax=figure('turnover'); tt=pd.to_datetime(r.signal)
    ax.plot(tt,b.turnover,color=C[0],label=L['baseline'],linewidth=1.35);ax.plot(tt,r.turnover,color=C[1],label=L['research'],linewidth=1.35);ax.set_ylabel('Bought + sold / NAV');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);save(f,'turnover')
    sm=s.loc['full']; sb,sr=sm.loc['baseline'],sm.loc['research']; market=r.benchmark
    bench=dict(annual=sb.benchmark,vol=market.std()*np.sqrt(12),sharpe=market.mean()/market.std()*np.sqrt(12),terminal=n.csi300.iloc[-1],mdd=(n.csi300/n.csi300.cummax()-1).min())
    fields=[('annual','年化收益率',pct),('vol','年化波动率',pct),('sharpe','夏普比率（无风险利率为零）',num),('mdd','日频最大回撤',pct),('terminal','期末净值',lambda x:num(x,3)),('excess','相对沪深300年化收益',pct),('tracking','年化跟踪误差',pct),('ir','信息比率',num),('turnover','月均双边换手',pct)]
    tbl['main']=[['指标','基准模型','研究模型','沪深300']]+[[label,fmt(sb[k]),fmt(sr[k]),fmt(bench[k]) if k in bench else '—'] for k,label,fmt in fields]
    tbl['split']=[['信号期','期数','基准年化','研究年化','沪深300年化','研究最大回撤']]
    for seg,name in [('full','全部'),('early','前段'),('holdout','后段')]:
        bb,rr=s.loc[(seg,'baseline')],s.loc[(seg,'research')];tbl['split'].append([name,str(int(rr.months)),pct(bb.annual),pct(rr.annual),pct(rr.benchmark),pct(rr.mdd)])
    tbl['annual']=[['持有期末年份','期数','基准模型','研究模型','沪深300']]
    for yr, data in r.groupby(r.end.str[:4]):
        ix=data.index; tbl['annual'].append([yr,str(len(ix)),pct((1+b.loc[ix,'net']).prod()-1),pct((1+data.net).prod()-1),pct((1+data.benchmark).prod()-1)])
    tbl['ic']=[['因子','均值','为正月份','HAC 95% 区间','Q5−Q1月均']]
    for k in ic:
        v=ic[k];mu,se,lo,hi=hac(v);tbl['ic'].append([CN[k],num(mu,3),pct(v.gt(0).mean(),1),f'[{lo:.3f}, {hi:.3f}]',pct(mm[k][5]-mm[k][1])])
    tbl['abl']=[['研究规格','全期年化','后段年化','日频最大回撤','月均换手']]
    for key,name in [('research','完整五因子')]+[(f'without_{k}','移除'+CN[k]) for k in ('momentum','reversal','low_vol')]:
        x=sm.loc[key];tbl['abl'].append([name,pct(x.annual),pct(s.loc[('holdout',key),'annual']),pct(x.mdd),pct(x.turnover)])
    tbl['fees']=[['单边成本','基准年化','研究年化','研究期末净值','研究最大回撤']]
    for rate in (0,15,30,50):
        x=[sm.loc[k if rate==15 else f'{k}_{rate}bp'] for k in ('baseline','research')];tbl['fees'].append([f'{rate} bp',pct(x[0].annual),pct(x[1].annual),num(x[1].terminal,3),pct(x[1].mdd)])
    tbl['attr']=[['归因项','月均贡献','单月最低','单月最高']]+[[CN[k],pct(a[k].mean(),3),pct(a[k].min()),pct(a[k].max())] for k in a]
    tbl['coverage']=[['范围','记录数','基础行情及估值','次日有开盘价','次日有正成交额']]
    for label,data in [('全期信号含末端',cov),('2020—2022',cov[cov.date.lt('2023')]),('2023—2025',cov[cov.date.ge('2023')])]:
        tbl['coverage'].append([label,str(len(data)),str(int(data.all_sources.sum())),str(int(data.next_session_open.sum())),str(int(data.next_session_amount.sum()))])
    tbl['cross']=[['每期股票数','最小','中位数','最大']]+[[label,str(c[k].min()),num(c[k].median(),1),str(c[k].max())] for k,label in [('eligible','符合月末状态'),('complete','五因子齐全'),('outcomes','两端价格均观察到')]]
    tbl['risk']=[['执行与持仓指标','基准','研究'],['月均双边换手',pct(sb.turnover),pct(sr.turnover)],['月均未实现目标调整',pct(sb.unfilled),pct(sr.unfilled)],['平均交易后现金',pct(sb.cash),pct(sr.cash)],['月均直接费用 / 调仓前净值',pct(sb.fee,3),pct(sr.fee,3)],['调仓日沿用旧价的月份',str(int(b.stale.gt(0).sum())),str(int(r.stale.gt(0).sum()))]]
    cap_counts=cap.groupby('signal').agg(orders=('code','size'), observed=('adv','count')); observed=cap_counts[cap_counts.orders.eq(cap_counts.observed)].index
    bottleneck=cap[cap.signal.isin(observed)].groupby('signal').capacity_yuan.min()/1e8
    tbl['cap']=[['月末日成交额代理参与率','可测月数','最紧月（亿元）','P10（亿元）','中位月（亿元）']]
    for f0 in (0.5,1,2):
        x=bottleneck*f0;tbl['cap'].append([pct(.10*f0,0),str(len(x)),num(x.min()),num(x.quantile(.1)),num(x.median())])
    mu,se,lo,hi=hac(r.net-b.net)
    cross_check=(a.sum(axis=1).to_numpy()-r.net).max()
    text.update(n=str(len(r)), cov_n=str(len(cov)), cov_all=str(int(cov.all_sources.sum())), cov_open=str(int(cov.next_session_open.sum())), cov_amount=str(int(cov.next_session_amount.sum())),
      research_ann=pct(sr.annual), baseline_ann=pct(sb.annual), csi_ann=pct(sb.benchmark), research_dd=pct(sr.mdd), baseline_dd=pct(sb.mdd), csi_dd=pct(bench['mdd']), research_vol=pct(sr.vol), baseline_vol=pct(sb.vol),
      research_nav=num(sr.terminal,3), baseline_nav=num(sb.terminal,3), delta_ann=f'{(sr.annual-sb.annual)*100:+.2f} 个百分点', rel_ann=pct((sr.terminal/sb.terminal)**(12/len(r))-1),
      hac_mu=pct(mu),hac_ci=f'[{pct(lo)}, {pct(hi)}]', research_turn=pct(sr.turnover), base_turn=pct(sb.turnover), full_ic=', '.join(f'{CN[k]} {ic[k].mean():.3f}' for k in ic), overlap_n=str(len(overlap)), overlap_p95=f'{overlap.relative_abs_error.quantile(.95):.2e}', cap_share=f'{cap.adv.notna().sum()}/{len(cap)}', cap_months=str(len(bottleneck)), cap10=num(bottleneck.quantile(.10)), capmedian=num(bottleneck.median()),
      check=f'{np.abs(a.sum(axis=1).to_numpy()-r.net).max():.2e}', research_fee=pct(sr.fee,3), research_fee50=pct(sm.loc['research_50bp','annual']), product_net=pct((1+sr.annual)*np.exp(-.006)-1), n_early=str(int(s.loc[('early','research'),'months'])), n_late=str(int(s.loc[('holdout','research'),'months'])), late_research=pct(s.loc[('holdout','research'),'annual']),late_baseline=pct(s.loc[('holdout','baseline'),'annual']),early_research=pct(s.loc[('early','research'),'annual']),early_baseline=pct(s.loc[('early','baseline'),'annual']), errors=str(len(csv('source_errors'))), stockmed=num(c.complete.median(),1), min_elig=str(c.eligible.min()), max_elig=str(c.eligible.max()), holdings_min=str(holdings.groupby('signal').size().min()), holdings_max=str(holdings.groupby('signal').size().max()))
    (D/'facts.json').write_text(json.dumps(text,ensure_ascii=False,indent=2)+'\n')
    (D/'tables.json').write_text(json.dumps(tbl,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
