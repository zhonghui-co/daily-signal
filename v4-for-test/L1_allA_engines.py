#!/usr/bin/env python3
"""
v4-for-test 第1层 — 全A N型 vs 价格法状态机 头对头 (000985 九年同窗)
判据(预注册于 README.md):
  1. 九年净收益 >= 状态机 - 3pp
  2. MaxDD <= 状态机 + 5pp
  3. 三个熊年(2018/2022/2023)每年收益<=-10%的次数 <= 状态机
W 稳健性: W∈{2,3,4,5} >= 3/4 通过; 只在 W=3 成立的结论作废
执行: 信号日收盘定夺 → 次日收盘成交; 单边成本 0.1%
"""
import json, os, sqlite3, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import causal_states, state_machine_56_3, state_machine_v1, simulate, yearly_table

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
COST = 0.001
BEAR_YEARS = ('2018', '2022', '2023')

def load_000985():
    df = pd.read_csv(os.path.join(DATA, '000985_csi.csv'), dtype={'trade_date': str})
    df = df.sort_values('trade_date')
    return list(df['trade_date']), df['close'].astype(float).values

def judge(n_ret, n_dd, sm_ret, sm_dd, n_yearly, sm_yearly, W, verbose=True):
    ok1 = n_ret >= sm_ret - 0.03
    ok2 = n_dd >= sm_dd - 0.05  # MaxDD为负数, 更差=更负 → 不低于基准-5pp
    nb = sum(1 for y in BEAR_YEARS if n_yearly.get(y, 0) <= -0.10)
    sb = sum(1 for y in BEAR_YEARS if sm_yearly.get(y, 0) <= -0.10)
    ok3 = nb <= sb
    ok = ok1 and ok2 and ok3
    if verbose:
        print(f'  W={W}: 净收益 {n_ret*100:+.1f}% vs 状态机 {sm_ret*100:+.1f}% (需>={ (sm_ret-0.03)*100:.1f}%) {"✅" if ok1 else "❌"}')
        print(f'         MaxDD {n_dd*100:.1f}% vs {sm_dd*100:.1f}% (需<={(sm_dd+0.05)*100:.1f}%) {"✅" if ok2 else "❌"}')
        print(f'         熊年≤-10%次数: {nb} vs {sb} (需<={sb}) {"✅" if ok3 else "❌"}')
        print(f'         → {"✅ 通过" if ok else "❌ 未通过"}')
    return ok

def main():
    dates, closes = load_000985()
    n = len(closes)
    rets = np.zeros(n)
    rets[1:] = closes[1:] / closes[:-1] - 1
    rets_pct = rets * 100
    print(f'🦉 第1层: 000985 九年同窗 {dates[0]}~{dates[-1]} ({n}天)  单边成本{COST*100:.1f}%')

    # 基准臂1: 状态机 v1 口径 (3.4/-2.0) — MEMORY记录中九年验证过的那套
    # 基准臂2: 状态机 5.6/-3 — 外部评审指定
    baselines = {}
    for name, fn in (('v1(3.4/-2.0)', state_machine_v1), ('5.6/-3', state_machine_56_3)):
        sm = fn(rets_pct)
        pos_sm = (sm == 1).astype(float)
        sm_ret, sm_dd, eq_sm = simulate(pos_sm, rets, COST)
        sm_yearly = yearly_table(eq_sm, dates)
        baselines[name] = (sm_ret, sm_dd, sm_yearly, eq_sm, sm)
        print(f'基准: 状态机{name}  九年净收益 {sm_ret*100:+.1f}%  MaxDD {sm_dd*100:.1f}%  多头{(sm==1).sum()}天')

    # 对照: 买入持有
    bh_ret, bh_dd, eq_bh = simulate(np.ones(n), rets, 0.0)
    print(f'对照: 买入持有      九年净收益 {bh_ret*100:+.1f}%  MaxDD {bh_dd*100:.1f}%')

    # 挑战者: N型 W∈{2,3,4,5} (vs 两个基准各判一次)
    for bname, (sm_ret, sm_dd, sm_yearly, eq_sm, sm) in baselines.items():
        print(f'\n===== 挑战者 vs 状态机{bname} =====')
        results = {}
        for W in (2, 3, 4, 5):
            st = causal_states(closes, W, n - 1)
            pos_n = (st == 1).astype(float)
            n_ret, n_dd, eq_n = simulate(pos_n, rets, COST)
            n_yearly = yearly_table(eq_n, dates)
            ok = judge(n_ret, n_dd, sm_ret, sm_dd, n_yearly, sm_yearly, W)
            results[W] = (ok, n_ret, n_dd)
        n_ok = sum(1 for W, (ok, _, _) in results.items() if ok)
        print(f'W网格稳健性: {n_ok}/4 通过 → {"✅ 稳健" if n_ok >= 3 else "❌ 不稳(参数孤点)"}')
    print(f'\n结论: {"N型可以挑战状态机" if n_ok >= 3 else "N型被淘汰, 全A层保留状态机"}')

if __name__ == '__main__':
    main()
