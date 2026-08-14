#!/usr/bin/env python3
"""
v4-for-test 第2层 — 板块 N型转涨 事件研究 (申万 L1 31指数 九年)
判据(预注册于 README.md):
  1. 20日超额中位数 > 0 且 > 随机日期臂中位数 + 1pp
  2. 5/60日方向一致(同正)
  3. W∈{2,3,4,5} 网格 >= 3/4 稳健
  4. whipsaw: 每板块年均状态切换 <= 10 次
设计: 剔beta(减000985) + 对照臂(随机日期 + 全体板块同日) + 切换频率先数后测
执行: 事件日t收盘定夺 → t+1收盘入场 → 前向 5/20/60 日超额
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import causal_states

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
HORIZONS = (5, 20, 60)
WHIPSAW_LIMIT = 10

def load():
    allA = pd.read_csv(os.path.join(DATA, '000985_csi.csv'), dtype={'trade_date': str}).sort_values('trade_date')
    sw = pd.read_csv(os.path.join(DATA, 'sw_l1_daily.csv'), dtype={'trade_date': str})
    # 对齐交易日
    dates = allA['trade_date'].tolist()
    sw = sw[sw['trade_date'].isin(set(dates))]
    P = sw.pivot_table(index='trade_date', columns='ts_code', values='close', aggfunc='first').reindex(dates)
    A = allA.set_index('trade_date')['close'].reindex(dates)
    return dates, A.values, P, list(P.columns)

def fwd_excess(sector_closes, allA_closes, t, k):
    """t+1收盘入场, 前向k日超额收益"""
    if t + 1 + k >= len(allA_closes):
        return np.nan
    if np.isnan(sector_closes[t + 1]) or np.isnan(sector_closes[t + 1 + k]):
        return np.nan
    s = sector_closes[t + 1 + k] / sector_closes[t + 1] - 1
    a = allA_closes[t + 1 + k] / allA_closes[t + 1] - 1
    return s - a

def main():
    dates, A, P, codes = load()
    n = len(dates)
    print(f'🦉 第2层: 申万L1 {len(codes)}板块 九年 {dates[0]}~{dates[-1]} ({n}天)')

    rng = np.random.default_rng(42)
    for W in (2, 3, 4, 5):
        print(f'\n===== W={W} =====')
        # 各板块因果状态
        states = np.zeros((len(codes), n), dtype=np.int8)
        switches = []
        for ci, code in enumerate(codes):
            st = causal_states(P[code].values, W, n - 1)
            states[ci] = st
            switches.append((np.diff(st) != 0).sum())
        switches_per_year = np.array(switches) / (n / 250)
        print(f'  whipsaw: 年均切换 中位{np.median(switches_per_year):.1f} 最大{np.max(switches_per_year):.1f} (限≤{WHIPSAW_LIMIT})')

        # 事件: 转涨日 (0/-1 → 1)
        ev_mask = np.zeros_like(states, dtype=bool)
        for ci in range(len(codes)):
            s = states[ci]
            for t in range(1, n):
                if s[t] == 1 and s[t - 1] != 1:
                    ev_mask[ci, t] = True
        ev_ci, ev_t = np.where(ev_mask)
        n_ev = len(ev_ci)
        print(f'  转涨事件: {n_ev} 个')

        # 事件臂: 前向超额
        ev_ex = {k: [] for k in HORIZONS}
        for ci, t in zip(ev_ci, ev_t):
            sc = P[codes[ci]].values
            for k in HORIZONS:
                x = fwd_excess(sc, A, t, k)
                if not np.isnan(x):
                    ev_ex[k].append(x)

        # 对照A: 随机日期臂 (同数量, 同板块集合)
        rd_ex = {k: [] for k in HORIZONS}
        for ci, _ in zip(ev_ci, ev_t):
            rt = int(rng.integers(60, n - 70))
            sc = P[codes[ci]].values
            for k in HORIZONS:
                x = fwd_excess(sc, A, rt, k)
                if not np.isnan(x):
                    rd_ex[k].append(x)

        # 对照B: 全体板块同日臂 (事件日当天, 全体板块平均超额)
        sd_ex = {k: [] for k in HORIZONS}
        for t in sorted(set(ev_t.tolist())):
            for k in HORIZONS:
                xs = []
                for ci in range(len(codes)):
                    x = fwd_excess(P[codes[ci]].values, A, t, k)
                    if not np.isnan(x):
                        xs.append(x)
                if xs:
                    sd_ex[k].append(np.mean(xs))

        print(f'  {"区间":<6}{"事件中位":>10}{"随机臂中位":>12}{"同日臂中位":>12}')
        verdict = []
        for k in HORIZONS:
            e = np.median(ev_ex[k]) * 100
            r = np.median(rd_ex[k]) * 100
            b = np.median(sd_ex[k]) * 100
            print(f'  {k:>4}日  {e:>+9.2f}% {r:>+11.2f}% {b:>+11.2f}%')
            verdict.append(e)
        ok1 = verdict[1] > 0 and verdict[1] > np.median(rd_ex[20]) * 100 + 1.0
        ok2 = verdict[0] > 0 and verdict[2] > 0
        ok4 = np.median(switches_per_year) <= WHIPSAW_LIMIT
        print(f'  判据: 20日中位>0且>随机+1pp: {"✅" if ok1 else "❌"} | 5/60日同向: {"✅" if ok2 else "❌"} | whipsaw≤10/年: {"✅" if ok4 else "❌"}')

if __name__ == '__main__':
    main()
