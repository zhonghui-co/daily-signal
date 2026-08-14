#!/usr/bin/env python3
"""v4-for-test 共用工具: 因果状态重放 + 执行模拟"""
import numpy as np

def causal_states(closes, W, t_end=None):
    """因果状态序列: 每天只用当时已知信息
    - 分形拐点需右窗口 W 日确认 → t 时只知 i <= t-W 的拐点
    - 今日收盘当作极值(交替类型)参与判定
    返回 (t_end+1) 长 int8 数组: 1=上涨N / -1=下跌N / 0=横盘"""
    n = len(closes) if t_end is None else t_end + 1
    piv = []  # [(i, 'H'|'L', v)]
    states = np.zeros(n, dtype=np.int8)
    for t in range(n):
        i = t - W
        if W <= i < len(closes) - W:
            c = closes[i]
            if c is not None:
                left = closes[i - W:i]
                right = closes[i + 1:i + W + 1]
                win = list(left) + list(right)
                if all(x is not None and c > x for x in win):
                    tt = 'H'
                elif all(x is not None and c < x for x in win):
                    tt = 'L'
                else:
                    tt = None
                if tt:
                    if piv and piv[-1][1] == tt:
                        if (tt == 'H' and c > piv[-1][2]) or (tt == 'L' and c < piv[-1][2]):
                            piv[-1] = (i, tt, c)
                    else:
                        piv.append((i, tt, c))
        states[t] = last_state(piv, t, closes[t])
    return states

def last_state(piv, t, close_t):
    """t 日收盘后的末态。piv: 已知拐点 [(i,'H'|'L',v)]
    规则与 compute_v4_state.process 完全一致, 只算最后一天"""
    pv = list(piv)
    if pv and pv[-1][0] < t:
        pv.append((t, 'L' if pv[-1][1] == 'H' else 'H', close_t))
    m = len(pv)
    if m == 0:
        return 0
    prev_same = []
    lastL = lastH = -1
    for j in range(m):
        if pv[j][1] == 'L':
            prev_same.append(lastL)
            lastL = j
        else:
            prev_same.append(lastH)
            lastH = j
    cmpv = [0] * m
    for j in range(2, m):
        p = prev_same[j]
        if p >= 0:
            cmpv[j] = 1 if pv[j][2] > pv[p][2] else (-1 if pv[j][2] < pv[p][2] else 0)
    DIR = 0
    s0 = 0
    for j in range(2, m):
        c = cmpv[j]
        if DIR == 0:
            if c != 0 and cmpv[j - 1] == c and j - 3 >= s0:
                s0 = j - 3
                DIR = c
        elif DIR == 1:
            if c <= 0:
                b = j - 2 if pv[j][1] == 'H' else j - 1
                s0 = b
                DIR = -1 if c < 0 else 0
        else:
            if c >= 0:
                b = j - 2 if pv[j][1] == 'L' else j - 1
                s0 = b
                DIR = 1 if c > 0 else 0
    # 最后结构有效 → 其方向; 否则末腿方向
    if DIR != 0 and (m - 1) - s0 >= 3:
        ok = all((prev_same[j] < s0) or (cmpv[j] == DIR) for j in range(s0 + 2, m))
        if ok:
            return DIR
    dg = 1 if pv[-1][2] > pv[-2][2] else (-1 if pv[-1][2] < pv[-2][2] else 0)
    return dg

def state_machine_56_3(rets_pct):
    """价格法状态机 5.6/-3: 1=多, -1=空. 空头中两日合计>+5.6% 转多"""
    return _state_machine(rets_pct, 5.6, 3.0)

def state_machine_v1(rets_pct):
    """价格法状态机 v1 口径 3.4/-2.0 (MEMORY: 九年回测验证过的那套)"""
    return _state_machine(rets_pct, 3.4, 2.0)

def _state_machine(rets_pct, th_up, th_dn):
    st = [1]
    for t in range(1, len(rets_pct)):
        r = rets_pct[t]
        if st[-1] == 1:
            st.append(-1 if r < -th_dn else 1)
        else:
            two = rets_pct[t] + rets_pct[t - 1]
            st.append(1 if (r > th_up or two > th_up) else -1)
    return np.array(st)

def simulate(pos, rets, cost=0.001):
    """pos[t] = t日收盘定的仓位, 应用于 t+1 日收益; cost=单边费率
    返回 (总收益, 年化, MaxDD, 净值序列)"""
    pos = np.asarray(pos, dtype=float)
    n = len(rets)
    eq = np.ones(n)
    turnover = np.zeros(n)
    for t in range(1, n):
        turnover[t] = abs(pos[t] - pos[t - 1])
    for t in range(1, n):
        p = pos[t - 1]
        eq[t] = eq[t - 1] * (1 + p * rets[t] - turnover[t] * cost)
    total = eq[-1] / eq[0] - 1
    dd = np.min(eq / np.maximum.accumulate(eq) - 1)
    return total, dd, eq

def yearly_table(eq, dates, pos=None):
    """按年汇总净收益"""
    years = sorted({d[:4] for d in dates})
    out = {}
    for y in years:
        idx = [i for i, d in enumerate(dates) if d[:4] == y]
        if len(idx) >= 2:
            out[y] = eq[idx[-1]] / eq[idx[0]] - 1
    return out
