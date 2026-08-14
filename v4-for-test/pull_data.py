#!/usr/bin/env python3
"""拉取回测指数数据 → v4-for-test/data/
- 000985.CSI 中证全指 (全A代理)
- 申万 L1 31指数 九年日线
"""
import os, sys, sqlite3, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processing'))
from config import TUSHARE_TOKEN
import tushare as ts

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA, exist_ok=True)
START, END = '20180101', '20260814'

def pull_index(code):
    pro = ts.pro_api(TUSHARE_TOKEN)
    frames = []
    for s, e in ((START, '20220101'), ('20220101', END)):
        df = pro.index_daily(ts_code=code, start_date=s, end_date=e)
        if df is not None and len(df):
            frames.append(df)
        time.sleep(0.3)
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates('trade_date').sort_values('trade_date')
    return df[['ts_code', 'trade_date', 'close', 'pct_chg']]

def main():
    # 000985.CSI
    df = pull_index('000985.CSI')
    if df is None or not len(df):
        print('❌ 000985.CSI 拉取失败')
        return
    df.to_csv(os.path.join(DATA, '000985_csi.csv'), index=False)
    print(f'✅ 000985.CSI: {len(df)} 天 {df.trade_date.min()}~{df.trade_date.max()}')

    # 申万 L1 指数代码
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pool.db')).cursor()
    codes = [r[0] for r in c.execute("SELECT index_code FROM sw_classify WHERE level='L1' ORDER BY index_code").fetchall()]
    print(f'申万 L1 指数: {len(codes)} 个, 开始拉取...')
    allframes = []
    for i, code in enumerate(codes):
        df = pull_index(code)
        if df is not None and len(df):
            allframes.append(df)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(codes)}')
        time.sleep(0.3)
    big = pd.concat(allframes)
    big.to_csv(os.path.join(DATA, 'sw_l1_daily.csv'), index=False)
    n_days = big.groupby('ts_code').size()
    print(f'✅ 申万L1: {len(allframes)} 个指数, 总{len(big)}行, 每个 {n_days.min()}~{n_days.max()} 天')

if __name__ == '__main__':
    main()
