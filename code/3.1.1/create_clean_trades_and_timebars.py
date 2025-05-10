"""
processes labeled trade data to create cleaned trade and timebar parquet files
"""
import json
import pandas as pd
import numpy as np

with open('kalshi_trades_labeled.json') as f:
    raw_labeled_trades = json.load(f)

rows_for_clean_df = []
for market_id, trades_in_market in raw_labeled_trades.items():
    for trade_info in trades_in_market:
        rows_for_clean_df.append({
            'market': market_id,
            'trade_id': trade_info['trade_id'],
            'time': pd.to_datetime(trade_info['created_time']),
            'size': trade_info['count'],
            'price': trade_info['yes_price'] if trade_info['taker_side'] == 'yes' else trade_info['no_price'],
            'label': trade_info['label']
        })

df_all_trades = pd.DataFrame(rows_for_clean_df)
clean_trades_df = df_all_trades[df_all_trades['label'] != 'Uncertain'].copy()

clean_trades_df.to_parquet('clean_trades.parquet', index=False)
print("clean_trades.parquet created successfully.")

timebars_per_market_list = []

if not clean_trades_df.empty:
    for market_id, group_df in clean_trades_df.groupby('market'):
        if group_df.empty:
            continue

        flow_market = (group_df.set_index('time')
                       .sort_index()
                       .groupby('label')
                       .resample('1min')['size']
                       .sum()
                       .unstack('label', fill_value=0))

        price_market = (group_df.set_index('time')['price']
                        .resample('1min')
                        .last())
        
        market_timebar_df = flow_market.join(price_market, how='outer')
        
        if 'Institutional' not in market_timebar_df.columns:
            market_timebar_df['Institutional'] = 0
        if 'Retail' not in market_timebar_df.columns:
            market_timebar_df['Retail'] = 0
        
        market_timebar_df['Institutional'] = market_timebar_df['Institutional'].fillna(0)
        market_timebar_df['Retail'] = market_timebar_df['Retail'].fillna(0)
        
        market_timebar_df['price'] = market_timebar_df['price'].ffill()
        
        market_timebar_df['market'] = market_id
        timebars_per_market_list.append(market_timebar_df.reset_index())

if timebars_per_market_list:
    final_timebars_df = pd.concat(timebars_per_market_list, ignore_index=True)
    
    desired_cols = ['market', 'time', 'Institutional', 'Retail', 'price']
    for col in desired_cols:
        if col not in final_timebars_df.columns:
            if col in ['Institutional', 'Retail']:
                 final_timebars_df[col] = 0.0
            else:
                 final_timebars_df[col] = np.nan
    
    final_timebars_df = final_timebars_df[desired_cols]
    final_timebars_df = final_timebars_df.sort_values(['market', 'time']).reset_index(drop=True)
    
    final_timebars_df.to_parquet('timebars.parquet', index=False)
    print("timebars.parquet created successfully with 'market' column.")
else:
    print("No data processed for timebars.parquet. 'clean_trades_df' might be empty or have no recognized markets.")