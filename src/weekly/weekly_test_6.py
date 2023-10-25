df = pd.read_parquet('../data/sp500.parquet', engine='fastparquet')

ff_factors_df = pd.read_parquet('ff_factors.parquet', engine='fastparquet')

merged_df = pd.merge(df, ff_factors_df, on='Date', how='left')

merged_df['Excess Return'] = merged_df['Return'] - merged_df['RF']

merged_df.sort_values(by='Date', inplace=True)

merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)



