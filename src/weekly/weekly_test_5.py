import pandas as pd

def change_price_to_float(input_df):
    output_df = input_df.copy()

    output_df['item_price'] = output_df['item_price'].str.replace('$', '').astype(float)

    return pd.DataFrame(output_df)

def number_of_observations(input_df):
    return len(input_df)


def items_and_prices(input_df: pd.DataFrame) -> pd.DataFrame:
    items_prices = input_df[['item_name', 'item_price']]
    return pd.DataFrame(items_prices)



def sorted_by_price(input_df):
    sorted_items = input_df.sort_values(by='item_price', ascending=False).reset_index(drop=True)
    return pd.DataFrame(sorted_items)

def avg_price(input_df):
    average_price = input_df['item_price'].mean().astype(float)
    return average_price


def unique_items_over_ten_dollars(input_df: pd.DataFrame) -> pd.DataFrame:
    items_over_ten = input_df[input_df['item_price'] > 10]
    unique_items = items_over_ten.drop_duplicates(subset=['item_name', 'choice_description', 'item_price'])
    relevant_columns = ['item_name', 'choice_description', 'item_price']
    unique_items = unique_items[relevant_columns]

    return pd.DataFrame(unique_items)



def items_starting_with_s(input_df):
    items_with_s = input_df[input_df['item_name'].str.startswith('S')]
    unique_names = items_with_s.drop_duplicates(subset=['item_name'])

    return pd.DataFrame(unique_names[['item_name']])


def first_three_columns(input_df):
    first_three = input_df.iloc[:, :3]
    return pd.DataFrame(first_three)

def every_column_except_last_two(input_df):
    selected_columns = input_df.iloc[:, :-2]
    return pd.DataFrame(selected_columns)


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]

    result_df = filtered_df[columns_to_keep]

    return pd.DataFrame(result_df)

def generate_quartile(input_df):
    df_with_quartile = input_df.copy()

    conditions = [
        (df_with_quartile['item_price'] < 10),
        (df_with_quartile['item_price'] >= 10) & (df_with_quartile['item_price'] < 20),
        (df_with_quartile['item_price'] >= 20) & (df_with_quartile['item_price'] < 30),
        (df_with_quartile['item_price'] >= 30)
    ]

    choices = ['low-cost', 'medium-cost', 'high-cost', 'premium']

    df_with_quartile['Quartile'] = pd.Series(np.select(conditions, choices)).astype(str)

    return pd.DataFrame(df_with_quartile)


def average_price_in_quartiles(input_df):
    avg_prices = input_df.groupby('Quartile')['item_price'].mean()

    return pd.DataFrame((avg_prices.reset_index()))

def minmaxmean_price_in_quartile(input_df):
    stats = input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])

    return pd.DataFrame(stats)







