#ab9oac

import pandas as pd
from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt

euro12 = pd.read_csv('../data/Euro_2012_stats_TEAM.csv')

def number_of_participants(input_df):
    return len(input_df)
participants = number_of_participants(euro12)


def goals(input_df):
    goals_df = input_df[['Team', 'Goals']]
    return goals_df

goals_data = goals(euro12)


def sorted_by_goal(input_df):
    goals_df = goals(input_df)
    sorted_goals_df = goals_df.sort_values(by='Goals', ascending=False)
    return sorted_goals_df

sorted_goals_data = sorted_by_goal(euro12)

def avg_goal(input_df):
    avg = input_df['Goals'].mean()
    return avg


average_goals = avg_goal(euro12)


def countries_over_six(input_df):
    selected_countries = input_df[input_df['Goals'] >= 6]
    return selected_countries

countries_over_six_goals = countries_over_six(euro12)

def countries_starting_with_g(input_df):
    selected_countries = input_df[input_df['Team'].str.startswith('G')]
    return selected_countries


countries_starting_with_g_df = countries_starting_with_g(euro12)



def first_seven_columns(input_df):
    first_seven = input_df.iloc[:, :7]
    return first_seven

first_seven_cols_df = first_seven_columns(euro12)


def every_column_except_last_three(input_df):
    columns_except_last_three = input_df.iloc[:, :-3]
    return columns_except_last_three

columns_except_last_three_df = every_column_except_last_three(euro12)



def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    selected_columns = input_df[columns_to_keep]
    filtered_rows = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return selected_columns.join(filtered_rows, how='inner', lsuffix='_left')

columns_to_keep = ['Goals']
column_to_filter = 'Team'
rows_to_keep = ['Germany', 'Portugal']
sliced_df = sliced_view(euro12, columns_to_keep, column_to_filter, rows_to_keep)


def generate_quartile(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], [-1, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    return input_df

euro12_with_quartile = generate_quartile(euro12)

def average_yellow_in_quartiles(input_df):
    average_yellow_df = input_df.groupby('Quartile')['Passes'].mean().reset_index()
    average_yellow_df.rename(columns={'Yellow Cards': 'Passes Completed'}, inplace=True)
    return average_yellow_df

average_yellow_in_quartiles_df = average_yellow_in_quartiles(euro12_with_quartile)

def minmax_block_in_quartile(input_df):
    minmax_block_df = input_df.groupby('Quartile')['Blocks'].agg(['min', 'max']).reset_index()
    minmax_block_df.rename(columns={'min': 'Minimum Blocks', 'max': 'Maximum Blocks'}, inplace=True)
    return minmax_block_df

minmax_block_in_quartile_df = minmax_block_in_quartile(euro12_with_quartile)


import matplotlib.pyplot as plt

def scatter_goals_shots(input_df):
    fig, ax = plt.subplots()
    ax.scatter(input_df['Goals'], input_df['Shots on target'])

    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')

    ax.set_title('Goals and Shot on target')

    plt.show()

    return fig

scatter_plot = scatter_goals_shots(euro12)


def scatter_goals_shots_by_quartile(input_df):
    quartiles = input_df['Quartile'].unique()

    colors = ['b', 'g', 'r', 'c']

    fig, ax = plt.subplots()
    for i, quartile in enumerate(quartiles):
        subset = input_df[input_df['Quartile'] == quartile]
        ax.scatter(subset['Goals'], subset['Shots on target'], label=f'Quartile {quartile}', color=colors[i])

    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')

    ax.set_title('Goals and Shot on target')

    ax.legend(title='Quartiles')

    plt.show()

    return fig

scatter_plot_by_quartile = scatter_goals_shots_by_quartile(euro12_with_quartile)

import random

def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)

    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0

        for _ in range(length_of_trajectory):
            random_value = pareto_distribution(1, 1)
            cumulative_sum += random_value
            trajectory.append(cumulative_sum / (length_of_trajectory + 1))

        trajectories.append(trajectory)

    return trajectories


pareto_distribution = lambda a, b: random.paretovariate(a)
number_of_trajectories = 5
length_of_trajectory = 10

result = gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory)



