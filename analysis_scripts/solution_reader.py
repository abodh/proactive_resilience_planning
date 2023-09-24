"""
    Author: Abodh Poudyal (GitHub: @abodh)
    Resilience Planning for Transmission
    Date: Aug, 2023
"""
import os

# start from the project directory for easier file access
os.chdir("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def expected_load_shed(optimized_df: pd.DataFrame,
                       probability: list):
    """
        Computes expected load shed for the optimization problem based on the load shed of each scenarios and their
        corresponding probability

            optimized_df (pd.DataFrame): dataframe of the solution obtained from the stochastic optimization problem
            probability: probability of each scenarios

            return: float, list
            expected load shed for the specific problem as well as list of load shed for each scenarios
    """

    # the final result of the optimization problem is not sorted in the order of scenarios
    # Hence, we sort the load shed results here
    load_shed_df = optimized_df.loc[optimized_df['var_name'] == 'load_shed_value']
    load_shed_df.loc[:, 'tree_node'] = load_shed_df['tree_node'].map(lambda x: x.lstrip('Scenario')).astype(int)
    load_shed_df = load_shed_df.sort_values(['tree_node'])

    # compute and return the expected as well as scenario wise load shed (sorted)
    scenario_load_shed = load_shed_df['var_value'].to_numpy()
    load_shed_expected = np.dot(scenario_load_shed, probability)

    return load_shed_expected, scenario_load_shed


def plot_load_shed(iteration: int,
                   investment_combination: int,
                   budget_value: float,
                   risk_value: float,
                   load_shed: dict,
                   plot_tick_labels: list,
                   DGs: list,
                   harden_cap_combination: int,
                   cvar_flag: bool = False) -> None:
    """
        plot the load shed and cvar (separately) for each combination of DG, hardening, and capacity upgrade
        The DGs constraint is represented by each lines and is observed in the legend
        hardening and capacity upgrade is represented on the x axis
        y-axis represents either expected load loss or CVaR

        iteration: int -> current iteration in terms of plot for risk combination
        investment_combination: int -> total number of combination for all investments eg. [0,0,0]
        budget_value: float -> current budget for plot title record keeping purpose
        risk_value: float -> current risk preference (lambda) for plot title record keeping purpose
        load_shed: dict -> expected_load_shed dict for each combination iteration,
        plot_tick_labels: list -> tick label combinations for hardening and capacity upgrade,
        DGs: list -> list of number of DGs,
        harden_cap_combination: int -> number of hardening and capacity upgrade combinations eg. [0-0],
        cvar_flag: bool -> boolean value for y-label assignment in the plot

        return: None

    """
    plt.figure()
    plt.title(f"budget: ${budget_value} bn and risk: {risk_value}")
    shed_lists = sorted(load_shed.items())
    _, shed_values = zip(*shed_lists)

    data_segments = [shed_values[i:i + harden_cap_combination] for i in
                     range(iteration - investment_combination, iteration, harden_cap_combination)]

    # list of some markers in python
    markers = {'o': 'circle',
               'v': 'triangle_down',
               '^': 'triangle_up',
               '<': 'triangle_left',
               '>': 'triangle_right',
               '1': 'tri_down',
               '2': 'tri_up',
               '3': 'tri_left',
               '4': 'tri_right',
               '8': 'octagon',
               's': 'square',
               'p': 'pentagon',
               '*': 'star',
               'h': 'hexagon1',
               'H': 'hexagon2',
               '+': 'plus',
               'x': 'x',
               'D': 'diamond',
               'd': 'thin_diamond'}
    markers_list = list(markers)
    for i, segment in enumerate(data_segments):
        plt.plot(segment, label=f'DG:{DGs[i]}', marker=random.choice(markers_list), linewidth=2, markersize=8)

    plt.xticks(ticks=range(harden_cap_combination), labels=plot_tick_labels, rotation=45)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.25)
    plt.xlabel("number of possible lines for (hardening-capacity_upgrade)")
    if cvar_flag:
        plt.ylabel("CVaR of load shed (pu)")
    else:
        plt.ylabel("expected load shed (pu)")

    plt.show()


def compute_cvar(load_shed_list: list,
                 value_at_risk_index: int) -> int:
    """
        computes cvar of the scenario-wise load shed data for each optimization problem

        load_shed_list (list): list with scenario-wise load shed data
        value_at_risk_index (int): index representing the value at risk for a confidence level

        return: int
        returns the conditional value-at-risk for the scenario-wise load shed data

    """
    return np.mean(load_shed_list[value_at_risk_index:])


if __name__ == "__main__":

    # flags to plot and compute cvar in the analysis
    plot_flag = False
    compute_cvar_flag = True

    # combination of risk, budget, DG, hardening, capacity upgrade used
    risk_preference = [0, 0.95]
    operator_budget_count = [0.5, 1, 2, 3]
    DG_counts = [0, 1, 5, 10]
    line_hardening_counts = [0, 1, 5, 10]
    line_capacity_upgrade_counts = [0, 1, 5, 10]

    # identify the overall investment combinations (including DG) size for plot purpose
    combination_length = len(DG_counts) * len(line_hardening_counts) * len(line_capacity_upgrade_counts)

    # # identify the overall investment combinations (excluding DG) size for plot purpose
    harden_cap_com_length = len(line_hardening_counts) * len(line_capacity_upgrade_counts)

    # obtain probability
    probability_file = 'scenario_files/probability.csv'
    scenario_probability = np.loadtxt(probability_file, delimiter=',')

    # identify the var inder for CVaR confidence of 0.05 i.e. 5%
    var_index = np.where(np.cumsum(scenario_probability) >= 0.95)[0][0]

    # tick labels for hardening-capacity upgrade combinations
    tick_labels = [f"{i}-{j}" for i in line_hardening_counts
                   for j in line_capacity_upgrade_counts]

    # initialize temporary variables
    iter_counter = 0                # iteration counter for all potential combinations (including risk)
    expected_load_shed_dict = {}    # expected load shed dict for each iteration
    cvar_dict = {}                       # cvar dict for each iteration
    solution_df = []                # empty list to store solution for each iteration

    for risk in risk_preference:
        for budget in operator_budget_count:
            for DG in DG_counts:
                for harden in line_hardening_counts:
                    for cap in line_capacity_upgrade_counts:
                        solution_dict = {'risk': risk, 'budget': budget, 'DG': DG, 'harden': harden, 'cap': cap}

                        # extract the result corresponding to each optimization problem
                        solution_file = pd.read_csv(f"results/ef-{float(risk)}-{float(budget)}-"
                                                    f"{DG}-{harden}-{cap}.csv", delimiter=',')

                        # expected load shed and list of sorted load shed for each optimization problem
                        expected_load_shed_dict[iter_counter], scenario_wise_load_shed = \
                            expected_load_shed(solution_file, scenario_probability)

                        # for some reason the
                        # compute_cvar(scenario_wise_load_shed, var_index)
                        cvar_dict[iter_counter] = compute_cvar(scenario_wise_load_shed, var_index)

                        solution_dict['load_shed'] = expected_load_shed_dict[iter_counter]
                        solution_dict['cvar'] = cvar_dict[iter_counter]
                        solution_df.append(solution_dict)

                        # print("No exception: ")
                        print(iter_counter, risk, budget, DG, harden, cap,
                              expected_load_shed_dict[iter_counter],
                              cvar_dict[iter_counter])

                        iter_counter += 1
                        del scenario_wise_load_shed, solution_file

            if plot_flag:
                plot_load_shed(iter_counter, combination_length, budget, risk, expected_load_shed_dict, tick_labels,
                               DG_counts, harden_cap_com_length)
                if compute_cvar_flag:
                    plot_load_shed(iter_counter, combination_length, budget, risk, cvar_dict, tick_labels,
                                   DG_counts, harden_cap_com_length, cvar_flag=True)

    # form the dataframe of the list and save a csv
    solution_df = pd.DataFrame(solution_df)
    solution_df.to_csv('analysis_scripts/overall_load_shed_cvar.csv', index=False, header=False)
