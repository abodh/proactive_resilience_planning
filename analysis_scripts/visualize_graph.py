"""
    Author: Abodh Poudyal (GitHub: @abodh)
    Resilience Planning for Transmission
    Date: Aug, 2023
"""
import os

# start from the project directory for easier file access
os.chdir("..")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as readmat
import pandas as pd


def extract_connectivity(optimized_df: pd.DataFrame, scenario_idx: int = 38) -> pd.DataFrame:
    """
        extracts the connectivity result of the optimization problem for specific scenario

            optimized_df (pd.DataFrame): dataframe of the solution obtained from the stochastic optimization problem
            scenario_idx (int): scenario under consideration

            return: pd.DataFrame
            connectivity df
    """

    # the final result of the optimization problem is not sorted in the order of scenarios
    # Hence, we sort the results here
    connectivity_df = optimized_df.loc[(optimized_df['var_name'] == 'xij') &
                                       (optimized_df['tree_node'] == 'Scenario' + str(scenario_idx))]
    connectivity_df.loc[:, 'var_idx'] = connectivity_df['var_idx'].astype(int)
    connectivity_df = connectivity_df.sort_values(['var_idx'])
    connectivity_df.reset_index(drop=True, inplace=True)
    return connectivity_df[['var_idx', 'var_value']]


def extract_load_shed(optimized_df: pd.DataFrame, scenario_idx: int = 38) -> pd.DataFrame:
    """
        extracts the load shed result of the optimization problem for specific scenario

            optimized_df (pd.DataFrame): dataframe of the solution obtained from the stochastic optimization problem
            scenario_idx (int): scenario under consideration

            return: pd.DataFrame
            load_shed df
    """

    # the final result of the optimization problem is not sorted in the order of scenarios
    # Hence, we sort the results here
    load_shed_df = optimized_df.loc[(optimized_df['var_name'] == 'load_shed') &
                                    (optimized_df['tree_node'] == 'Scenario' + str(scenario_idx))]
    load_shed_df.loc[:, 'var_idx'] = load_shed_df['var_idx'].astype(int)
    load_shed_df = load_shed_df.sort_values(['var_idx'])
    load_shed_df.reset_index(drop=True, inplace=True)
    return load_shed_df[['var_idx', 'var_value']]


def extract_hardening(optimized_df: pd.DataFrame) -> pd.DataFrame:
    """
        extracts the hardening result of the optimization problem

            optimized_df (pd.DataFrame): dataframe of the solution obtained from the stochastic optimization problem

            return: pd.DataFrame
            hardening df
    """

    # the final result of the optimization problem is not sorted in the order of scenarios
    # Hence, we sort the results here
    hardening_df = optimized_df.loc[(optimized_df['var_name'] == 'harden')]
    hardening_df.loc[:, 'var_idx'] = hardening_df['var_idx'].astype(int)
    hardening_df = hardening_df.sort_values(['var_idx'])
    hardening_df.reset_index(drop=True, inplace=True)
    return hardening_df[['var_idx', 'var_value']]


def extract_cap_upgrade(optimized_df: pd.DataFrame) -> pd.DataFrame:
    """
        extracts the capacity upgrade result of the optimization problem

            optimized_df (pd.DataFrame): dataframe of the solution obtained from the stochastic optimization problem

            return: pd.DataFrame
            capacity upgrade df
    """

    # the final result of the optimization problem is not sorted in the order of scenarios
    # Hence, we sort the results here
    cap_df = optimized_df.loc[(optimized_df['var_name'] == 'linecap_upgrade')]
    cap_df.loc[:, 'var_idx'] = cap_df['var_idx'].astype(int)
    cap_df = cap_df.sort_values(['var_idx'])
    cap_df.reset_index(drop=True, inplace=True)
    return cap_df[['var_idx', 'var_value']]


def extract_DG(optimized_df: pd.DataFrame) -> pd.DataFrame:
    """
        extracts the DG sizes result of the optimization problem

            optimized_df (pd.DataFrame): dataframe of the solution obtained from the stochastic optimization problem

            return: pd.DataFrame
            DG df
    """

    # the final result of the optimization problem is not sorted in the order of scenarios
    # Hence, we sort the results here
    DG_df = optimized_df.loc[(optimized_df['var_name'] == 'DG_size')]
    DG_df.loc[:, 'var_idx'] = DG_df['var_idx'].astype(int)
    DG_df = DG_df.sort_values(['var_idx'])
    DG_df.reset_index(drop=True, inplace=True)
    return DG_df[['var_idx', 'var_value']]


# Define a function to draw a gradient circle with center (x, y), radius r, and transparency alpha
def draw_gradient_circle(center, r, alpha):
    circle = plt.Circle(center, r, color='red', alpha=alpha)
    ax.add_artist(circle)


def draw_donut(center, radius, alpha):
    circle = plt.Circle(center, radius, color='lime', alpha=alpha)
    ax.add_artist(circle)


def min_max_norm(array):
    num = array - np.min(array)
    denom = np.max(array) - np.min(array)
    norm_array = num / denom
    return norm_array


if __name__ == "__main__":

    # get the current working directory
    dir_file = os.getcwd()

    # testcase to use for the optimization
    test_case = 'RTSGMLC'
    Pbase = 100
    noninvestment_key = ["DCOPF", "redispatch"]

    test_case_selector = {
        'risk': 0.95,
        'budget': 0.5,
        'key': 1
    }

    scenario_selector = 41

    # load matpower test case in mat format
    matpower_mat_file = readmat.loadmat(dir_file + '/power_system_test_cases/case' + test_case + '.mat',
                                        struct_as_record=False,
                                        squeeze_me=True)

    # ensure that all the saved mat file are saved under workspace var name 'matpower_testcase'
    test_case_mat = matpower_mat_file['matpower_testcase']

    # load on each nodes
    base_case_load = test_case_mat.bus[:, 2] / Pbase

    # extract the result corresponding to each optimization problem
    if test_case_selector['budget'] != 0:
        solution_file = pd.read_csv(f"results/ef-{float(test_case_selector['risk'])}-"
                                    f"{float(test_case_selector['budget'])}-"
                                    f"10-10-10.csv", delimiter=',')

    else:
        if test_case_selector['key'] != 0:
            solution_file = pd.read_csv(f"results/ef-{float(test_case_selector['risk'])}-"
                                        f"0.5-0-0-0.csv", delimiter=',')
        else:
            solution_file = pd.read_csv(f"results/ef-{float(test_case_selector['risk'])}-"
                                        f"0.0-0-0-0.csv", delimiter=',')

    bus_data = pd.read_csv(dir_file + '/power_system_test_cases/case' + test_case + '_bus.csv', delimiter=',')
    branch_data = pd.read_csv(dir_file + '/power_system_test_cases/case' + test_case + '_branch.csv', delimiter=',')

    connectivity = extract_connectivity(solution_file, scenario_idx=scenario_selector).to_numpy()
    load_shed_data = extract_load_shed(solution_file, scenario_idx=scenario_selector).to_numpy()
    hardening_data = extract_hardening(solution_file)
    upgrade_data = extract_cap_upgrade(solution_file)
    DG_data = extract_DG(solution_file).to_numpy()

    for idx, each_branch in branch_data.iterrows():
        from_kV = bus_data[bus_data['Bus ID'] == each_branch['From Bus']]['BaseKV'].item()
        to_kV = bus_data[bus_data['Bus ID'] == each_branch['To Bus']]['BaseKV'].item()
        if from_kV == to_kV:
            hardening_data.loc[idx, 'kV'] = to_kV
            upgrade_data.loc[idx, 'kV'] = to_kV
        else:
            hardening_data.loc[idx, 'kV'] = max(from_kV, from_kV)
            upgrade_data.loc[idx, 'kV'] = max(from_kV, from_kV)

    hardening_data = hardening_data.to_numpy()
    upgrade_data = upgrade_data.to_numpy()

    # convert the value to float instead of object
    load_shed_data = load_shed_data.astype(float)

    if test_case_selector['budget'] != 0:
        DG_data = DG_data.astype(float)

        # normalize DG data
        DG_data[:, 1] = min_max_norm(DG_data[:, 1])

    # normalize the load data by its overall demand in each node to obtain fraction of load shed
    # replace nan with 0 for division with 0 demand
    with np.errstate(divide='ignore', invalid='ignore'):
        temp = base_case_load[:]
        temp[temp == 0] = 1
        load_shed_data[:, 1] = load_shed_data[:, 1] / temp

    load_shed_data[:, 1] = np.round(load_shed_data[:, 1], 4)

    # Define the edges of the graph
    edges = list(zip(test_case_mat.branch[:, 0], test_case_mat.branch[:, 1]))
    edges_reversed = [(y, x) for x, y in edges]

    # Create a new graph
    G = nx.Graph()

    # Add the edges to the graph:
    G.add_edges_from(edges)

    positions = {}
    for nodes in G.nodes:
        lon = bus_data.loc[bus_data['Bus ID'] == int(nodes)]['lng'].item()
        lat = bus_data.loc[bus_data['Bus ID'] == int(nodes)]['lat'].item()
        positions[nodes] = (lon, lat)

    # Define node labels as their node number
    labels = {node: str(int(node)) for node in G.nodes}

    # for scen_idx in range(len(savefile_key)):
    # scenario = scen_idx + 1
    edge_styles = {}
    edge_colors = {}
    edge_width = {}
    edge_alpha = {}
    for idx, edges_list in enumerate(G.edges):
        try:
            if connectivity[edges.index(edges_list), 1] == 0:
                edge_styles[edges_list] = "dashed"
            else:
                edge_styles[edges_list] = "solid"
        except:
            if connectivity[edges_reversed.index(edges_list), 1] == 0:
                edge_styles[edges_list] = "dashed"
            else:
                edge_styles[edges_list] = "solid"
        try:
            if hardening_data[edges.index(edges_list), 1] == 1:
                if hardening_data[edges.index(edges_list), 2] == 138:
                    edge_colors[edges_list] = "g"
                    edge_width[edges_list] = 5
                    edge_alpha[edges_list] = 1
                else:
                    # 230 kV test
                    edge_colors[edges_list] = "salmon"
                    edge_width[edges_list] = 5
                    edge_alpha[edges_list] = 1

            else:
                # non hardened
                edge_colors[edges_list] = "b"
                edge_width[edges_list] = 1
                edge_alpha[edges_list] = 0.75

        except:
            # reverse edge case
            if hardening_data[edges_reversed.index(edges_list), 1] == 1:
                if hardening_data[edges_reversed.index(edges_list), 2] == 138:
                    edge_colors[edges_list] = "g"
                    edge_width[edges_list] = 5
                    edge_alpha[edges_list] = 1
                else:
                    # 230 kV test
                    edge_colors[edges_list] = "salmon"
                    edge_width[edges_list] = 5
                    edge_alpha[edges_list] = 1

            else:
                # non hardened
                edge_colors[edges_list] = "b"
                edge_width[edges_list] = 1
                edge_alpha[edges_list] = 0.75

        try:
            if upgrade_data[edges.index(edges_list), 1] == 1:
                if upgrade_data[edges.index(edges_list), 2] == 138:
                    edge_colors[edges_list] = "magenta"
                    edge_width[edges_list] = 2
                    edge_alpha[edges_list] = 1
                else:
                    # 230 kV test
                    edge_colors[edges_list] = "darkviolet"
                    edge_width[edges_list] = 2
                    edge_alpha[edges_list] = 1

        except:
            # reverse edge case
            if upgrade_data[edges_reversed.index(edges_list), 1] == 1:
                if upgrade_data[edges_reversed.index(edges_list), 2] == 138:
                    edge_colors[edges_list] = "magenta"
                    edge_width[edges_list] = 2
                    edge_alpha[edges_list] = 1
                else:
                    # 230 kV test
                    edge_colors[edges_list] = "darkviolet"
                    edge_width[edges_list] = 2
                    edge_alpha[edges_list] = 1

        try:
            if upgrade_data[edges.index(edges_list), 1] == 1 and hardening_data[edges.index(edges_list), 1] == 1:
                if upgrade_data[edges.index(edges_list), 2] == 138:
                    edge_colors[edges_list] = "aqua"
                    edge_width[edges_list] = 4
                    edge_alpha[edges_list] = 1
                else:
                    # 230 kV test
                    edge_colors[edges_list] = "slategray"
                    edge_width[edges_list] = 4
                    edge_alpha[edges_list] = 1

        except:
            # reverse edge case
            if upgrade_data[edges_reversed.index(edges_list), 1] == 1 and \
                    hardening_data[edges_reversed.index(edges_list), 1] == 1:
                if upgrade_data[edges_reversed.index(edges_list), 2] == 138:
                    edge_colors[edges_list] = "aqua"
                    edge_width[edges_list] = 4
                    edge_alpha[edges_list] = 1
                else:
                    # 230 kV test
                    edge_colors[edges_list] = "slategray"
                    edge_width[edges_list] = 4
                    edge_alpha[edges_list] = 1

    # Assign load shed and DG size data to each node
    for node_idx in G.nodes:
        G.nodes[node_idx]['load_shed'] = load_shed_data[np.where(bus_data['Bus ID'] == node_idx)[0][0], 1]
        G.nodes[node_idx]['DG_size'] = DG_data[np.where(bus_data['Bus ID'] == node_idx)[0][0], 1]

    # Draw the graph
    fig, ax = plt.subplots()
    # pos = nx.spring_layout(G, k=0.5, seed=3, scale=2)  # positions for all nodes
    nx.draw_networkx_nodes(G, positions, node_color='black', node_size=10, edgecolors='black', linewidths=2)
    # nx.draw_networkx_labels(G, positions, labels, font_size=8, verticalalignment='bottom',
    #                         horizontalalignment='right')

    for nodes in G.nodes:
        x, y = positions[nodes]
        r = 0.03
        dg = 0.6
        draw_gradient_circle((x, y), 3.5 * r, G.nodes[nodes]['load_shed'])
        if test_case_selector['budget'] != 0:
            draw_donut((x, y), 3.5 * r * dg, G.nodes[nodes]['DG_size'])

    # nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)
    nx.draw_networkx_edges(G, positions, edgelist=edge_styles.keys(), edge_color=edge_colors.values(),
                           alpha=[a for a in edge_alpha.values()], style=edge_styles.values(),
                           width=[lw for lw in edge_width.values()])

    plt.axis('off')
    ax.set_aspect('equal')

    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname': 'Times New Roman', 'size': '14', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    # axis_font = {'fontname': 'Times', 'size': '12'}

    if test_case_selector['budget'] != 0:
        plt.title(f"$\mathrm{{\lambda}}$ = {test_case_selector['risk']} and "
                  f"$\mathrm{{\mathcal{{C}}_T^{{max}}}}$: \${test_case_selector['budget']} bn", **title_font, y=0.93)
        plt.tight_layout()
        plt.savefig(f"analysis_scripts\{test_case_selector['risk']}_{test_case_selector['budget']}_graph" + '.pdf',
                    bbox_inches="tight", dpi=300, format='pdf')

    else:
        if test_case_selector['key'] != 0:
            plt.title("resilient re-dispatch", **title_font, y=0.93)
            plt.tight_layout()
            plt.savefig(r"analysis_scripts\redispatch_graph" + '.pdf',
                        bbox_inches="tight", dpi=300, format='pdf')

        else:
            plt.title("DCOPF", **title_font, y=0.93)
            plt.tight_layout()
            plt.savefig(r"analysis_scripts\DCOPF_graph" + '.pdf',
                        bbox_inches="tight", dpi=300, format='pdf')

    plt.show()

    plt.figure()
    out_of_service = plt.plot([], [], color='blue', linestyle='--', label='out-of-service')
    in_service = plt.plot([], [], color='blue', label='in-service')
    harden_132 = plt.plot([], [], color='green', linewidth=5, label='harden 132kV')
    harden_230 = plt.plot([], [], color='salmon', linewidth=5, label='harden 230 kV')
    upgrade_132 = plt.plot([], [], color='magenta', linewidth=2, label='upgrade 132kV')
    upgrade_230 = plt.plot([], [], color='darkviolet', linewidth=2, label='upgrade 230 kV')
    upgrade_harden_132 = plt.plot([], [], color='aqua', linewidth=4, label='upgrade+harden 132kV')
    upgrade_harden_230 = plt.plot([], [], color='slategray', linewidth=4, label='upgrade+harden 230 kV')
    load_shed = plt.plot([], [], marker='o', markeredgecolor='r', markerfacecolor='r', markersize=5,
                         label='load shed', linestyle='')
    DG = plt.plot([], [], marker='o', markeredgecolor='lime', markerfacecolor=plt.cm[''], markersize=5,
                  label='DG', linestyle='')

    legend = plt.legend(framealpha=1, frameon=True, ncol=5, markerscale=2, prop={"family": "Times New Roman",
                                                                                 'size': 10})
    plt.axis('off')

    expand = [-1, -1, 1, 1]
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    if not os.path.isfile(r"analysis_scripts\figure_legend.pdf"):
        fig.savefig(r"analysis_scripts\figure_legend.pdf", dpi=300, bbox_inches=bbox)
