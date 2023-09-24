import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as readmat
import pdb

# get the current working directory
dir_file = os.getcwd()

# testcase to use for the optimization
test_case = '39'
savefile_key = ["DCOPF", "redispatch", "1_harden", "2_harden", "5_harden", "10_harden"]
scenario = 2  # [1-> DCOPF, 2->redispatch 3->1 harden 4-> 2 harden, 5 -> 5 harden, 6-> 10 harden]

# load matpower test case in mat format
matpower_mat_file = readmat.loadmat(dir_file + '/DCOPF/power_system_test_cases/case' + test_case + '.mat',
                                    struct_as_record=False,
                                    squeeze_me=True)
# ensure that all the saved mat file are saved under workspace var name 'matpower_testcase'
test_case = matpower_mat_file['matpower_testcase']

# connectivity data for 39 bus
connectivity = np.loadtxt("connectivity.csv", delimiter=',')

# load shed data for 39 bus
load_shed_data = np.loadtxt("load_shed.csv", delimiter=',')

# line hardening data for 39 bus
hardening_data = np.loadtxt("hardening.csv", delimiter=',')

# normalize the data by its max demand replace nan with 0 for division with 0 demand
with np.errstate(divide='ignore', invalid='ignore'):
    load_shed_data[:, 1:7] = np.where(load_shed_data[:, 7:8] != 0, load_shed_data[:, 1:7] / load_shed_data[:, 7:8], 0)

# Define the edges of the graph
edges = list(zip(test_case.branch[:, 0], test_case.branch[:, 1]))
edges_reversed = [(y, x) for x, y in edges]

# Create a new graph
G = nx.Graph()

# Add the edges to the graph:
G.add_edges_from(edges)

# Define node labels as their node number
labels = {node: str(int(node)) for node in G.nodes}

# Define a function to draw a gradient circle with center (x, y), radius r, and transparency alpha
def draw_gradient_circle(center, r, alpha):
    circle = plt.Circle(center, r, color='red', alpha=alpha)
    ax.add_artist(circle)

for scen_idx in range(len(savefile_key)):
    scenario = scen_idx + 1
    edge_styles = {}
    edge_colors = {}
    edge_width = {}
    edge_alpha = {}
    for idx, edges_list in enumerate(G.edges):
        try:
            if connectivity[edges.index(edges_list), scenario] == 0:
                edge_styles[edges_list] = "dashed"
            else:
                edge_styles[edges_list] = "solid"
        except:
            if connectivity[edges_reversed.index(edges_list), scenario] == 0:
                edge_styles[edges_list] = "dashed"
            else:
                edge_styles[edges_list] = "solid"
        try:
            if hardening_data[edges.index(edges_list), scenario] == 1:
                edge_colors[edges_list] = "g"
                edge_width[edges_list] = 2
                edge_alpha[edges_list] = 1
            else:
                edge_colors[edges_list] = "b"
                edge_width[edges_list] = 1.5
                edge_alpha[edges_list] = 0.75
        except:
            if hardening_data[edges_reversed.index(edges_list), scenario] == 1:
                edge_colors[edges_list] = "g"
                edge_width[edges_list] = 2
                edge_alpha[edges_list] = 1
            else:
                edge_colors[edges_list] = "b"
                edge_width[edges_list] = 1.5
                edge_alpha[edges_list] = 0.75

    # Assign load shed data to each node
    for node_idx in G.nodes:
        G.nodes[node_idx]['load_shed'] = load_shed_data[np.where(load_shed_data[:, 0] == node_idx)[0][0], scenario]

    # Draw the graph
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, k=0.5, seed=3, scale=2)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=10, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, labels, font_size=9, verticalalignment='bottom',
                            horizontalalignment='right')

    for n in G.nodes:
        x, y = pos[n]
        r = 0.03
        draw_gradient_circle((x, y), 3.5 * r, G.nodes[n]['load_shed'])

    # nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=edge_styles.keys(), edge_color=edge_colors.values(),
                           alpha=[a for a in edge_alpha.values()], style=edge_styles.values(),
                           width=[lw for lw in edge_width.values()])

    plt.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    # plt.savefig(savefile_key[scenario-1] + '.png', bbox_inches="tight", dpi=300)
    plt.show()