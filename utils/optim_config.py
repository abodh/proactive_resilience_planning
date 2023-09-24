import os, sys

if os.getcwd() + r'\utils' not in sys.path:
    sys.path.insert(0, os.getcwd() + r'\utils')

from branch_cost_analysis import upgrade_cost
from main import test_case_loop
import numpy as np

# configuration and parameters for the optimization problem
configs = {}

configs = {
    'test_case_id': 'RTSGMLC',              # test case under
    'upgrade_costs_kV': {
        '138': 1500 * 1000,                 # $ per mile (source: VEA2022 per unit cost; rounded from $1480)
        '230': 1800 * 1000                  # $ per mile (source: PG&E per unit cost on 230 kV; rounded from $1816)
    },
    'backup_DG_cost_perMW': 1800 * 1000,    # DG installation cost per MW
                                            # ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=895482
    'backup_DG_BF': 0.5,                    # % (in fraction) of maximum demand of each node; backup factor
    'backup_DG_UF': 0.5,                    # utilization factor of backup DG
    'base_MVA': 100,                        # base MVA for the system
    'CVaR_alpha': 0.95,                     # alpha value for the CVaR in (1 - alpha)
    'line_upgrade_factor': 100,             # capacity upgrade factor in % for lines -> 100% means double the capacity
    'generator_ramp_time': 5,               # the amount of time (in minutes) that the generators can ramp the MW power
    'generator_ramp_rate': 0.02,            # fraction of generators Pmax that it can ramp per min -> MW/min.
                                            # this is approximated value and will be only used if data is unavailable
    'MCS_converged_scenarios_only': True,   # Only MCS converged scenarios are taken -> mean else [min,median,max]
    'ED_solution_check': False               # assign economic dispatch solution to the generators
}

# obtains the hardening and upgrade cost for each lines based on the kV ratings and corresponding $/mile parameter
configs['branch_cost'] = upgrade_cost(configs['test_case_id'], configs['upgrade_costs_kV'])

if not test_case_loop:
    # change the parameters here if running the model for a single instance

    configs['risk_preference'] = 0              # lambda value in the equation; defines risk preference
    configs['total_operator_budget'] = 0 * 1e9  # total budget of the system operator in billion dollars
    configs['DG_connection_points'] = 0         # number of backup DGs that can be
    configs['lines_to_harden'] = 0              # number of lines to harden
    configs['lines_to_upgrade'] = 0             # number of lines to be upgraded

else:
    # the parameters here are collected from an external text file updated for various test cases from "main.py"
    risk_preference = np.loadtxt('results/iterator/risk_num.txt', dtype=float).item()
    total_operator_budget = np.loadtxt('results/iterator/budget_num.txt', dtype=float).item()
    DG_connection_points = np.loadtxt('results/iterator/DG_num.txt', dtype=int).item()
    lines_to_harden = np.loadtxt('results/iterator/harden_num.txt', dtype=int).item()
    lines_to_upgrade = np.loadtxt('results/iterator/linecap_num.txt', dtype=int).item()

    configs['risk_preference'] = risk_preference
    configs['total_operator_budget'] = total_operator_budget * 1e9
    configs['DG_connection_points'] = DG_connection_points
    configs['lines_to_harden'] = lines_to_harden
    configs['lines_to_upgrade'] = lines_to_upgrade
