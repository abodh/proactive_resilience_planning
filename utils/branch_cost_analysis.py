"""
    ################################# References used in this work: ####################################################

    # GMLC_cost
    # upgrade means reconductor with advanced conductors to enhance MVA limits
    # major source (MISO): https://rb.gy/5g7nh
    # this reference is used this in this work

    # other references are from PG&E unit costs, VEA unit costs

"""

import pandas as pd


def upgrade_cost(test_case_id, upgrade_cost_kV):
    branch_data = pd.read_csv('power_system_test_cases/case' + test_case_id + '_branch.csv', delimiter=',')
    bus_data = pd.read_csv('power_system_test_cases/case' + test_case_id + '_bus.csv', delimiter=',')

    # cost for hardening is approx. 5 times the cost for overhead upgrade
    branch_data['kV'] = 0 * len(branch_data)
    branch_data['upgrade_cost'] = 0 * len(branch_data)
    branch_data['underground_cost'] = 0 * len(branch_data)

    for idx, each_branch in branch_data.iterrows():
        from_kV = bus_data[bus_data['Bus ID'] == each_branch['From Bus']]['BaseKV'].item()
        to_kV = bus_data[bus_data['Bus ID'] == each_branch['To Bus']]['BaseKV'].item()

        if from_kV == to_kV:
            branch_data.loc[idx, 'kV'] = to_kV
        else:
            branch_data.loc[idx, 'kV'] = max(from_kV, from_kV)

        if each_branch['Length'] == 0:
            # for transformer the line length is usually 0 so assuming an average line length for xfrmr sections
            each_branch['Length'] = branch_data['Length'].mean()

        # $/mile * mile
        branch_data.loc[idx, 'upgrade_cost'] = upgrade_cost_kV[str(branch_data['kV'][idx])] * each_branch['Length']

        # assuming underground is 5 times expensive
        branch_data.loc[idx, 'underground_cost'] = upgrade_cost_kV[str(branch_data['kV'][idx])] * 5 * \
                                                   each_branch['Length']

    return branch_data


if __name__ == "__main__":
    test_case_id = 'RTSGMLC'
    upgrade_cost_kV = {
                    '138': 1500 * 1000,  # $ per mile (source: VEA2022 per unit cost; rounded off from 1480)
                    '230': 1800 * 1000   # $ per mile (source: PG&E per unit cost on 230 kV; rounded off from 1816)
                    }
    branch_cost_data = upgrade_cost(test_case_id, upgrade_cost_kV)