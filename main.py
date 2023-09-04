import os
import numpy as np
from time import time
import pdb
import pandas as pd

'''
The model can be run independently from the terminal. Use this main file if you want to loop through multiple test
cases.
'''
test_case_loop = True


def planning_simulation():
    # for now please modify this to manage the arguments to be sent to the solver or the pyomo model
    os.system("runef -m grid_investment.py --traceback --output-solver-log --solver=gurobi --solve "
              "--solution-writer=utils\csvsolutionwriter")
    pass


if __name__ == "__main__":

    risk_preference = [0, 0.5, 0.95]
    operator_budget_count = [0.5, 1, 2, 3]
    DG_counts = [0, 1, 5, 10]
    line_hardening_counts = [0, 1, 5, 10]
    line_capacity_upgrade_counts = [0, 1, 5, 10]

    iter_counter = 0
    execution_time_record = {}

    for risk in risk_preference:
        np.savetxt("results/iterator/risk_num.txt", [risk], delimiter=',', fmt='%.2f')

        for budget in operator_budget_count:
            np.savetxt("results/iterator/budget_num.txt", [budget], delimiter=',', fmt='%.2f')

            for DG_num in DG_counts:
                np.savetxt("results/iterator/DG_num.txt", [DG_num], delimiter=',', fmt='%i')

                for harden_num in line_hardening_counts:
                    np.savetxt("results/iterator/harden_num.txt", [harden_num], delimiter=',', fmt='%i')

                    for linecap_num in line_capacity_upgrade_counts:
                        np.savetxt("results/iterator/linecap_num.txt", [linecap_num], delimiter=',', fmt='%i')

                        # this condition avoids 0 resource case except for one fully risk-neutral (expected) case
                        # and one fully risk-averse case
                        if (DG_num == 0 and harden_num == 0 and linecap_num == 0) and (budget > 0.5 or
                                                                                       (0 < risk < 0.95)):
                            continue

                        # start timer
                        start_time = time()

                        # RUN THE PLANNING SIMULATION FOR ALL OF THE CASES
                        planning_simulation()

                        # end timer
                        end_time = time()

                        elapsed_time = end_time - start_time
                        execution_time_record[iter_counter] = round(elapsed_time, 2)

                        iter_counter += 1
                        pdb.set_trace()

    execution_time_df = pd.DataFrame(execution_time_record.items(), columns=['iteration', 'execution_time(s)'])
    execution_time_df.to_csv('results/total_execution_time.csv', delimiter=',')
