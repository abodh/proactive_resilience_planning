#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# major update from Abodh on creating a pandas df and store them in separate excel sheets of same file

import pyomo.common.plugin
from pysp import solutionwriter
from pysp.scenariotree.tree_structure import \
    ScenarioTree

import pdb
import pandas as pd
from optim_config import configs
import numpy as np


#
# a simple utility to munge the index name into something a bit more csv-friendly and in general more readable.
# At the current time, we just eliminate any leading and trailing parentheses and change commas to colons - the latter
# because it's a csv file!
#

def index_to_string(index):
    result = str(index)
    result = result.lstrip('(').rstrip(')')
    result = result.replace(',', ':')
    result = result.replace(' ', '')

    return result


def write_solution_in_excel(scenario_tree, output_file_prefix):
    """
    Write the solution to a excel file with customized sheets.
    Args: scenario_tree: a scenario tree object populated with a solution.
          output_file_prefix: a string to indicate the file names for output.
                              output_file_prefix + ".csv"
                              ef if extensive form
    """

    if not isinstance(scenario_tree, ScenarioTree):
        raise RuntimeError(
            "CSVSolutionWriter write method expects "
            "ScenarioTree object - type of supplied "
            "object=" + str(type(scenario_tree)))

    solution_filename = output_file_prefix + \
                        str(configs['risk_preference']) + \
                        str(configs['total_operator_budget']/1e9) + ".xlsx"

    results_df = []
    for stage in scenario_tree.stages:
        for tree_node in sorted(stage.nodes, key=lambda x: x.name):
            for variable_id in sorted(tree_node._variable_ids):
                results_dict = {}
                var_name, index = \
                    tree_node._variable_ids[variable_id]
                results_dict['stage'] = stage.name
                results_dict['tree_node'] = tree_node.name
                results_dict['var_name'] = var_name
                results_dict['var_idx'] = index_to_string(index)
                results_dict['var_value'] = tree_node._solution[variable_id]
                results_df.append(results_dict)

    final_results_df = pd.DataFrame(results_df)

    # use excel writer object to display result in different sheets
    excel_writer = pd.ExcelWriter(f"results/{solution_filename}", engine='xlsxwriter')
    final_results_df.to_excel(excel_writer, sheet_name=f"{str(configs['DG_connection_points'])}-"
                                                       f"{str(configs['lines_to_harden'])}-"
                                                       f"{str(configs['lines_to_upgrade'])}")
    excel_writer.close()

    print("Scenario tree solution written to file=" + solution_filename)


class CSVSolutionWriter(pyomo.common.plugin.SingletonPlugin):
    pyomo.common.plugin.implements(
        solutionwriter.ISolutionWriterExtension)

    def write(self, scenario_tree, output_file_prefix):
        write_solution_in_excel(scenario_tree, output_file_prefix)
