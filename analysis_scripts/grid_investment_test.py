# import libraries
import networkx
from pyomo.environ import *
import numpy as np
import scipy.io as readmat
import pdb
import math
import os

# get the current working directory
dir_file = os.getcwd()

# testcase to use for the optimization
test_case = '39'

# degree to radian conversion factor
deg_to_rad = math.pi / 180

# extract the scenarios and their probabilities from external file
scenario_file = dir_file + r"/scenario_files/case" + test_case + "/failure_scenarios.mat"
probability_file = dir_file + r"/scenario_files/scenario_probabilities_49.csv"

# create a concrete pyomo model
model = ConcreteModel()

# load matpower test case in mat format
matpower_mat_file = readmat.loadmat(dir_file + '/power_system_test_cases/case' + test_case + '.mat',
                                    struct_as_record=False,
                                    squeeze_me=True)
# ensure that all the saved mat file are saved under workspace var name 'matpower_testcase'
test_case = matpower_mat_file['matpower_testcase']

# load the bus, generation, and line data
model.bus = test_case.bus
model.line = test_case.branch
model.gen = test_case.gen
model.gen_cost = test_case.gencost

# initialize the parameters
model.nEdges = len(model.line)  # total number of edges
model.nNodes = len(model.bus)  # total number of nodes
model.nGen = len(model.gen)  # total number of generators

# gen dataset also has specific MVA base; use that instead when updating this version
model.Pbase = 100  # MVA base 100 MVA to convert the system to pu system

# let us assume that infinite capacity is equal to 100 GW
Inf_transfer_Pmax = 10e6
Pmax_line = 10e6 / model.Pbase

# it is feasible to keep the max generation of each generator to be the max capacity of all generators
Gmax = max(model.gen[:, 8]) / model.Pbase
Gmin = min(model.gen[:, 9]) / model.Pbase

# max load is for load shed that can be maximum at that point
max_load = max(model.bus[:, 2]) / model.Pbase
max_demand = sum(model.bus[:, 2]) / model.Pbase

# number of lines to harden(in other words asking the question: how many lines can be hardened with current investment?)
model.nHarden = 0

# variable ranges
model.branch_ij = range(0, model.nEdges)  # branch variable range
model.bus_i = range(0, model.nNodes)  # buses variable range
model.gen_i = range(0, model.nGen)  # generators variable range

###############################################################################################################
####################################### Variables #############################################################
###############################################################################################################

# declaring pyomo variables

# first we declare steady state variables i.e. power flow needs to be maintained while in the steady state
# since these should be the same for all scenarios, they are first stage variables
# they have ss at the end for representation

# although bounds are mentioned here to maintain the standard, they will be redefined as per gen bus
# bus generation variable in steady state operation before an outage
model.bus_gen_ss = Var(model.bus_i, model.gen_i, bounds=(Gmin, Gmax), within=NonNegativeReals, initialize=0)

# active power flowing through each lines in steady state operation before an outage
model.Pij_ss = Var(model.branch_ij, bounds=(-Pmax_line, Pmax_line), within=Reals, initialize=0)

# angle of each bus in steady state operation before an outage
model.theta_ss = Var(model.bus_i, bounds=(-2 * math.pi, 2 * math.pi), within=Reals, initialize=0)

# planning decision which is a first-stage investment decision variable
model.harden = Var(model.branch_ij, bounds=(0, 1), within=Binary, initialize=0)  # connectivity of each line

# next we declare scenario-dependent variables
# notice that some of these variables are similar to the steady state dispatch variables
# bus generation variable
model.bus_gen = Var(model.bus_i, model.gen_i, bounds=(Gmin, Gmax), within=NonNegativeReals, initialize=0)

# active power flowing through each lines for each scenario
model.Pij = Var(model.branch_ij, bounds=(-Pmax_line, Pmax_line), within=Reals, initialize=0)

# angle of each bus for each scenario
model.theta = Var(model.bus_i, bounds=(-2 * math.pi, 2 * math.pi), within=Reals, initialize=0)

# edge connectivity status
model.xij = Var(model.branch_ij, bounds=(0, 1), within=Binary, initialize=0)

# variable indicating the load shed in each of the buses
model.load_shed = Var(model.bus_i, bounds=(0, max_load), within=NonNegativeReals, initialize=0)
# calculates total load shed for each scenario
model.load_shed_value = Var(bounds=(0, max_demand), within=NonNegativeReals, initialize=0)

# ramp time of time_for_ramp minutes is selected here
time_for_ramp = 5

# ramp rate assumption i.e. it is assumed that the generators can provide a ramp of ramp_rate% of their capacity/minute
ramp_rate = 0.02

# variable indicating the generation curtailment of each gen in the gen buses
model.gen_curtailment = Var(model.bus_i, model.gen_i, bounds=(0, Gmax), within=NonNegativeReals, initialize=0)

# mutable parameter to introduce branch outage in the model [0: out of service]
# this is introduced as a scenario
model.outage_status = Param(model.branch_ij, mutable=True, initialize=1, within=Binary)

###############################################################################################################
####################################### Constraints ###########################################################
###############################################################################################################

# pyomo constraints
# creates a list of constraints as placeholders
#################### bus power balance constraints ############################
model.power_balance_ss = ConstraintList()

# bus data col 3: active power demand, col 5: shunt conductance
for bus_num_idx in range(model.nNodes):
    bus_num = model.bus[bus_num_idx, 0]

    # identify the list of generators connected to each bus
    gens = np.where(model.gen[:, 0] == bus_num)[0].tolist()
    to_bus_list = np.where(model.line[:, 1] == bus_num)[0].tolist()
    from_bus_list = np.where(model.line[:, 0] == bus_num)[0].tolist()

    model.power_balance_ss.add(sum(model.bus_gen_ss[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num]
                                   for gen_num in gens) +
                               sum(model.Pij_ss[to_bus] for to_bus in to_bus_list) -
                               sum(model.Pij_ss[from_bus] for from_bus in from_bus_list) ==
                               model.bus[bus_num_idx, 2] / model.Pbase + model.bus[bus_num_idx, 4] / model.Pbase)

################## generator power limit constraint ###########################
model.gen_limit_ss = ConstraintList()
# generator should generate power between its min and max active power limit
# col 9: PMAX and col 10: PMIN (Note: in Python number starts from 0)
for gen_num in range(model.nGen):
    model.gen_limit_ss.add(model.bus_gen_ss[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] <=
                           model.gen[gen_num, 8] / model.Pbase)
    model.gen_limit_ss.add(model.bus_gen_ss[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] >=
                           model.gen[gen_num, 9] / model.Pbase)

# make sure non-generating bus do not generate anything
for bus_num_idx in range(model.nNodes):
    bus_num = model.bus[bus_num_idx, 0]

    if not np.any(np.equal(model.gen[:, 0], bus_num)):
        for gen_idx in range(model.nGen):
            model.gen_limit_ss.add(model.bus_gen_ss[bus_num_idx, gen_idx] == 0)

####################### active power flow constraint on each line ################
model.power_flow_ss = ConstraintList()
'''
Note: in Python number starts from 0

linedata:
col 4: reactance (X)
col 9: transformer tap ratio
col 10: transformer phase shift (in degrees)

busdata:
col 9: voltage angle (in degrees) -> this is a variable here so no need to use as parameter
'''

for line_num in range(model.nEdges):

    # MATPOWER keeps 0 for transmission lines without transformer
    # here we need to ensure tap ratio for transmission line is 1
    if model.line[line_num, 8] == 0:
        model.line[line_num, 8] = 1

    reciprocal_term = 1 / (model.line[line_num, 3] * model.line[line_num, 8])
    model.power_flow_ss.add(model.Pij_ss[line_num] ==
                            reciprocal_term * (model.theta_ss[np.where(model.bus[:, 0] ==
                                                                       model.line[line_num, 0])[0][0]] -
                                               model.theta_ss[np.where(model.bus[:, 0] ==
                                                                       model.line[line_num, 1])[0][0]] -
                                               (model.line[line_num, 9] * deg_to_rad)))

################### thermal limit (MVA_limits) ############################
# since the flow can be bi-directional, limits range from neg to positive value
# col 6: max MVA limit of the line (0 means unlimited capacity)
# this constraint tightens the -inf, inf bound set during variable initialization
for line_num in range(model.nEdges):
    if model.line[line_num, 5] == 0:
        model.line[line_num, 5] = Pmax_line
    model.power_flow_ss.add(model.Pij_ss[line_num] <= model.line[line_num, 5] / model.Pbase)
    model.power_flow_ss.add(model.Pij_ss[line_num] >= - model.line[line_num, 5] / model.Pbase)

################### angle difference between two buses on each line ################
model.angle_limit_ss = ConstraintList()
# from bus and to bus reference is obtained via line
# col 12: min angle difference (degree), col 13: max angle difference (degree)
for angle_num in range(model.nEdges):
    model.angle_limit_ss.add((model.theta_ss[np.where(model.bus[:, 0] == model.line[angle_num, 0])[0][0]] -
                              model.theta_ss[np.where(model.bus[:, 0] == model.line[angle_num, 1])[0][0]])
                             <= model.line[angle_num, 12] * deg_to_rad)
    model.angle_limit_ss.add((model.theta_ss[np.where(model.bus[:, 0] == model.line[angle_num, 0])[0][0]] -
                              model.theta_ss[np.where(model.bus[:, 0] == model.line[angle_num, 1])[0][0]])
                             >= model.line[angle_num, 11] * deg_to_rad)

# the angle can be anywhere from -2pi to 2pi hence we need to maintain 0 angle at reference (slack) bus

# identifying slack bus
slack_bus = np.where(model.bus[:, 1] == 3)[0][0]

# ensure the angle at reference bus is 0
model.angle_limit_ss.add(model.theta_ss[slack_bus] == 0)

###################################### SECOND STAGE CONSTRAINTS #############################################
# now the constraints will go for the second stage which includes the investment variables

#################### bus power balance constraints ############################
model.power_balance = ConstraintList()

# bus data col 3: active power demand, col 5: shunt conductance
for bus_num_idx in range(model.nNodes):
    bus_num = model.bus[bus_num_idx, 0]

    # identify the list of generators connected to each bus
    gens = np.where(model.gen[:, 0] == bus_num)[0].tolist()
    to_bus_list = np.where(model.line[:, 1] == bus_num)[0].tolist()
    from_bus_list = np.where(model.line[:, 0] == bus_num)[0].tolist()

    model.power_balance.add(sum(model.bus_gen[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] -
                                model.gen_curtailment[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num]
                                for gen_num in gens) +
                            sum(model.Pij[to_bus] for to_bus in to_bus_list) -
                            sum(model.Pij[from_bus] for from_bus in from_bus_list) ==
                            model.bus[bus_num_idx, 2] / model.Pbase + model.bus[bus_num_idx, 4] / model.Pbase -
                            model.load_shed[bus_num_idx])

################## generator power limit constraint ###########################
model.gen_limit = ConstraintList()
# generator should generate power between its min and max active power limit
# col 9: PMAX and col 10: PMIN (Note: in Python number starts from 0)
for gen_num in range(model.nGen):
    model.gen_limit.add(model.bus_gen[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] <=
                        model.gen[gen_num, 8] / model.Pbase)
    model.gen_limit.add(model.bus_gen[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] >=
                        model.gen[gen_num, 9] / model.Pbase)

# make sure non-generating bus do not generate anything and do not have any curtailment
for bus_num_idx in range(model.nNodes):
    bus_num = model.bus[bus_num_idx, 0]

    if not np.any(np.equal(model.gen[:, 0], bus_num)):
        for gen_idx in range(model.nGen):
            model.gen_limit.add(model.bus_gen[bus_num_idx, gen_idx] == 0)
            model.gen_limit.add(model.gen_curtailment[bus_num_idx, gen_idx] == 0)

####################### active power flow constraint on each line ################
model.power_flow = ConstraintList()
model.line_connectivity = ConstraintList()
'''
Note: in Python number starts from 0

linedata:
col 4: reactance (X)
col 9: transformer tap ratio
col 10: transformer phase shift (in degrees)

busdata:
col 9: voltage angle (in degrees) -> this is a variable here so no need to use as parameter
'''

# check the connectivity status of lines based on the outage and hardening decision
'''
outages_scenario    hardening_decision      connectivity
       0                    0                    0
       0                    1                    1
       1                    0                    1
       1                    1                    1    
'''
for line_num in range(model.nEdges):
    model.line_connectivity.add(model.xij[line_num] == model.outage_status[line_num] + model.harden[line_num] -
                                model.outage_status[line_num] * model.harden[line_num])

for line_num in range(model.nEdges):

    # MATPOWER keeps 0 for transmission lines without transformer
    # here we need to ensure tap ratio for transmission line is 1
    if model.line[line_num, 8] == 0:
        model.line[line_num, 8] = 1

    reciprocal_term = 1 / (model.line[line_num, 3] * model.line[line_num, 8])

    ''' to maintain power balance, line flow must equal to B*angle diff (in general). however, when connectivity is 0 
    this balance is disrupted intentionally so as to ensure this balance constraint is not activated. Hence, when the 
    line is connected i.e., xij = 1 then former power balance equation is automatically introduced
    '''
    model.power_flow.add(model.Pij[line_num] * (1 / reciprocal_term) -
                         (model.theta[np.where(model.bus[:, 0] == model.line[line_num, 0])[0][0]] -
                          model.theta[np.where(model.bus[:, 0] == model.line[line_num, 1])[0][0]] -
                          (model.line[line_num, 9] * deg_to_rad)) <= (1 - model.xij[line_num]) * Pmax_line)

    model.power_flow.add(model.Pij[line_num] * (1 / reciprocal_term) -
                         (model.theta[np.where(model.bus[:, 0] == model.line[line_num, 0])[0][0]] -
                          model.theta[np.where(model.bus[:, 0] == model.line[line_num, 1])[0][0]] -
                          (model.line[line_num, 9] * deg_to_rad)) >= -(1 - model.xij[line_num]) * Pmax_line)

################### thermal limit (MVA_limits) ############################
# since the flow can be bi-directional, limits range from neg to positive value
# col 6: max MVA limit of the line (0 means unlimited capacity)
# this constraint tightens the -inf, inf bound set during variable initialization
for line_num in range(model.nEdges):
    if model.line[line_num, 5] == 0:
        model.line[line_num, 5] = Pmax_line

    # line flow is 0 if the line is disconnected due to outage
    model.power_flow.add(model.Pij[line_num] <= model.xij[line_num] * model.line[line_num, 5] / model.Pbase)
    model.power_flow.add(model.Pij[line_num] >= - model.xij[line_num] * model.line[line_num, 5] / model.Pbase)

################### angle difference between two buses on each line ################
model.angle_limit = ConstraintList()
# from bus and to bus reference is obtained via line
# col 12: min angle difference (degree), col 13: max angle difference (degree)
for angle_num in range(model.nEdges):
    # Pmax_line is used as a bigM here and the end term ensures that angle differences are not constrained when the
    # corresponding line connecting those buses are disconnected
    model.angle_limit.add((model.theta[np.where(model.bus[:, 0] == model.line[angle_num, 0])[0][0]] -
                           model.theta[np.where(model.bus[:, 0] == model.line[angle_num, 1])[0][0]])
                          <= model.line[angle_num, 12] * deg_to_rad + Pmax_line * (1 - model.xij[angle_num]))
    model.angle_limit.add((model.theta[np.where(model.bus[:, 0] == model.line[angle_num, 0])[0][0]] -
                           model.theta[np.where(model.bus[:, 0] == model.line[angle_num, 1])[0][0]])
                          >= model.line[angle_num, 11] * deg_to_rad - Pmax_line * (1 - model.xij[angle_num]))

# the angle can be anywhere from -2pi to 2pi hence we need to maintain 0 angle at reference (slack) bus

# identifying slack bus
slack_bus = np.where(model.bus[:, 1] == 3)[0][0]

# ensure the angle at reference bus is 0
model.angle_limit.add(model.theta[slack_bus] == 0)

################################ investment budget constraint ################################
model.investment = ConstraintList()
model.investment.add(sum(model.harden[harden_lines] for harden_lines in range(model.nEdges)) <= model.nHarden)

###################### max load shedding constraint  ############################################
# for each bus load shed cannot be more than the load at that bus
model.load_shed_limit = ConstraintList()
for load_bus_num in range(model.nNodes):
    model.load_shed_limit.add(model.load_shed[load_bus_num] >= 0)
    model.load_shed_limit.add(model.load_shed[load_bus_num] <= model.bus[load_bus_num, 2]/model.Pbase)

######################### ramp rate limit #####################################################
# for each gen bus and for each scenario, generators should ramp up or down to provide excess generation required
model.ramp_rate = ConstraintList()

for gen_num in range(model.nGen):
    if model.gen[gen_num, 16] == 0:
        # if no ramp rate data is given ramp rate (in per min) is obtained from their capacity
        model.gen[gen_num, 16] = ramp_rate * (model.gen[gen_num, 8]/model.Pbase) * time_for_ramp

    model.ramp_rate.add(model.bus_gen_ss[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] -
                        model.bus_gen[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num]
                        <= (model.gen[gen_num, 16] / model.Pbase) * time_for_ramp)
    model.ramp_rate.add(model.bus_gen_ss[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] -
                        model.bus_gen[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num]
                        >= - (model.gen[gen_num, 16] / model.Pbase) * time_for_ramp)

############################## generator curtailment to balance the load ###################################
# if the load on the particular system is more than generation then the generators should ramp down
model.generation_curtailment_limit = ConstraintList()

for gen_num in range(model.nGen):
    # curtailment can be maximum generation curtailment for that specific scenario
    model.generation_curtailment_limit.add \
        (model.gen_curtailment[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] <=
         model.bus_gen[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num])

    # there is no such thing as minimum curtailment so it does not follow min generation limit
    model.generation_curtailment_limit.add \
        (model.gen_curtailment[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num] >= 0)

# ######################### constraint to just observe the results of the variables ###################
def load_shed_value(model):
    return model.load_shed_value - (sum(model.load_shed[bus_num] for bus_num in range(model.nNodes))) == 0


model.compute_load_shed = Constraint(rule=load_shed_value)


# use these constraints to fix the first stage dispatch if the solution is known
model.gen_fix = ConstraintList()

# # this is DC-OPF solution for 39 bus
# model.gen_fix.add(model.bus_gen_ss[29] == 6.6084)
# model.gen_fix.add(model.bus_gen_ss[30] == 6.4600)
# model.gen_fix.add(model.bus_gen_ss[31] == 6.6084)
# model.gen_fix.add(model.bus_gen_ss[32] == 6.5200)
# model.gen_fix.add(model.bus_gen_ss[33] == 5.0800)
# model.gen_fix.add(model.bus_gen_ss[34] == 6.6084)
# model.gen_fix.add(model.bus_gen_ss[35] == 5.8000)
# model.gen_fix.add(model.bus_gen_ss[36] == 5.6400)
# model.gen_fix.add(model.bus_gen_ss[37] == 6.6084)

'''
for some "weird" reason fixing all bus gen variables made the problem infeasible so lets fix n-1 variables and 
let the solver decide the value of final generator. It does not hamper our objective as the final value will be 
close to what we want 
'''
# model.gen_fix.add(model.bus_gen_ss[38] == 10.9450)


#########################################################################################################
######################################   Define Objective   #############################################
#########################################################################################################


############################ Stochastic Investment Planning Objective #################
def ComputeFirstStageCost_rule(model):
    expr = 0
    return expr


model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)


def ComputeSecondStageCost_rule(model):
    # this is minimized for each scenario
    # minimize the sum of load shedding the generation curtailment
    expr = sum(model.load_shed[bus_num_idx] for bus_num_idx in range(model.nNodes)) + \
           0.01*sum(model.gen_curtailment[np.where(model.bus[:, 0] == model.gen[gen_num, 0])[0][0], gen_num]
               for gen_num in range(model.nGen))

    return expr


model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)


# the total cost is the sum of first stage cost and second stage cost
def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost


model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

#########################################################################################################
######################################   Stochastic Data  ###############################################
#########################################################################################################

# define dictionary to store the scenarios
Fault = {}
prob_value = {}

# name_tag for the scenario -> generally named as "Scenario"
string = "Scenario"

# load the fault_indices
fault_idx = readmat.loadmat(scenario_file)['fail_scenario']

# load the individual probabilities
prob = np.loadtxt(probability_file, delimiter=',', dtype=float)
num_scenarios = prob.__len__()

# the module does not work if sum(probability) != 1 so anyhow make it 1
while abs(1 - sum(prob)) > 0.00001:
    prob = prob + ((1 - sum(prob)) / num_scenarios)

prob[int(np.floor(prob.__len__() / 3))] = prob[int(np.floor(prob.__len__() / 3))] + (1 - sum(prob))

# convert matlab indices to python indices
fault_idx = fault_idx - 1

for i in range(0, num_scenarios):
    try:
        Fault[string + str(i + 1)] = set(fault_idx[0][i].squeeze().tolist())
    except:
        Fault[string + str(i + 1)] = set(fault_idx[0][i][0].tolist())

    prob_value[string + str(i + 1)] = prob[i]

print('\n ######## SCENARIOS LOCKED AND LOADED ############ \n')


#########################################################################################################
###############################   Scenario Callback Function  ###########################################
#########################################################################################################

def pysp_instance_creation_callback(scenario_name, node_names):
    # callback function that calls scenarios with the above-created names

    # cloning the model for each scenarios
    instance = model.clone()

    for faulty in Fault[scenario_name]:
        # inserting the fault in the system for each scenarios
        instance.outage_status[faulty] = 0
    return instance


#########################################################################################################
###############################   Scenario Tree Generation ##############################################
#########################################################################################################

def pysp_scenario_tree_model_callback():
    # Return a NetworkX scenario tree.
    # this avoids creating any dat file
    g = networkx.DiGraph()

    first_stage_var = ["bus_gen_ss[*,*]",
                       "harden[*]",
                       "Pij_ss[*]",
                       "theta_ss[*]"]

    second_stage_var = ["bus_gen[*,*]",
                        "Pij[*]",
                        "theta[*]",
                        "xij[*]",
                        "load_shed[*]",
                        "gen_curtailment[*,*]",
                        "load_shed_value"]

    # for the root node
    g.add_node("Root",
               cost="FirstStageCost",
               variables=first_stage_var,
               derived_variables=[])

    # for other stages
    for key, item in Fault.items():
        g.add_node(key,
                   cost="SecondStageCost",
                   variables=second_stage_var,
                   derived_variables=[])
        g.add_edge("Root", key, weight=prob_value[key].item())

    return g
