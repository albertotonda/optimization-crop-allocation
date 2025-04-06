# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 15:56:24 2025

This script computes the hypervolumes of the Pareto fronts found by the different
approaches, in order to compare them for the different scenarios.

@author: Alberto
"""
import numpy as np
import os
import pandas as pd

from pymoo.indicators.hv import HV
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__" :
    
    # hard-coded values
    data_file = "../data/hypervolumes-pareto-fronts/opti_eu27.csv"
    # file with metadata is not really used by the script, but it's important
    # to know that it exists, as it contains the description of the features
    #metadata_file = "../data/hypervolumes-pareto-fronts/metadata_opti_eu27.xlsx"
    results_file = "hypervolume_comparison.csv"
    
    # column names for Pareto front objectives
    objective_columns = ["mean_soja", "std_soja", "total_surface"]
    # objectives from the type of experiment
    experiment2objectives = {'ps' : ["mean_soja", "total_surface"],
                             'pv' : ["mean_soja", "std_soja"],
                             'pvs' : ["mean_soja", "std_soja", "total_surface"]}
    
    # data structure to store results
    results = {'experiment' : [], 'surface_constraint' : [], 
               'HV_lin' : [], 'HV_nsga' : []}
    
    # load data
    df = pd.read_csv(data_file)
    
    # types of experiments
    for experiment, objectives in experiment2objectives.items() :
        
        df_experiment = df[df["opti_criteria"] == experiment]
        constraints = df_experiment["surface_constraint"].unique()
        
        for constraint in constraints :
        
            print("Now considering experiment \"%s\" and surface constraint \"%s\"..." % (experiment, constraint))
            
            # select all relevant data
            df_selected = df_experiment[df_experiment["surface_constraint"] == constraint]
            
            # generate the Pareto fronts for the two different algorithms;
            # 'mean_soja' needs to be converted to a minimization objective
            df_lin = df_experiment[df_experiment["algo"] == "lin"]
            pf_lin = df_lin[objectives].values
            pf_lin[:,0] = -1 * pf_lin[:,0] # 'mean_soja' is always objective 0
            
            df_nsga = df_experiment[df_experiment["algo"] == "nsga"]
            pf_nsga = df_nsga[objectives].values
            pf_nsga[:,0] = -1 * pf_nsga[:,0]
            
            # now, normalize everything
            scaler = MinMaxScaler()
            scaler.fit(df_experiment[objectives].values)
            pf_lin_norm = scaler.transform(pf_lin)
            pf_nsga_norm = scaler.transform(pf_nsga)
            
            # compute hypervolumes; first, we need to compute the reference point
            ref_point = []
            for objective_index, objective in enumerate(objectives) :
                max_objective = max(max(pf_lin_norm[:, objective_index]), max(pf_nsga_norm[:,objective_index]))
                ref_point.append(max_objective)
            ref_point = np.array(ref_point)
            
            # then, instantiate the object and perform the proper computation
            hv_experiment = HV(ref_point=ref_point)
            hv_lin = hv_experiment(pf_lin_norm)
            hv_nsga = hv_experiment(pf_nsga_norm)
            
            # update results
            results['experiment'].append(experiment)
            results['surface_constraint'].append(constraint)
            results['HV_lin'].append(hv_lin)
            results['HV_nsga'].append(hv_nsga)
            
    # at the end of the loop for each experiment, save results to table
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(results_file, index=False)
            