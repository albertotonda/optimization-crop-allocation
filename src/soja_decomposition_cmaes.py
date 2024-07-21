# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:45:00 2024

Attempt to perform a single-objective decomposition of a multi-objective problem,
in order to use CMA-ES, which is pretty effective for this specific case (all real values)

@author: Alberto
"""
import cma
import os
import pandas as pd
import sys

from local_utility import fitness_function

def fitness_function_cmaes(individual, args) :
    """
    Wrapper function for the real fitness function, specifically designed for
    CMA-ES
    """
    fitness_names = args["fitness_names"]
    fitness_weights = args["fitness_weights"]
    mean_soja, std_soja, total_surface = fitness_function(individual, args)
    
    # there is a special case where we have less weight values than objectives;
    # in that case, we create a dictionary to associate each objective to a weight
    weights_dictionary = {fitness_names[i] : fitness_weights[i] for i in range(0, len(fitness_names))}
    
    # add each separate weighted objective to find the global fitness value;
    # if a fitness name is not found in the dictionary, the weight will be 0.0
    fitness_value = 0.0
    
    # mean_soja has to be multiplied by -1, in order to minimize the sum
    fitness_value += weights_dictionary.get("mean_soja", 0.0) * (-1) * mean_soja 
    fitness_value += weights_dictionary.get("std_soja", 0.0) * std_soja 
    fitness_value += weights_dictionary.get("total_surface", 0.0) * total_surface
    
    return fitness_value

def generate_weights(number_of_objectives=3, parts=10) :
    """
    Generates a set of arrays of weights that always add up to 1.0
    """
    # first, find a step size that splits the range [0.0, 1.0] into the desired
    # number of parts
    step_size = 1.0 / parts
    
    weights = []
    current_weights = []
    steps = [round(step_size * i, 3) for i in range(1, parts-number_of_objectives+2)]
    depth = 1
    max_depth = number_of_objectives
    
    recursive_generate_weights(weights, current_weights, steps, depth, max_depth)
    
    return weights

def recursive_generate_weights(weights, current_weights, steps, depth, max_depth) :
    
    # if the recursive depth is equal to the maximum depth, add current weights
    # to the list of all generated weights and return
    if depth == max_depth :
        # select steps that make the sum exactly 1.0
        selected_steps = [s for s in steps if sum(current_weights) + s == 1.0]
        
        for s in selected_steps :
            cw = list(current_weights)
            cw.append(s)
            weights.append(cw)
    
    else :
        # take a look at the sum of the current weights, and select all steps
        # that would make the sum LESS THAN ONE (if max_depth is not reached)
        selected_steps = [s for s in steps if sum(current_weights) + s < 1.0]
        
        # call the recursive function, adding the step to the current weights
        for s in selected_steps :
            cw = list(current_weights)
            cw.append(s)
            
            recursive_generate_weights(weights, cw, steps, depth+1, max_depth)
        
    return

def main() :
    
    # hard-coded values
    random_seed = 42
    data_file = "../data/A_preds_eu27.csv"
    save_directory = "2024-07-20-soja-cma-es"
    results_file = os.path.join(save_directory, "results.csv")
    fitness_names = ["mean_soja", "std_soja"]
    weight_parts = 100
    
    # we are decomposing the problem into a number of separate single-objective problems
    fitness_weights = generate_weights(number_of_objectives=len(fitness_names), parts=weight_parts)
    
    max_evaluations = 1e5
    sigma0 = 1e-2
    
    individual_minimum = 0.0
    individual_maximum = 1.0
    
    # if the directory does not exist, create it
    if not os.path.exists(save_directory) :
        os.makedirs(save_directory)
    
    # prepare data and dictionary that will be used for the fitness function
    df = pd.read_csv(data_file, sep=";", decimal=",") # unfortunately in European format
    selected_columns = [c for c in df.columns if c.startswith("2")] # get year columns, like 2001, ..., 2017
    print("Data file \"%s\" seems to include predictions for years %s..." % 
                (data_file, str(selected_columns)))
    model_predictions = df[selected_columns].values
    
    # from the same file, we also take the cropland area (total) and we consider
    # the maximum usable using the hard-coded percentage defined at the beginning
    max_cropland_area = df["soybean_area"].values
    
    # get number of dimensions based on the file
    individual_size = model_predictions.shape[0] 
    
    args = {}
    args["fitness_names"] = fitness_names
    args["model_predictions"] = model_predictions
    args["max_cropland_area"] = max_cropland_area
    
    # set up cma-es
    x0 = [(individual_maximum - individual_minimum) / 2] * individual_size
    options = {
        'seed' : random_seed, 'bounds' : [individual_minimum, individual_maximum],
        'maxfevals' : max_evaluations,
               }
    
    for weights in fitness_weights :
        # weights change at each iteration, so we need to overwrite the proper
        # key in the 'args' dictionary
        args["fitness_weights"] = weights
        
        # this dictionary will be used later
        fitness_name_2_weight = {fitness_names[i] : weights[i] for i in range(0, len(fitness_names))}
        
        # instantiate and run cma-es
        print("Running experiment with weights:", fitness_name_2_weight)
        es = cma.CMAEvolutionStrategy(x0, sigma0, options)
        while not es.stop() :
            # get new batch of candidate solutions
            candidate_solutions = es.ask()
            # get the fitness value for each solution
            # TODO computation of candidate solutions *might* be performed in parallel
            es.tell(candidate_solutions, [ fitness_function_cmaes(s, args)
                                          for s in candidate_solutions])
            # print something to screen
            es.disp()
            
        # now, it would be cool to save everything to a file; let's prepare all
        # the necessary pieces
        
        # get the best solution
        best_solution = es.result[0]
        
        # get separate fitness values
        mean_soja, std_soja, total_surface = fitness_function(best_solution, args)
        
        # if results file exists, read it and get the dictionary; otherwise, create the dictionary
        results_dictionary = None
        if not os.path.exists(results_file) :
            results_dictionary_keys = list(fitness_names)
            results_dictionary_keys += [fn + "_weight" for fn in fitness_names]
            results_dictionary_keys += ["gene_%d" % i for i in range(0, individual_size)]
            
            results_dictionary = {k : [] for k in results_dictionary_keys}
            
        else :
            df_results = pd.read_csv(results_file)
            results_dictionary = df_results.to_dict(orient='list')
        
        # add the new fitness values to the dictionary
        for fn in fitness_names :
            if fn == "mean_soja" :
                results_dictionary[fn].append(mean_soja)
            elif fn == "std_soja" :
                results_dictionary[fn].append(std_soja)
            elif fn == "total_surface" :
                results_dictionary[fn].append(total_surface)
            
            # add the weight corresponding to that fitness name
            results_dictionary[fn + "_weight"].append(fitness_name_2_weight[fn])
        
        # add the values of the individual
        for i in range(0, len(best_solution)) :
            results_dictionary["gene_%d" % i].append(best_solution[i])
            
        # overwrite the dataframe as a CSV
        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(results_file, index=False)
        
if __name__ == "__main__" :
    sys.exit(main())