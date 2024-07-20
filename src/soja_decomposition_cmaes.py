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
    fitness_weights = args["fitness_weights"]
    mean_soja, std_soja, total_surface = fitness_function(individual, args)
    
    # mean_soja has to be multiplied by -1, in order to minimize the sum
    fitness_value = fitness_weights[0] * (-1) * mean_soja + fitness_weights[1] * std_soja + fitness_weights[2] * total_surface
    
    return fitness_value

def generate_weights(number_of_objectives=3, parts=10) :
    """
    Generates a set of arrays of weights that always add up to 1.0
    """
    weights = []
    step = 1.0 / parts
    maximum = parts - number_of_objectives + 2
    
    for i in range(1, maximum) :
        current_weights = [i * step]
        for j in range(1, maximum - i) :
            current_weights.append(j * step)
            current_weights.append(1.0 -i * step -j * step)
    
            weights.append(current_weights)
            current_weights = [i * step]
        
    return weights

def main() :
    
    # hard-coded values
    random_seed = 42
    data_file = "../data/A_preds_eu27.csv"
    save_directory = "2024-07-20-soja-cma-es"
    fitness_names = ["mean_soja", "std_soja"]
    fitness_weights = [0.1, 0.0, 0.0]
    
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
    args["fitness_weights"] = fitness_weights
    args["model_predictions"] = model_predictions
    args["max_cropland_area"] = max_cropland_area
    
    # set up cma-es
    x0 = [(individual_maximum - individual_minimum) / 2] * individual_size
    sigma0 = 5e-2
    options = {'seed' : random_seed, 'bounds' : [individual_minimum, individual_maximum]}
    
    # instantiate and run cma-es
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
        
        # TODO after a certain number of iterations, print the current best individual
        
    # get the best solution out and print it
    best_solution = es.result[0]
    print(best_solution)
        
if __name__ == "__main__" :
    sys.exit(main())