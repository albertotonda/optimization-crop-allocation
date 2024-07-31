# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:43:49 2024

An attempt at using pymoo's MOEA/D and NSGA2 implementation to tackle the problem.

@author: Alberto
"""
import datetime
import numpy as np
import os
import pandas as pd
import sys

# imports from pymoo
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions 

# local package, containing the fitness function that returns the three separate
# objectives
from local_utility import fitness_function

# the function is defined as an instance of a subclass of the Problem class
class MultiObjectiveSojaProblem(Problem) :
    
    fitness_functions = []
    fitness_function_args = {}
    max_soja_production = 0.0
    
    def __init__(self, n_variables, fitness_functions, model_predictions, max_cropland_area, max_soja_production=0.0) :
        # xl and xu are the lower and upper bounds for each variable
        super().__init__(n_var=n_variables, n_obj=len(fitness_functions), xl=0.0, xu=1.0)
        # also keep track of the fitness functions, used later during the evaluation
        self.fitness_functions = fitness_functions
        self.fitness_function_args["model_predictions"] = model_predictions
        self.fitness_function_args["max_cropland_area"] = max_cropland_area
        self.max_soja_production = max_soja_production
        
    def _evaluate(self, x, out, *args, **kwargs) :
        # this evaluation function starts from the assumption that 'x' is actually
        # an array containing all individuals; so we can shape the fitness values
        # numpy array accordingly
        fitness_values = np.zeros((x.shape[0], 1, self.n_obj))
        
        # run the evaluation
        for i in range(0, x.shape[0]) :
            mean_soja, std_soja, total_surface = fitness_function(x[i], self.fitness_function_args)
            
            # fitness function list
            x_fitness_values = []
            if "mean_soja" in self.fitness_functions :
                # one way of turning the problem of maximizing yield into a minimization
                # problem is by turning the result negative;
                # x_fitness_values.append(-1 * mean_soja)
                # another way is to minimize the difference between the maximum
                # production available and the current production
                x_fitness_values.append(self.max_soja_production - mean_soja)
            if "std_soja" in self.fitness_functions :
                x_fitness_values.append(std_soja)
            if "total_surface" in self.fitness_functions :
                x_fitness_values.append(total_surface)
                
            # convert list to vector, place it in the appropriate place
            fitness_values[i,0,:] = np.array(x_fitness_values)
            
        # place the appropriate result in the 'out' dictionary
        out["F"] = fitness_values
        
        return
        

def main() :
    
    # hard-coded values
    random_seed = 42
    data_file = "../data/A_preds_eu27.csv"
    results_file = "results.csv"
    fitness_names = ["mean_soja", "std_soja", "total_surface"]
    fitness_names = ["mean_soja", "std_soja"]
    n_objectives = len(fitness_names)
    algorithm_class = MOEAD # it can also be NSGA2
    
    # hyperparameters for MOEA/D
    population_size = 500
    max_generations = 100
    n_partitions = 1000
    n_neighbors = 15
    prob_neighbor_mating = 0.7
    
    # create folder for the results
    results_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + "-".join(fitness_names) + "-" + algorithm_class.__name__
    print("Saving results to folder \"%s\"..." % results_folder)
    
    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
        
    # initialize the problem instance, reading the file and getting the information
    df = pd.read_csv(data_file, sep=";", decimal=",") # unfortunately in European format
    selected_columns = [c for c in df.columns if c.startswith("2")] # get year columns, like 2001, ..., 2017
    print("Data file \"%s\" seems to include predictions for years %s..." % 
                (data_file, str(selected_columns)))
    n_variables = df.shape[0] # one variable per row
    model_predictions = df[selected_columns].values
    max_cropland_area = df["soybean_area"].values
    
    # it is also interesting to know the maximum possible production; there are
    # multiple ways of knowing this, but the quick-and-dirty one is to just initialize
    # an invididual to all [1.0, 1.0,..., 1.0], and run a fitness evaluation
    max_soja_production_individual = np.ones((n_variables,))
    max_soja_production, std_soja, total_surface = fitness_function(max_soja_production_individual,
                                    {"model_predictions" : model_predictions,
                                     "max_cropland_area" : max_cropland_area})
    
    problem = MultiObjectiveSojaProblem(n_variables, fitness_names, 
                                        model_predictions, max_cropland_area, 
                                        max_soja_production)
    
    algorithm = None
    if algorithm_class is MOEAD :
        # set up the algorithm, with the reference directions
        ref_dirs = get_reference_directions("uniform", n_objectives, n_partitions=n_partitions)
        algorithm = algorithm_class(ref_dirs, n_neighbors=n_neighbors, prob_neighbor_mating=prob_neighbor_mating)
    
    elif algorithm_class is NSGA2 :
        algorithm = algorithm_class(pop_size=population_size)
    
    # start the run
    result = minimize(problem, algorithm, ('n_gen', max_generations), seed=random_seed, verbose=True)
    
    # do something with the results! ideally, it would be nice to save them
    # result.X contains the genotype of the final non-dominated solutions
    # result.F contains the corresponding objective values; however, the
    # objective values for this problem (especially mean_soja) have been manipulated
    # to transform the problem into a minimization problem; so, it might be worth it
    # to re-evaluate all individuals using the basic fitness function and save
    # the raw values of the fitness
    results_dictionary = {"max_generation" : [] }
    for f in fitness_names :
        results_dictionary[f] = np.zeros((result.X.shape[0],))
    for i in range(0, n_variables) :
        results_dictionary["gene_%d" % i] = result.X[:,i]
    
    # re-evaluate all individuals
    print("Re-evaluating all individuals before saving everything to file...")
    for i in range(0, result.X.shape[0]) :
        results_dictionary["max_generation"].append(max_generations)
        
        mean_soja, std_soja, total_surface = fitness_function(result.X[i], 
                                                              {"model_predictions" : model_predictions,
                                                               "max_cropland_area" : max_cropland_area})
        if "mean_soja" in results_dictionary :
            results_dictionary["mean_soja"][i] = mean_soja
        if "std_soja" in results_dictionary :
            results_dictionary["std_soja"][i] = std_soja
        if "total_surface" in results_dictionary :
            results_dictionary["total_surface"][i] = total_surface
    
    df_results = pd.DataFrame.from_dict(results_dictionary)
    df_results.to_csv(os.path.join(results_folder, results_file), index=False)
    
    return

if __name__ == "__main__" :
    sys.exit(main())