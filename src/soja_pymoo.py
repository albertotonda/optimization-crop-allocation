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
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions 

# local package, containing the fitness function that returns the three separate
# objectives
from local_utility import fitness_function

# the function is defined as an instance of a subclass of the Problem class
class MultiObjectiveSojaProblem(Problem) :
    
    fitness_functions = []
    fitness_function_args = {}
    max_soja_production = 0.0
    max_std_soja = 0.0
    max_total_surface = 0.0
    normalize_objectives = True
    
    def __init__(self, n_variables, fitness_functions, model_predictions, max_cropland_area, 
                 max_soja_production=0.0, max_std_soja=0.0, max_total_surface=0.0,
                 normalize_objectives=True) :
        # xl and xu are the lower and upper bounds for each variable
        super().__init__(n_var=n_variables, n_obj=len(fitness_functions), xl=0.0, xu=1.0)
        # also keep track of the fitness functions, used later during the evaluation
        self.fitness_functions = fitness_functions
        self.fitness_function_args["model_predictions"] = model_predictions
        self.fitness_function_args["max_cropland_area"] = max_cropland_area
        self.max_soja_production = max_soja_production
        self.max_std_soja = max_std_soja
        self.max_cropland_area = max_total_surface
        
    def _evaluate(self, x, out, *args, **kwargs) :
        # this evaluation function starts from the assumption that 'x' is actually
        # an array containing all individuals; so we can shape the fitness values
        # numpy array accordingly
        fitness_values = np.zeros((x.shape[0], 1, self.n_obj))
        
        # run the evaluation
        for i in range(0, x.shape[0]) :
            mean_soja, std_soja, total_surface = fitness_function(x[i], self.fitness_function_args)
            
            # fitness values list
            x_fitness_values = []
            
            if self.normalize_objectives == True :
                # this is a variant where the values of the objectives are internally
                # normalized, so that maybe NSGA-II will give them the same importance
                if "mean_soja" in self.fitness_functions :
                    normalized_mean_soja = (self.max_soja_production - mean_soja) / self.max_soja_production
                    x_fitness_values.append(normalized_mean_soja)
                if "std_soja" in self.fitness_functions :
                    normalized_std_soja = std_soja / self.max_std_soja
                    x_fitness_values.append(normalized_std_soja)
                if "total_surface" in self.fitness_functions :
                    normalized_total_surface = total_surface / self.max_cropland_area
                    x_fitness_values.append(normalized_total_surface)
            else :
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
    
# this is a class inheriting from Callback, to save the population every few generations;
# the algorithm will be set so that the Callback method 'notify' is invoked at the end
# of every generation
class SavePopulationCallback(Callback) :
    
    folder = ""
    population_file_name = ""
    generational_interval = 100
    
    # class constructor
    def __init__(self, folder, population_file_name, generational_interval=100,
                 fitness_names=None, overwrite_file=False) :
        super().__init__()
        self.folder = folder
        self.population_file_name = population_file_name
        self.generational_interval = generational_interval
        self.fitness_names = fitness_names
        
    # this method is called at every iteration of the algorithm
    def notify(self, algorithm) :
        
        # get the current generation and other information
        generation = algorithm.n_gen
        
        if generation % self.generational_interval == 0 :
            X = algorithm.pop.get("X")
            F = algorithm.pop.get("F")
            
            # check: if fitness_names has not been specified, set it up
            if self.fitness_names is None :
                self.fitness_names = ["fitness_%d" % (i+1) for i in range(0, F.shape[1])]
            
            results_dictionary = dict()
            results_dictionary["generation"] = [generation] * X.shape[0]
            for i in range(0, F.shape[1]) :
                results_dictionary[ self.fitness_names[i] ] = F[:,i]
            for i in range(0, X.shape[1]) :
                results_dictionary["variable_%d" % i] = X[:,i]
            df_results = pd.DataFrame.from_dict(results_dictionary)
            df_results.to_csv(os.path.join(self.folder, self.population_file_name + "-%d.csv" % generation), index=False)
            
        
def main() :
    
    # hard-coded values
    random_seed = 42
    data_file = "../data/A_preds_eu42.csv"
    results_base_file_name = "results.csv"
    fitness_names = ["mean_soja", "std_soja", "total_surface"]
    fitness_names = ["mean_soja", "std_soja"]
    n_objectives = len(fitness_names)
    algorithm_class = NSGA2 # it can also be NSGA2
    seed_initial_population = True
    
    # hyperparameters for MOEA/D and NSGA2
    population_size = 1000
    max_generations = 100000
    n_partitions = 1000
    n_neighbors = 15
    prob_neighbor_mating = 0.7
    
    # this is a more proper termination check, with several conditions; it stops
    # when one of the conditions is satisfied
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=max_generations,
        n_max_evals=max_generations * population_size
        )
    
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
    max_soja_production, max_std_soja, max_total_surface = fitness_function(max_soja_production_individual,
                                    {"model_predictions" : model_predictions,
                                     "max_cropland_area" : max_cropland_area})
    
    # initialize the problem instance with all the information gathered so far
    problem = MultiObjectiveSojaProblem(n_variables, fitness_names, 
                                        model_predictions, max_cropland_area, 
                                        max_soja_production=max_soja_production,
                                        max_std_soja=max_std_soja,
                                        max_total_surface=max_total_surface,
                                        )
    
    # prepare an instance of the Callback class that will be used to save the population
    callback = SavePopulationCallback(results_folder, "population-generation",
                                      generational_interval=10, fitness_names=fitness_names)
    
    # prepare the initial population; if population seeding has been specified,
    # proceed accordingly
    np.random.seed(seed=random_seed)
    initial_population = np.random.random_sample(size=(population_size, n_variables))
    if seed_initial_population == True :
        initial_population[0] = max_soja_production_individual
        initial_population[1] = np.zeros((n_variables,)) # min surface and standard deviation
    
    # set up the algorithm, depending on the class selected
    algorithm = None
    if algorithm_class is MOEAD :
        # set up the algorithm, with the reference directions
        ref_dirs = get_reference_directions("uniform", n_objectives, n_partitions=n_partitions)
        algorithm = algorithm_class(ref_dirs, n_neighbors=n_neighbors, 
                                    prob_neighbor_mating=prob_neighbor_mating,
                                    sampling=initial_population)
    
    elif algorithm_class is NSGA2 :
        algorithm = algorithm_class(pop_size=population_size, sampling=initial_population)
    
    # start the run
    result = minimize(problem, 
                      algorithm, 
                      #termination, # computing the convex hull takes too much time 
                      ('n_gen', max_generations), 
                      callback=callback,
                      seed=random_seed, 
                      verbose=True)
    
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
            
    # let's also create a results file with a specific name, to remember the settings
    results_file_name = datetime.datetime.now().strftime("%Y-%m-%d-") 
    results_file_name += "-".join(fitness_names) + "-" + algorithm_class.__name__ + "-"
    if algorithm_class is MOEAD :
        results_file_name += "partitions%d-neighbors%d-probmating%.2f-" % (n_partitions, n_neighbors, prob_neighbor_mating)
    elif algorithm_class is NSGA2 :
        results_file_name += "mu%d-" % (population_size)
    results_file_name += results_base_file_name
    
    df_results = pd.DataFrame.from_dict(results_dictionary)
    df_results.to_csv(os.path.join(results_folder, results_file_name), index=False)
    
    return

if __name__ == "__main__" :
    sys.exit(main())