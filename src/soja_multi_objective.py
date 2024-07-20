# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:26:49 2023

Script for the multi-objective optimization of crop allocation in Europe.

Each variable to be optimized is a real in [0, 0.2], describing the percentage
of land use for the crop. We have one variable per "pixel" (unitary surface) on
the map.

Objectives:
    - Maximize mean forecasted amount of soja
    - Minimize variance in the forecasted amount of soja
    - Minimize total surface occupied by soja

@author: Alberto
"""

import datetime
import inspyred
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import sys
import traceback

from threading import Lock

# local library with utility functions, refactoring to clean up the code a bit
from local_utility import ThreadPool, fitness_function, initialize_logging, close_logging


def generator(random, args) :
    """
    Generator function creates an array of random numbers in range (min, max) 
    with the correct shape; individuals are created as lists, and converted 
    to numpy arrays when needed. This introduces an overhead on the computations,
    but it is easier.
    """
    return [random.uniform(0.0, 1.0) for _ in range(0, args["n_dimensions"])]

@inspyred.ec.variators.crossover
def variator_with_strength(random, parent1, parent2, args) :
    """
    The basic idea of this variator is to select one between multiple possible
    other variators (e.g. mutation or cross-over) and then apply some 
    """
    # get all parameters related to the mutation
    mutation_strength = args["strength"]
    mutation_mean = args["gaussian_mean"]
    mutation_std = args["gaussian_std"]
    bounder = args["_ec"].bounder
    
    # invoke classic n-point crossover, probability is evaluated inside
    children = inspyred.ec.variators.n_point_crossover(random, [parent1, parent2], args)
    
    # and now, modified mutation with strength
    for child in children :
        while random.random() < mutation_strength :
            # pick random element, add Gaussian mutation
            index = random.choice(range(0, len(child)))
            child[index] += random.gauss(mutation_mean, mutation_std)
        child = bounder(child, args)
    
    return children

def observer(population, num_generations, num_evaluations, args) :
    """
    The observer is a classic function for inspyred, that prints out information and/or saves individuals. 
    However, it can be easily re-used by other evolutionary approaches.
    In this particular case, we are going to test a self-adapting of the mutation
    strength.
    """
    # self-adapting: multiplying everything by a value < 1.0, slowly reducing
    # both the mutation rate and the mean of the Gaussian mutation
    #decay = args["decay"]
    #args["mutation_rate"] *= decay
    #args["gaussian_mean"] *= decay
    
    # logging and saving the population
    logger = args["logger"]
    save_directory = args["save_directory"]
    save_at_every_iteration = args["save_at_every_iteration"]
    fitness_names = args.get("fitness_names", None)

    best_individual = best_fitness = None

    best_individual = population[0].candidate
    best_fitness = population[0].fitness

    # 'best fitness' does not mean much here; let's try to find the best value for each one
    # this is useful to know if they actually improve over time
    best_values = []
    for i in range(0, len(best_fitness)) :
        best_value = None
        for individual in population :
            if best_value is None or individual.fitness[i] < best_value :
                best_value = individual.fitness[i]
        best_values.append(best_value)

    # some output
    best_values_string = "; ".join(["%.4e" % bv for bv in best_values])
    logger.info("Generation %d (%d evaluations), best fitness value found for each fitness function: %s" % (num_generations, num_evaluations, best_values_string))

    # depending on a flag we set, we either save the population at each generation separately, or we overwrite the file each time;
    # the first solution allows us to track the development of the optimization, the second saves disk space
    population_file_name = "%d-%s-generation-%d.csv" % (args["random_seed"], args["population_file_name"], num_generations)
    if args["overwrite_save_files"] == True :
        population_file_name = "%d-%s-last-generation.csv" % (args["random_seed"], args["population_file_name"])
    logger.info("Saving population file as \"%s\"..." % population_file_name)

    population_file_name = os.path.join(save_directory, population_file_name)
    save_population_to_csv(population, num_generations, population_file_name, fitness_names)

    # save archive (current Pareto front)
    archive_file_name = "%d-archive-generation-%d.csv" % (args["random_seed"], num_generations)
    if args["overwrite_save_files"] == True :
        archive_file_name = "%d-archive.csv" % args["random_seed"]
    logger.info("Saving archive file as \"%s\"..." % archive_file_name)

    archive = args["_ec"].archive # key "_ec" corresponds to the current instance of the evolutionary algorithm
    save_population_to_csv(archive, num_generations, os.path.join(save_directory, archive_file_name), fitness_names)

    return

def save_population_to_csv(population, num_generations, population_file_name, fitness_names) :

    best_individual = population[0].candidate
    best_fitness = population[0].fitness

    # create dictionary
    dictionary_df_keys = ["generation"]
    if fitness_names is None :
        dictionary_df_keys += ["fitness_value_%d" % i for i in range(0, len(best_fitness))]
    else :
        dictionary_df_keys += fitness_names
    dictionary_df_keys += ["gene_%d" % i for i in range(0, len(best_individual))]
    
    dictionary_df = { k : [] for k in dictionary_df_keys }

    # check the different cases
    for individual in population :

        dictionary_df["generation"].append(num_generations)
        
        for i in range(0, len(best_fitness)) :
            key = "fitness_value_%d" % i
            if fitness_names is not None :
                key = fitness_names[i]
            dictionary_df[key].append(individual.fitness.values[i])
        
        for i in range(0, len(individual.candidate)) :
            dictionary_df["gene_%d" % i].append(individual.candidate[i])

    # conver dictionary to DataFrame, save as CSV
    df = pd.DataFrame.from_dict(dictionary_df)
    df.to_csv(population_file_name, index=False)

    return

def multi_thread_evaluator(candidates, args) :
    """
    Wrapper function for multi-thread evaluation of the fitness.
    """

    # get logger from the args
    logger = args["logger"]
    n_threads = args["n_threads"]

    # create list of fitness values, for each individual to be evaluated
    # initially set to 0.0 (setting it to None is also possible)
    fitness_list = [0.0] * len(candidates)

    # create Lock object and initialize thread pool
    thread_lock = Lock()
    #thread_pool = ThreadPool(n_threads) 
    thread_pool = args["thread_pool"]

    # create list of arguments for threads
    arguments = [ (candidates[i], args, i, fitness_list, thread_lock) for i in range(0, len(candidates)) ]
    # queue function and arguments for the thread pool
    thread_pool.map(evaluate_individual, arguments)

    # wait the completion of all threads
    logger.debug("Starting multi-threaded evaluation...")
    thread_pool.wait_completion()

    return fitness_list

def evaluate_individual(individual, args, index, fitness_list, thread_lock, thread_id) :
    """
    Wrapper function for individual evaluation, to be run inside a thread.
    """

    logger = args["logger"]

    logger.debug("[Thread %d] Starting evaluation..." % thread_id)

    # thread_lock is a threading.Lock object used for synchronization and avoiding
    # writing on the same resource from multiple threads at the same time
    thread_lock.acquire()
    fitness_list[index] = fitness_function_inspyred(individual, args) # TODO put your evaluation function here, also maybe add logger and thread_id 
    thread_lock.release()

    logger.debug("[Thread %d] Evaluation finished." % thread_id)

    return

def fitness_function_inspyred(individual, args) :
    """
    This is the fitness function for inspyred. It's a wrapper because I decided
    to de-couple the fitness computation from the algorithm, in order to re-use
    the same fitness computation for different algorithms, without changing
    the core computations.
    """
    # if no fitness name has been specified, we default to using all three objectives
    fitness_names = args.get("fitness_names", ["mean_soja", "std_soja", "total_surface"])
    
    mean_soja, std_soja, total_surface = fitness_function(individual, args)
    
    # check the fitness names that are actually requested
    fitness_values_list = []
    for fitness_name in fitness_names :
        if fitness_name == "mean_soja" :
            # numerically, we use a negative value for the mean soja produced per year,
            # in order to transform the problem into a minimization problem
            fitness_values_list.append(-1.0 * mean_soja)
        elif fitness_name == "std_soja" :
            # this is already correct for minimization, so we just use it as it is
            fitness_values_list.append(std_soja)
        elif fitness_name == "total_surface" :
            # same goes for this one
            fitness_values_list.append(total_surface)
    
    # prepare values in the format that is correct for inspyred, a Pareto object
    # initialized with a list of values
    fitness_values = inspyred.ec.emo.Pareto(fitness_values_list)
    
    return fitness_values

def main() :
    
    # unfortunately, there are some values that we need to put hard-coded
    # here, because I did not find a good way of putting them in the files

    # there are a lot of moving parts inside an EA, so some modifications will still need to be performed by hand
    # a few hard-coded values, to be changed depending on the problem
    population_size = int(1e2)
    offspring_size = int(2e2)
    max_evaluations = int(1e4)
    max_generations = int(max_evaluations/offspring_size) + 1
    tournament_selection_size = int(0.02 * population_size)
    
    # TODO add here some extra option for the evolutionary operators
    
    # options for logging and saving files; setting overwrite to True will result
    # in a single file with the individuals at the end of the run; setting it to
    # False will create a separate file with population and archive at each generation
    overwrite_save_files = False
    
    # relevant variables are stored in a dictionary, to ensure compatibility with inspyred
    args = dict()

    # hard-coded values
    #args["data_file"] = "../data/pred_2000_2017_avg.m.csv"
    #args["data_file"] = "../data/soybean_pred_2000_2023_avg.m_s20.csv"
    #args["data_file"] = "../data/soybean_pred_2000_2023_pca.m.2_new_1perc_eu27.csv"
    args["data_file"] = "../data/A_preds_eu27.csv"
    args["log_directory"] = "2024-07-20-soja-allocation-2-objectives-mean-std-eu27"
    args["save_directory"] = args["log_directory"]
    args["population_file_name"] = "population"
    args["save_at_every_iteration"] = True # save the whole population at every iteration
    args["random_seeds"] = [42] # list of random seeds, because we might want to run the evolutionary algorithm in a loop 
    args["n_threads"] = 60 # TODO change number of threads 
    fitness_names = ["mean_soja", "std_soja"]#, "total_surface"]

    # initialize logging, using a logger that smartly manages disk occupation
    logger = initialize_logging(args["log_directory"])

    # also save pointer to the logger into the dictionary
    args["logger"] = logger

    # start program
    logger.info("Hi, I am a program, starting now!")
    logger.debug(type(logger))
    
    # load the matrix with all the data; the data file is supposed to contain a
    # value for each square in Europe, for each year considered
    df = pd.read_csv(args["data_file"], sep=";", decimal=",") # unfortunately in European format
    selected_columns = [c for c in df.columns if c.startswith("2")] # get year columns, like 2001, ..., 2017
    logger.info("Data file \"%s\" seems to include predictions for years %s..." % 
                (args["data_file"], str(selected_columns)))
    model_predictions = df[selected_columns].values
    
    # from the same file, we also take the cropland area (total) and we consider
    # the maximum usable using the hard-coded percentage defined at the beginning
    max_cropland_area = df["soybean_area"].values # actually, no, everything is already included

    # compute maximum theoretical production, sum everything and divide by year
    max_theoretical_soja = np.sum(model_predictions) / model_predictions.shape[1]
    # also, the number of dimensions in the problem is equal to the number
    # of squares available in Europe
    n_dimensions = model_predictions.shape[0]

    # also, let's initialize the ThreadPool here, so that we just pass it to the functions later
    # and we do not need to create a new one every time we start the evaluations
    thread_pool = ThreadPool(args["n_threads"])
    
    # TODO it could be interesting to add two "extreme" individuals to the initial
    # population: one trivial solution with all parameters at 0.0, and one with all
    # parameters at 0.2; however, we should try this AFTER regular experiments
    seed_no_surface = [0.0] * n_dimensions
    seed_all_surface = [1.0] * n_dimensions
    #seeds = [seed_no_surface, seed_all_surface]

    # start a series of experiments, for each random seed
    for random_seed in args["random_seeds"] :

        logger.info("Starting experiment with random seed %d..." % random_seed)
        args["random_seed"] = random_seed

        # initalization of ALL random number generators, to try and ensure repatability
        prng = random.Random(random_seed)
        nprng = np.random.default_rng(seed=random_seed) # this might become deprecated, and creating a dedicated numpy pseudo-random number generator instance would be better 
        
        # create an instance of EvolutionaryComputation (generic EA) and set up its parameters
        # define all parts of the evolutionary algorithm (mutation, selection, etc., including observer)
        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.nonuniform_mutation]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = observer
        ea.logger = args["logger"]

        # printout with the experimental parameters
        #logger.info("Experimental hyperparameters of NSGA-II: population size=%d, offspring size=%d, stop condition after %d generations, tournament selection size=%d, mutation rate=%.4f, mutation mean=%.4f, mutation stdev=%.4f, crossover rate=%.4f" % 
        #            (population_size, offspring_size, max_generations, tournament_selection_size, mutation_rate, mutation_mean, mutation_stdev, crossover_rate))

        final_population = ea.evolve(
                                generator=generator,
                                evaluator=multi_thread_evaluator,
                                pop_size=population_size,
                                num_selected=offspring_size,
                                maximize=False,
                                bounder=inspyred.ec.Bounder(0.0, 1.0),
                                max_generations=max_generations,
                                
                                # parameters of the tournament selection
                                tournament_size = tournament_selection_size,
                                # parameters of the Gaussian mutation
                                #mutation_rate = mutation_rate, # applied as an element-by-element basis
                                #gaussian_mean = mutation_mean,
                                #gaussian_std = mutation_stdev, # default was 1
                                #crossover_rate = crossover_rate,
                                # self-adapting
                                #decay = decay,
                                #strength = 0.9,
                                
                                # seeding: adding handcrafted individuals to the initial population
                                #seeds = seeds,

                                # all items below this line go into the 'args' dictionary passed to each function
                                logger = args["logger"],
                                n_dimensions = n_dimensions,
                                n_threads = args["n_threads"],
                                population_file_name = args["population_file_name"],
                                random_seed = args["random_seed"],
                                save_directory = args["save_directory"],
                                save_at_every_iteration = args["save_at_every_iteration"],
                                overwrite_save_files = overwrite_save_files,
                                fitness_names = fitness_names,
                                nprng = nprng,
                                thread_pool = thread_pool,
                                
                                # used to compute the fitness
                                model_predictions = model_predictions,
                                max_cropland_area = max_cropland_area,
                                max_theoretical_soja = max_theoretical_soja,
                                )

    logger.info("Evolution terminated!")

    # extract the Pareto front and plot it
    pareto_front = ea.archive
    
    mean_soja_values = std_soja_values = total_surface_values = None
    for fitness_name in fitness_names :
        if fitness_name == "mean_soja" :
            mean_soja_values = [individual.fitness[0] for individual in pareto_front]
        elif fitness_name == "std_soja" :
            std_soja_values = [individual.fitness[1] for individual in pareto_front]
        elif fitness_name == "total_surface" :
            total_surface_values = [individual.fitness[2] for individual in pareto_front]
    
    sns.set_style('darkgrid')
    
    if mean_soja_values is not None and std_soja_values is not None and total_surface_values is not None :
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(mean_soja_values, std_soja_values, total_surface_values, alpha=0.7)
        ax.set_title("Pareto front in 3D")
        ax.set_xlabel(fitness_names[0])
        ax.set_ylabel(fitness_names[1])
        ax.set_zlabel(fitness_names[2])
        plt.savefig(os.path.join(args["save_directory"], "pareto-front-3d.png"), dpi=300)
        plt.show()
        plt.close(fig)
    
    else :
        
        fitness_0_values = fitness_1_values = None
        if "mean_soja" in fitness_names :
            fitness_0_values = mean_soja_values
        else :
            fitness_0_values = std_soja_values
            
        if "total_surface" in fitness_names :
            fitness_1_values = total_surface_values
        else :
            fitness_1_values = std_soja_values
        
        # also plot 2D projections of the plots
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        sns.scatterplot(x=fitness_0_values, y=fitness_1_values, alpha=0.7)
        ax.set_title("2D projection of the Pareto front")
        ax.set_xlabel(fitness_names[0])
        ax.set_ylabel(fitness_names[1])
        # we also change the labels for the 'x' axis, trying to go back to the actual production
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        #new_labels = ["%.2e" % (max_theoretical_soja - float(label)) for label in labels]
        #ax.set_xticklabels(new_labels)
        plt.savefig(os.path.join(args["save_directory"], "pareto-front-%s-%s.png" % (fitness_names[0], fitness_names[1])), dpi=300)
        plt.close(fig)

    # save final archive (Pareto front) to csv
    save_population_to_csv(pareto_front, ea.num_generations, os.path.join(args["save_directory"], "final-pareto-front.csv"), fitness_names)
    
    # close logger
    close_logging(logger)

    return

if __name__ == "__main__" :
    sys.exit( main() )
