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
import math

from logging.handlers import RotatingFileHandler
from queue import Queue
from threading import Thread, Lock

def initialize_logging(path: str, log_name: str = "", date: bool = True) -> logging.Logger :
    """
    Function that initializes the logger, opening one (DEBUG level) for a file and one (INFO level) for the screen printouts.
    """

    if date:
        log_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + log_name
    log_name = os.path.join(path, log_name + ".log")

    # create log folder if it does not exists
    if not os.path.isdir(path):
        os.mkdir(path)

    # remove old logger if it exists
    if os.path.exists(log_name):
        os.remove(log_name)

    # create an additional logger
    logger = logging.getLogger(log_name)

    # format log file
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    # the 'RotatingFileHandler' object implements a log file that is automatically limited in size
    fh = RotatingFileHandler(log_name,
                             mode='a',
                             maxBytes=100*1024*1024,
                             backupCount=2,
                             encoding=None,
                             delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Starting " + log_name + "!")

    return logger


def close_logging(logger: logging.Logger) :
    """
    Simple function that properly closes the logger, avoiding issues when the program ends.
    """

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    return


def generator(random, args) :
    """
    Generator function exploits numpy's instance of a pseudo-random number generator
    to create an array of random numbers in range (min, max) with the correct shape;
    TODO nope, when candidates are pure numpy arrays it creates several issues at
    the level of the internal comparisions, so I decided to create individuals
    as lists, and convert them to numpy arrays when needed. This introduces an
    overhead on the computations, but it is easier than redoing everything.
    """
    return [random.uniform(0.0, 0.1) for _ in range(0, args["n_dimensions"])]

def best_archiver_numpy(random, population, archive, args):
    """Archive only the best individual(s).
    
    This function is basically a copy of 'best_archiver', but it is modified
    to work with individuals that are numpy arrays.
    
    .. Arguments:
       random -- the random number generator object
       population -- the population of individuals
       archive -- the current archive of individuals
       args -- a dictionary of keyword arguments
    
    """
    new_archive = archive
    for ind in population:
        if len(new_archive) == 0:
            new_archive.append(ind)
        else:
            should_remove = []
            should_add = True
            for a in new_archive:
                #if ind.candidate == a.candidate:
                if np.array_equal(ind.candidate, a.candidate):
                    should_add = False
                    break
                elif ind < a:
                    should_add = False
                elif ind > a:
                    should_remove.append(a)
            for r in should_remove:
                new_archive.remove(r)
            if should_add:
                new_archive.append(ind)
    return new_archive

def observer(population, num_generations, num_evaluations, args) :
    """
    The observer is a classic function for inspyred, that prints out information and/or saves individuals. However, it can be easily re-used by other
    evolutionary approaches.
    """
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
    fitness_list[index] = fitness_function(individual, args) # TODO put your evaluation function here, also maybe add logger and thread_id 
    thread_lock.release()

    logger.debug("[Thread %d] Evaluation finished." % thread_id)

    return

def fitness_function(individual, args) : 
    """
    This is the fitness function. It should be replaced by the 'true' fitness function to be optimized.
    """
    # load data
    model_predictions = args["model_predictions"]
    max_cropland_area = args["max_cropland_area"]
    
    # convert individual to a more maneagable numpy array
    individual_numpy = np.array(individual)

    # first fitness function is the total soja produced over the years;
    # second fitness function is the standard deviation inter-year;
    # for this reason, it's better to first compute the year-by-year production
    production_by_year = np.zeros((model_predictions.shape[1],))
    
    for year in range(0, model_predictions.shape[1]) :
        
        # select column of data corresponding to a year
        model_predictions_year = model_predictions[:, year]
        
        # multiply, element-wise, each element of the candidate solution with
        # the predicted production for the corresponding square for that year
        production_by_year[year] = np.sum(np.multiply(individual_numpy, model_predictions_year))
    
    # now that we have the production by year, we can easily compute the first
    # and second fitness values
    mean_soja = np.mean(production_by_year)
    std_soja = np.std(production_by_year)
    
    # third fitness function is easy: it's just a sum of the surfaces used in
    # the candidate solution, so a sum of the values in the single individual
    #total_surface = np.sum(individual)
    
    # actually, we now have a better way of computing the total surface used by
    # a candidate solution; since we have the maximum cropland area for each pixel,
    # we can just use the sum of an element-wise multiplication between the
    # candidate solution and the array containing the maximum cropland area per pixel
    total_surface = np.sum(np.multiply(individual_numpy, max_cropland_area))
    
    # numerically, we use a negative value for the mean soja produced per year,
    # in order to transform the problem into a minimization problem
    #fitness_values = inspyred.ec.emo.Pareto([-1 * mean_soja, std_soja, total_surface])

    return (-1 * mean_soja, std_soja, total_surface)

def fitness_function_stddev(individual, args) : 
    """
    Fitness function restricted to stddev
    """
    # load data
    model_predictions = args["model_predictions"]
    max_cropland_area = args["max_cropland_area"]
    
    # convert individual to a more maneagable numpy array
    individual_numpy = np.array(individual)

    # first fitness function is the total soja produced over the years;
    # second fitness function is the standard deviation inter-year;
    # for this reason, it's better to first compute the year-by-year production
    production_by_year = np.zeros((model_predictions.shape[1],))
    
    for year in range(0, model_predictions.shape[1]) :
        
        # select column of data corresponding to a year
        model_predictions_year = model_predictions[:, year]
        
        # multiply, element-wise, each element of the candidate solution with
        # the predicted production for the corresponding square for that year
        production_by_year[year] = np.sum(np.multiply(individual_numpy, model_predictions_year))
    
    std_soja = np.std(production_by_year)
    
    return std_soja

def main() :
    
    # unfortunately, there are some values that we need to put hard-coded
    # here, because I did not find a good way of putting them in the files
    max_cropland_area_percentage_usable = 0.2 # this is the maximum surface usable, 20% of all cropland in a pixel

    # options for logging and saving files
    overwrite_save_files = False

    # cplex LP format or Bensolve?
    use_cplex = False
    # cplex can generate the quadratic objective or not
    use_stddev = True
    # again, cplex only: weights of the objectives
    w_area = 3
    w_prod = 1
    w_stdd = 0.01
    
    # relevant variables are stored in a dictionary, to ensure compatibility with inspyred
    args = dict()

    # hard-coded values
    #args["data_file"] = "../data/pred_2000_2017_avg.m.csv"
    #args["data_file"] = "../data/soybean_pred_2000_2023_avg.m_s20.csv"
    #args["data_file"] = "../data/soybean_pred_2000_2023_pca.m.2_new_5perc.csv"
    #args["data_file"] = "../data/soybean_pred_2000_2023_pca.m.2_new_1perc_eu27.csv"
    args["data_file"] = "../data/B_preds_eu27.csv"
    args["log_directory"] = "2024-01-26-soja-allocation-2-objectives"
    args["save_directory"] = args["log_directory"]
    #args["population_file_name"] = "population"
    #args["save_at_every_iteration"] = True # save the whole population at every iteration
    #args["random_seeds"] = [42] # list of random seeds, because we might want to run the evolutionary algorithm in a loop 
    #args["n_threads"] = 60 # TODO change number of threads 
    #fitness_names = ["mean_soja", "std_soja", "total_surface"]

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
    args["model_predictions"] = model_predictions

    # from the same file, we also take the cropland area (total) and we consider
    # the maximum usable using the hard-coded percentage defined at the beginning
    max_cropland_area = df["soybean_area"].values # actually, no, everything is already included
    args["max_cropland_area"] = max_cropland_area
    
    # compute maximum theoretical production, sum everything and divide by year
    max_theoretical_soja = np.sum(model_predictions) / model_predictions.shape[1]
    # also, the number of dimensions in the problem is equal to the number
    # of squares available in Europe
    n_dimensions = model_predictions.shape[0]
    n_years = model_predictions.shape[1]

    #priority: production + variance before 3 objectives
    if use_cplex:
        # first, normalize weights
        if use_stddev:
            w_tot = w_area+w_prod+w_stdd
        else:
            w_tot = w_area+w_prod
        # rw = "real weight"
        rw_area = w_area / w_tot
        rw_prod = w_prod / w_tot
        if use_stddev:
            rw_stdd = w_stdd / w_tot
            f_scaled = open('3obj.lp', 'w')
        else:
            f_scaled = open('2obj.lp', 'w')
            
        f_area = open('obj-area.lp', 'w')
        f_prod = open('obj-prod.lp', 'w')
            
        if use_stddev:
            f_stdd = open('obj-stdd.lp', 'w')
            logger.info("3 objectives. Normalized Weights: area %s production %s stddev %s" % (rw_area, rw_prod, rw_stdd))
            files = [f_area, f_prod, f_stdd]
        else:
            logger.info("2 linear objectives. Normalized Weights: area %s production %s" % (rw_area, rw_prod))
            files = [f_area, f_prod]
        all_files = files[:]
        all_files.append(f_scaled)
            
        mtx = pd.DataFrame(0.0, range(n_dimensions), range(n_dimensions))

        for f in all_files:
            f.write ("Minimize\nobj:\n")

        # std deviation coefficients
        if use_stddev:
            f_scaled.write ("[ ")
            f_stdd.write("[ ")
            # x^2
            coeff = pd.Series(0.0, range(n_dimensions))
            first = True
            for i in range(0, n_dimensions):
                ind = pd.Series(0.0, range(n_dimensions))
                ind[i] = 1
                #(f1, f2, f3) = fitness_function(ind, args)
                f2 = fitness_function_stddev(ind, args)
                c = 2.0*f2*f2
                cscaled = c*rw_stdd
                if first:
                    f_scaled.write("%s x_%s^2\n" % (cscaled, i))
                    f_stdd.write("%s x_%s^2\n" % (c, i))
                    first = False
                else:
                    f_scaled.write("+ %s x_%s^2\n" % (cscaled, i))
                    f_stdd.write("+ %s x_%s^2\n" % (c, i))
                    
                coeff[i] = f2*f2

            # xy
            for i in range(0, n_dimensions):
                for j in range(i+1, n_dimensions):
                    ind = pd.Series(0.0, range(n_dimensions))
                    ind[i] = 1
                    ind[j] = 1
                    #(f1, f2, f3) = fitness_function(ind, args)
                    f2 = fitness_function_stddev(ind, args)
                    c = (f2*f2 - coeff[i] - coeff[j])
                    cscaled = rw_stdd * c
                    if c > 0:
                        f_scaled.write("+ %s x_%s * x_%s\n" % (cscaled, i, j))
                        f_stdd.write("+ %s x_%s * x_%s\n" % (c, i, j))
                    elif c < 0:
                        f_scaled.write("%s x_%s * x_%s\n" % (cscaled, i, j))
                        f_stdd.write("%s x_%s * x_%s\n" % (c, i, j))
            f_scaled.write("]/2\n")
            f_stdd.write("]/2\n")

        first_lin = True
        if use_stddev:
            first_lin = False
        first_area = True
        # -production+area coefficients. Note that production is already negated in fitness_function()
        for i in range(0, n_dimensions):
            ind = pd.Series(0.0, range(n_dimensions))
            ind[i] = 1
            (f1, f2, f3) = fitness_function(ind, args)
            c = rw_area*f3 + rw_prod*f1
            if not first_area:
                f_area.write("+ ")
            else:
                first_area = False
            f_area.write("%s x_%s\n" % (f3, i))
            f_prod.write("%s x_%s\n" % (f1, i))
            if c > 0:
                if first_lin:
                    f_scaled.write ("%s x_%s\n" % (c, i))
                    first_lin = False
                else:
                    f_scaled.write("+ %s x_%s\n" %(c, i))
            elif c < 0:
                f_scaled.write ("%s x_%s\n" % (c, i))
                first_lin = False

        for f in all_files:
            f.write ("Bounds\n")
            for i in range(0, n_dimensions):
                f.write ("0 <= x_%s <= 1\n" % (i))
            f.write("End\n")

        f_area.close()
        f_prod.close()
        if use_stddev:
            f_stdd.close()
    else:
        # p vlp DIR ROWS COLS NZ OBJ OBJNZ
        print("p vlp min 0 %s 0 2 %s" % (n_dimensions, 2*n_dimensions))
    
        for i in range(0, n_dimensions):
            print("o 1 %s %s" % (i+1, -model_predictions.mean(1)[i]))
            print("o 2 %s %s" % (i+1, max_cropland_area[i]))

        for i in range(0, n_dimensions):
            #print("j %s d 0 %s" % (i+1, max_cropland_area_percentage_usable))
            print("j %s d 0 %s" % (i+1, 1.0))
        
        print("e")
        
    # save final archive (Pareto front) to csv
    #save_population_to_csv(pareto_front, ea.num_generations, os.path.join(args["save_directory"], "final-pareto-front.csv"), fitness_names)
    
    # close logger
    close_logging(logger)

    return

if __name__ == "__main__" :
    sys.exit( main() )
