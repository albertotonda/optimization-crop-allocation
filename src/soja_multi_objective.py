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

from logging.handlers import RotatingFileHandler
from queue import Queue
from threading import Thread, Lock

class Worker(Thread):
    """
    Thread executing tasks from a given tasks queue. 
    """
    def __init__(self, tasks, thread_id, logger=None):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.id = thread_id
        self.logger = logger
        self.start()

    def run(self):
        while True:
            # extract arguments and organize them properly
            func, args, kargs = self.tasks.get()
            if self.logger :
                self.logger.debug("[Thread %d] Args retrieved: \"%s\"" % (self.id, args))
            new_args = []
            if self.logger :
                self.logger.debug("[Thread %d] Length of args: %d" % (self.id, len(args)))
            for a in args[0]:
                new_args.append(a)
            new_args.append(self.id)
            if self.logger :
                self.logger.debug("[Thread %d] Length of new_args: %d" % (self.id, len(new_args)))
            try:
                # call the function with the arguments previously extracted
                func(*new_args, **kargs)
            except Exception as e:
                # an exception happened in this thread
                if self.logger :
                    self.logger.error(traceback.format_exc())
                else :
                    print(traceback.format_exc())
            finally:
                # mark this task as done, whether an exception happened or not
                if self.logger :
                    self.logger.debug("[Thread %d] Task completed." % self.id)
                self.tasks.task_done()

        return

class ThreadPool:
    """
    Pool of threads consuming tasks from a queue.
    """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for i in range(num_threads):
            Worker(self.tasks, i)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))
        return

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)
        return

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()
        return


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
    #return args["nprng"].uniform(low=0.0, high=0.2, size=(args["n_dimensions"],))
    return [random.uniform(0.0, 0.2) for _ in range(0, args["n_dimensions"])]

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
    archive_file_name = "%d-gen-%d-archive.csv" % (args["random_seed"], num_generations)
    if args["overwrite_save_files"] == True :
        archive_file_name = "%d-archive.csv" % args["random_seed"]
    logger.info("Saving archive file as \"%s\"..." % archive_file_name)

    archive = args["_ec"].archive # key "_ec" corresponds to the current instance of the evolutionary algorithm
    save_population_to_csv(archive, num_generations, os.path.join(save_directory, archive_file_name), fitness_names)

    # old code
    # save the whole population to file
    #if save_at_every_iteration :

        ## create file name, with information on random seed and population
        #population_file_name = os.path.join(save_directory, population_file_name)
        #logger.debug("Saving population file to \"%s\"..." % population_file_name)

        # create dictionary
        #dictionary_df_keys = ["generation"]
        #if fitness_names is None :
        #    dictionary_df_keys += ["fitness_value_%d" % i for i in range(0, len(best_fitness))]
        #else :
        #    dictionary_df_keys += fitness_names
        #dictionary_df_keys += ["gene_%d" % i for i in range(0, len(best_individual))]
        #
        #dictionary_df = { k : [] for k in dictionary_df_keys }
        #
        ## check the different cases
        #for individual in population :
        #
        #    dictionary_df["generation"].append(num_generations)
        #    
        #    for i in range(0, len(best_fitness)) :
        #        key = "fitness_value_%d" % i
        #        if fitness_names is not None :
        #            key = fitness_names[i]
        #        dictionary_df[key].append(individual.fitness.values[i])
        #    
        #    for i in range(0, len(individual.candidate)) :
        #        dictionary_df["gene_%d" % i].append(individual.candidate[i])
        #
        ## conver dictionary to DataFrame, save as CSV
        #df = pd.DataFrame.from_dict(dictionary_df)
        #df.to_csv(population_file_name, index=False)

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
    thread_pool = ThreadPool(n_threads) 

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
    total_surface = np.sum(individual)
    
    # numerically, we subtract the mean soja production from the theoretical max,
    # in order to transform the problem into a minimization problem
    #fitness_values = inspyred.ec.emo.Pareto([args["max_theoretical_soja"] - mean_soja, std_soja, total_surface])
    # NOTE new attempt, we just use a negative value
    fitness_values = inspyred.ec.emo.Pareto([-1 * mean_soja, std_soja, total_surface])

    return fitness_values

def main() :

    # there are a lot of moving parts inside an EA, so some modifications will still need to be performed by hand
    # a few hard-coded values, to be changed depending on the problem
    population_size = int(1e4)
    offspring_size = int(2e4)
    max_evaluations = int(1e8)
    tournament_selection_size = int(0.02 * population_size)

    mutation_rate = 0.1
    mutation_mean = 0.0
    mutation_stdev = 0.1

    crossover_rate = 0.8

    # options for logging and saving files
    overwrite_save_files = True
    
    # relevant variables are stored in a dictionary, to ensure compatibility with inspyred
    args = dict()

    # hard-coded values
    args["data_file"] = "../data/pred_2000_2017_avg.m.csv"
    args["log_directory"] = "soja-allocation-3-objectives"
    args["save_directory"] = args["log_directory"]
    args["population_file_name"] = "population"
    args["save_at_every_iteration"] = True # save the whole population at every iteration
    args["random_seeds"] = [42] # list of random seeds, because we might want to run the evolutionary algorithm in a loop 
    args["n_threads"] = 60 # TODO change number of threads 
    fitness_names = ["mean_soja", "std_soja", "total_surface"]

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

    # compute maximum theoretical production, sum everything and divide by year
    max_theoretical_soja = np.sum(model_predictions) / model_predictions.shape[1]
    # also, the number of dimensions in the problem is equal to the number
    # of squares available in Europe
    n_dimensions = model_predictions.shape[0]
    
    # TODO it could be interesting to add two "extreme" individuals to the initial
    # population: one trivial solution with all parameters at 0.0, and one with all
    # parameters at 0.2; however, we should try this AFTER regular experiments
    seed_no_surface = [0.0] * n_dimensions
    seed_all_surface = [0.2] * n_dimensions
    seeds = [seed_no_surface, seed_all_surface]

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
        #ea.selector = inspyred.ec.selectors.tournament_selection 
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]
        #ea.replacer = inspyred.ec.replacers.plus_replacement
        ea.terminator = inspyred.ec.terminators.evaluation_termination
        ea.observer = observer
        ea.logger = args["logger"]

        # printout with the experimental parameters
        logger.info("Experimental hyperparameters of NSGA-II: population size=%d, offspring size=%d, stop condition after %d evaluations, tournament selection size=%d, mutation rate=%.4f, mutation mean=%.4f, mutation stdev=%.4f, crossover rate=%.4f" % 
                    (population_size, offspring_size, max_evaluations, tournament_selection_size, mutation_rate, mutation_mean, mutation_stdev, crossover_rate))

        final_population = ea.evolve(
                                generator=generator,
                                evaluator=multi_thread_evaluator,
                                pop_size=population_size,
                                num_selected=offspring_size,
                                maximize=False,
                                bounder=inspyred.ec.Bounder(0.0, 0.2),
                                max_evaluations=max_evaluations,
                                
                                # parameters of the tournament selection
                                tournament_size = tournament_selection_size,
                                # parameters of the Gaussian mutation
                                mutation_rate = mutation_rate, # applied as an element-by-element basis
                                gaussian_mean = mutation_mean,
                                gaussian_stdev = mutation_stdev, # default was 1
                                crossover_rate = crossover_rate,
                                
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
                                
                                # used to compute the fitness
                                model_predictions = model_predictions,
                                max_theoretical_soja = max_theoretical_soja,
                                )


    # extract the Pareto front and plot it
    pareto_front = ea.archive
    
    mean_soja_values = [individual.fitness[0] for individual in pareto_front]
    std_soja_values = [individual.fitness[1] for individual in pareto_front]
    total_surface_values = [individual.fitness[2] for individual in pareto_front]
    
    sns.set_style('darkgrid')
    
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
    
    # also plot 2D projections of the plots
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    sns.scatterplot(x=mean_soja_values, y=std_soja_values, alpha=0.7)
    ax.set_title("2D projection of the Pareto front")
    ax.set_xlabel(fitness_names[0])
    ax.set_ylabel(fitness_names[1])
    # we also change the labels for the 'x' axis, trying to go back to the actual production
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #new_labels = ["%.2e" % (max_theoretical_soja - float(label)) for label in labels]
    #ax.set_xticklabels(new_labels)
    plt.savefig(os.path.join(args["save_directory"], "pareto-front-%s-%s.png" % (fitness_names[0], fitness_names[1])), dpi=300)
    plt.close(fig)

    # another 2D projection
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    sns.scatterplot(x=mean_soja_values, y=total_surface_values, alpha=0.7)
    ax.set_title("2D projection of the Pareto front")
    ax.set_xlabel(fitness_names[0])
    ax.set_ylabel(fitness_names[2])
    # we also change the labels for the 'x' axis, trying to go back to the actual production
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #new_labels = ["%.2e" % (max_theoretical_soja - float(label)) for label in labels]
    #ax.set_xticklabels(new_labels)
    plt.savefig(os.path.join(args["save_directory"], "pareto-front-%s-%s.png" % (fitness_names[0], fitness_names[2])), dpi=300)
    plt.close(fig)

    # save final archive (Pareto front) to csv
    save_population_to_csv(pareto_front, ea.num_generations, os.path.join(args["save_directory"], "final-pareto-front.csv"), fitness_names)
    
    # close logger
    close_logging(logger)

    return

if __name__ == "__main__" :
    sys.exit( main() )
