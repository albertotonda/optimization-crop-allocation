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
import numpy as np
import os
import random
import pandas as pd
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

def observer(population, num_generations, num_evaluations, args) :
    """
    The observer is a classic function for inspyred, that prints out information and/or saves individuals. However, it can be easily re-used by other
    evolutionary approaches.
    """
    logger = args["logger"]
    save_directory = args["save_directory"]
    save_at_every_iteration = args["save_at_every_iteration"]
    fitness_names = args.get("fitness_names", None)

    # first, a check to verify the type of library we are working with
    library_used = "unknown"
    if hasattr(population[0], "candidate") and hasattr(population[0], "fitness") :
        # we are using inspyred
        library_used = "inspyred"
    else :
        library_used = "cma-es"

    best_individual = best_fitness = None

    if library_used == "inspyred" :
        best_individual = population[0].candidate
        best_fitness = population[0].fitness

    # some output
    logger.info("Generation %d (%d evaluations), best individual fitness: %s" % (num_generations, num_evaluations, str(best_fitness)))

    # save the whole population to file
    if save_at_every_iteration :

        # create file name, with information on random seed and population
        population_file_name = "%d-%s-generation-%d.csv" % (args["random_seed"], args["population_file_name"], num_generations)
        population_file_name = os.path.join(save_directory, population_file_name)
        logger.debug("Saving population file to \"%s\"..." % population_file_name)

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

    # for the moment, there is just a placeholder here
    # TODO replace it with the correct fitness function
    from pymoo.problems import get_problem
    problem = get_problem("dascmop7", 12)
    
    fitness_values, _ = problem.evaluate(np.array(individual))
    fitness_values = inspyred.ec.emo.Pareto(fitness_values)

    return fitness_values

def main() :

    # there are a lot of moving parts inside an EA, so some modifications will still need to be performed by hand
    
    # a few hard-coded values, to be changed depending on the problem
    # relevant variables are stored in a dictionary, to ensure compatibility with inspyred
    args = dict()

    # unique name of the directory
    args["log_directory"] = "unique-name"
    args["save_directory"] = args["log_directory"]
    args["population_file_name"] = "population.csv"
    args["save_at_every_iteration"] = True # save the whole population at every iteration
    args["random_seeds"] = [42] # list of random seeds, because we might want to run the evolutionary algorithm in a loop 
    args["n_threads"] = 8 # TODO change number of threads 

    # initialize logging, using a logger that smartly manages disk occupation
    logger = initialize_logging(args["log_directory"])

    # also save pointer to the logger into the dictionary
    args["logger"] = logger

    # start program
    logger.info("Hi, I am a program, starting now!")
    logger.debug(type(logger))

    # start a series of experiments, for each random seed
    for random_seed in args["random_seeds"] :

        logger.info("Starting experiment with random seed %d..." % random_seed)
        args["random_seed"] = random_seed

        # initalization of ALL random number generators, to try and ensure repatability
        prng = random.Random(random_seed)
        np.random.seed(random_seed) # this might become deprecated, and creating a dedicated numpy pseudo-random number generator instance would be better 

        # create an instance of EvolutionaryComputation (generic EA) and set up its parameters
        # define all parts of the evolutionary algorithm (mutation, selection, etc., including observer)
        ea = inspyred.ec.emo.NSGA2(prng)
        ea.selector = inspyred.ec.selectors.tournament_selection 
        ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.gaussian_mutation]
        ea.replacer = inspyred.ec.replacers.plus_replacement
        ea.terminator = inspyred.ec.terminators.evaluation_termination
        ea.observer = observer
        ea.logger = args["logger"]

        # also create a generator function
        def generator(random, args) :
            return [ random.uniform(0.0, 0.2) for _ in range(args["n_dimensions"]) ]

        final_population = ea.evolve(
                                generator=generator,
                                evaluator=multi_thread_evaluator,
                                pop_size=100,
                                num_selected=150,
                                maximize=False,
                                bounder=inspyred.ec.Bounder(0.0, 0.2),
                                max_evaluations=10000,

                                # all items below this line go into the 'args' dictionary passed to each function
                                logger = args["logger"],
                                n_dimensions = 30,
                                n_threads = args["n_threads"],
                                population_file_name = args["population_file_name"],
                                random_seed = args["random_seed"],
                                save_directory = args["save_directory"],
                                save_at_every_iteration = args["save_at_every_iteration"],
                                fitness_names = ["mean_soja", "std_soja", "total_surface"],
                                )


    # TODO do something with the best individual

    # close logger
    close_logging(logger)

    return

if __name__ == "__main__" :
    sys.exit( main() )
