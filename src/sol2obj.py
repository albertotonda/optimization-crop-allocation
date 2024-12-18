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
    use_cplex = True
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
    args["sol_file"] = "eu27-B/focused-7M-vps/eu27-B-vps-7M-solutions.csv"
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
    
    n_dimensions = model_predictions.shape[0]
    n_years = model_predictions.shape[1]

    sol_f = pd.read_csv(args["sol_file"], sep=",", decimal=".")
    logger.info("Solution file \"%s\" has shape %s" % (args["sol_file"], sol_f.shape))
    n_solutions = sol_f.shape[0]

    all_obj = {}
    all_obj["obj-area"] = []
    all_obj["obj-prod"] = []
    all_obj["obj-stdd"] = []
    for i in range(1, n_solutions):
        s = sol_f.T[i].values
        print(i, s)
        obj = fitness_function(s, args)
        all_obj["obj-prod"].append(obj[0])
        all_obj["obj-stdd"].append(obj[1])
        all_obj["obj-area"].append(obj[2])

    obj_pd = pd.DataFrame.from_dict(all_obj)
    obj_pd.to_csv("objectives-rec.csv", index=True)
        
    # close logger
    logger.info("All done")
    close_logging(logger)

    return

if __name__ == "__main__" :
    sys.exit( main() )
