# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:53:05 2024

@author: Alberto
"""

import datetime
import logging
import numpy as np
import os

from logging.handlers import RotatingFileHandler
from queue import Queue
from threading import Thread

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

def fitness_function(individual, args) : 
    """
    This is the fitness function. It has been isolated from inspyred's version
    in order to re-use it for other algorithms without changing the code.
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
    
    return mean_soja, std_soja, total_surface