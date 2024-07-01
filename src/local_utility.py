# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:53:05 2024

@author: Alberto
"""

import datetime
import logging
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