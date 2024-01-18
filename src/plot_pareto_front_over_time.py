# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:25:05 2023

Script to try and visualize the spread of the Pareto front over the generations
in an image, or maybe a video.

@author: Alberto
"""
import matplotlib.pyplot as plt
import os
import pandas as pd
import re as regex
import seaborn as sns
import sys

def main() :
    
    # hard-coded values
    fitness_x = "mean_soja"
    fitness_y = "std_soja"
    
    # target directory, containing all files
    target_directory = "soja-allocation-3-objectives"
    pareto_front_files = [f for f in os.listdir(target_directory) if f.endswith(".csv") and f.find("archive-generation") != -1]
    
    # set cool style for figures
    sns.set_style(style='darkgrid')
    
    print("I am going to work on %d files: %s" % (len(pareto_front_files), str(pareto_front_files)))
    
    # now we need some parsing magic to actually sort the files by number and
    # not alphabetically, otherwise we get a sequence like 1 - 10 - 100 - 1000 - 1001 - 1002...
    regular_expression = "\-([0-9]+)[\-|\.]" # captures any integer
    id_and_file = [ [int(regex.search(regular_expression, f).group(1)), f] for f in pareto_front_files]
    id_and_file = sorted(id_and_file, key=lambda x : x[0])
    
    print("Sorted list of files:", id_and_file)
    
    # fantastic! now we have to create a disgustingly large image that will show
    # the points in the Pareto front at each step
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    for file_id, file_name in [id_and_file[-1]] :
        df = pd.read_csv(os.path.join(target_directory, file_name))
        sns.scatterplot(data=df, x=fitness_x, y=fitness_y, alpha=0.3, label="Gen %d" % file_id)
    
    ax.legend(loc='best')
    plt.savefig("pareto-front-over-time.png", dpi=300)
    plt.close(fig)
    
    return

if __name__ == "__main__" :
    sys.exit(main())