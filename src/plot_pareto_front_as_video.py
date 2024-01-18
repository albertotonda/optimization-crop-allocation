# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 22:59:49 2023

@author: Alberto
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
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
    
    # before starting the plotting, we need to get the smallest values for each
    # fitness value, likely in the last file (as this is a minimization problem)
    df = pd.read_csv(os.path.join(target_directory, id_and_file[-1][1]))
    x_smallest = df[fitness_x].values.min()
    y_smallest = df[fitness_y].values.min()
    
    # now, we find the largest values (in file #0)
    file_name = id_and_file[0][1]
    df = pd.read_csv(os.path.join(target_directory, file_name))
    x_largest = df[fitness_x].values.max()
    y_largest = df[fitness_y].values.max()
    
    print("Limits of the axes: x=[%.2e,%.2e], y=[%.2e,%.2e]" % (x_smallest, x_largest, y_smallest, y_largest))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    #pareto_front = sns.scatterplot(data=df, x=fitness_x, y=fitness_y, alpha=0.3, ax=ax)
    pareto_front = ax.scatter(df[fitness_x].values, df[fitness_y].values, color="blue", alpha=0.7)
    ax.set_title("Pareto front at generation 0")
    #ax.set(xlim=[x_smallest, x_largest], ylim=[y_smallest, y_largest], xlabel=fitness_x, ylabel=fitness_y)
    ax.set(xlim=[-4.8e7, -3.0e7], ylim=[1.5e6, 2.4e6], xlabel=fitness_x, ylabel=fitness_y)
    
    def update(frame, ax=ax, id_and_file=id_and_file, pareto_front=pareto_front) :
        
        file_name = id_and_file[frame][1]
        df = pd.read_csv(os.path.join(target_directory, file_name))
        
        print("Now updating plot with file \"%s\"..." % file_name)
        
        data = np.stack([df[fitness_x].values, df[fitness_y].values]).T
        pareto_front.set_offsets(data)
        
        ax.set_title("Pareto front at generation %d" % frame)
        
        return
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(id_and_file), interval=100)
    ani.save(filename="ffmpeg_example.mp4", writer="ffmpeg")
    
    
    # all this part was an old example
    if False :
        fig, ax = plt.subplots()
        t = np.linspace(0, 3, 40)
        g = -9.81
        v0 = 12
        z = g * t**2 / 2 + v0 * t
        
        v02 = 5
        z2 = g * t**2 / 2 + v02 * t
        
        scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
        line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
        ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
        ax.legend()
    
        def update(frame):
            # for each frame, update the data stored on each artist.
            x = t[:frame]
            y = z[:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            # update the line plot:
            line2.set_xdata(t[:frame])
            line2.set_ydata(z2[:frame])
            return (scat, line2)
    
    
        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
        ani.save(filename="ffmpeg_example.mp4", writer="ffmpeg")
    
    
if __name__ == "__main__" :
    sys.exit( main() )