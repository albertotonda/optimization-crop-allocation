# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:00:36 2023

Script that tries to greedily maximize one objective, by taking the most performing
pixels.

@author: Alberto
"""
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    
    # hard-coded values
    max_surface = 500
    
    # load the data file
    data_file = "../data/pred_2000_2017_avg.m.csv"
    df = pd.read_csv(data_file, sep=";", decimal=",") # unfortunately in European format
    year_columns = [c for c in df.columns if c.startswith("2")] # get year columns, like 2001, ..., 2017
    print("Data file \"%s\" seems to include predictions for years %s..." % 
                (data_file, str(year_columns)))
    print(df)
    
    
    # create a data structure with the mean soja production
    performances = []
    for i, row in df.iterrows() :
        
        row = df.iloc[i]
        index = row["Unnamed: 0"]
        production = sum(row[year_columns].values)
        
        performances.append([i, production])
        
    print(performances)
        
    # sort by (descending) production
    performances = sorted(performances, reverse=True, key=lambda x : x[1])
    
    i = 0
    total_production = []
    surface = 0.0
    while surface < max_surface :
        
        surface += 0.2 
        total_production.append(performances[i][1])
        
        i += 1
    
    mean_soja = np.mean(total_production)
    std_soja = np.std(total_production)
    print("Total surface: %.4f; Mean production: %.4e (std %.4e)" % (surface, mean_soja, std_soja))
    
    # another way of approaching the problem; we already sorted the pixels by
    # total production (sum over all years). We can take them in order, but
    # recompute the mean and the standard deviation as a sum of all pixels' production
    # by year, and then compute the standard deviation over that
    
    candidate_solution = np.zeros((df.shape[0],))
    
    i = 0
    surface = 0.0 
    while surface < max_surface :
        
        surface += 0.2
        candidate_solution[performances[i][0]] = 0.2
        
        i += 1
        
        
    model_predictions = df[year_columns].values
    production_by_year = np.zeros((model_predictions.shape[1],))
    for year in range(0, model_predictions.shape[1]) :
        
        # select column of data corresponding to a year
        model_predictions_year = model_predictions[:, year]
        
        # multiply, element-wise, each element of the candidate solution with
        # the predicted production for the corresponding square for that year
        production_by_year[year] = np.sum(np.multiply(candidate_solution, model_predictions_year))
    
    # now that we have the production by year, we can easily compute the first
    # and second fitness values
    mean_soja = np.mean(production_by_year)
    std_soja = np.std(production_by_year)
    
    print("Values recomputed in another way for total surface %.4f: mean %.4e; std %.4e" % (surface, mean_soja, std_soja))