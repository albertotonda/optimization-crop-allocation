# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:24:14 2023

Extract some points from a Pareto front of the experiments.

@author: Alberto
"""
import numpy as np
import os
import pandas as pd

if __name__ == "__main__" :
    
    # hard-coded values
    population_file = "../results/2023-12-06-results/42-population.csv-generation-500.csv"
    data_file = "../data/pred_2000_2017_avg.m.csv"
    
    # load CSV file
    df = pd.read_csv(population_file)
    
    # find the extreme (lowest) values for each fitness name
    fitness_names = [c for c in df.columns if not c.startswith("gene") and c != "generation"]
    print("Names of the columns with fitness values:", fitness_names)

    indexes = []    
    for fitness_name in fitness_names :
        
        min_rows = df[df[fitness_name] == df[fitness_name].min()]
        indexes.extend(min_rows.index.tolist())
        print(min_rows[fitness_names])
        
    # select only the rows previously identified, save them to disk
    df_selected = df.loc[indexes]
    print(df_selected[fitness_names])
    
    df_selected.to_csv("extraction_" + os.path.basename(population_file), index=False)

    # TODO we also perform a sanity check, where we try to recompute the fitness
    # values starting directly from the data
    print("Performing sanity check with data from \"%s\"..." % data_file)
    df_data = pd.read_csv(data_file, sep=";", decimal=",")
    selected_columns = [c for c in df_data.columns if c.startswith("2")]
    model_predictions = df_data[selected_columns].values
    max_theoretical_soja = np.sum(model_predictions) / model_predictions.shape[1]
    
    for index, row in df_selected.iterrows() :
        
        individual_numpy = row[[c for c in df_selected.columns if c.startswith("gene_")]].values
        
        production_by_year = np.zeros((model_predictions.shape[1],))
        
        for year in range(0, model_predictions.shape[1]) :
            
            # select column of data corresponding to a year
            model_predictions_year = model_predictions[:, year]
            
            # multiply, element-wise, each element of the candidate solution with
            # the predicted production for the corresponding square for that year
            production_by_year[year] = np.sum(np.multiply(individual_numpy, model_predictions_year))
            
        mean_soja = np.mean(production_by_year)
        std_soja = np.std(production_by_year)
        total_surface = np.sum(individual_numpy)
        
        print("Individual %d: fitness values read %.4e %.4e %.4e; recomputed %.4e %.4e %.4e" %
              (index, row["mean_soja"], row["std_soja"], row["total_surface"], max_theoretical_soja - mean_soja, std_soja, total_surface))
        
        print("Mean soja production: %.4f" % mean_soja)
        
    # let's perform another extraction, this time using a different formalism
    keys = ["id", "x", "y"] + ["best_" + fn for fn in fitness_names]
    extraction_dictionary = {k : [] for k in keys}
    
    extraction_dictionary["id"] = df_data["Unnamed: 0"].values
    extraction_dictionary["x"] = df_data["x"].values
    extraction_dictionary["y"] = df_data["y"].values
    
    for i in range(df_selected.shape[0]) :
        row = df_selected.iloc[i]
        extraction_dictionary["best_" + fitness_names[i]] = row[[c for c in df_selected.columns if c.startswith("gene_")]].values
        
    df_extraction = pd.DataFrame.from_dict(extraction_dictionary)
    df_extraction.to_csv("extraction.csv", index=False)