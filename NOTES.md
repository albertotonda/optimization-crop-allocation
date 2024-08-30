# Multi-objective optimization of crop allocation
Repository for the project with Mathilde Chen, David Makowski and Georgios Katsirelos.

## Summary
We have crop yields (predicted or recorded/detrended) for several years, in a time series, for several adjacent regions. In the most recent case, soy allocation.

## Links to Google Drive
[General folder] 
[Experimental plan] https://docs.google.com/document/d/1rOV_k2apjmq_Z5UWfzW9ISwiRqY4oppb5GHv_Svkq2g/edit?usp=sharing
[Folders with results]

## TODO
1. Compare Pareto fronts of 2- and 3-objective runs. What are the best solutions found?
2. What happens if we inject the two "best" solutions as seeds?
3. Idea: use CMA-ES with varying weights, multiple times, to find points on the Pareto front.
4. Otherwise, MOEA/D.

## Data
Soybean yield predictions in Europe over 2000-2017 period, following 2 hypotheses:
1) each pixel has a surface of 55 km²
2) soybean production = 20% of the surface of each pixel (corresponding to the maximum surface possible).

Four different models were used to produce the yield predictions:
- avg.m: based on monthly averages of climate data
- avg.s: based on seasonal (i.e. over the full soybean growing season) averages of climate data
- pca.m.2: based on the 2 first component of monthly data
- pca.m.3: based on the 3 first component of monthly data

Each files corresponds to 1 of these models. The columns names are:
model: model used to predict soybean
x: longitude
y: latitude
mean_Ya_pred and sd_Ya_pred: mean and standard deviation of yield over the 2000-2017 period per site
Following columns (from 2000 to 2017) are estimated annual yields.

## 2024-08-13
Turns out, most of the issues were due to algorithmic problems. The pymoo implementation of NSGA-II works really well, now the last thing is to tune hyperparameters. Unfortunately, pymoo's NSGA-II becomes much slower when it starts checking stagnation (it has to recompute the hypervolume of the Pareto front at every generation). So, right now probably the best way to go is to set `mu=1000` and `max_generations=100000`.

## 2024-01-26
There might be some issues.

Arguments to discuss:
- experiments, deadline paper
- linear programming with George

## 2023-11-30
Multi-objective approach to the problem of soja crops allocation:
- maximize soja production
- minimize inter-year variance
- minimize surface

## Links to data and/or papers
https://doi.org/10.6084/m9.figshare.11903277
https://www.nature.com/articles/s43016-022-00481-3
https://www.fao.org/faostat/en/#home