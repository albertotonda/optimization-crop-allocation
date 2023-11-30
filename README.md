# Multi-objective optimization of crop allocation
Repository for the project with David Makowski and Mathilde Chen.

## Summary
We have crop yields (predicted or recorded/detrended) for several years, in a time series, for several adjacent regions. In the most recent case, soy allocation.

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

## 2023-11-30
Multi-objective approach to the problem of soja crops allocation:
- maximize soja production
- minimize inter-year variance
- minimize surface

## TODO
EA (single- or multi-objective), using threads.

## Links to data and/or papers
https://doi.org/10.6084/m9.figshare.11903277
https://www.nature.com/articles/s43016-022-00481-3
https://www.fao.org/faostat/en/#home
