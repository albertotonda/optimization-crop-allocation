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

