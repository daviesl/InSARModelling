#!/usr/bin/env bash

module purge
module load python/2.7.11 
module load python/2.7.11-matplotlib
module load gdal/1.11.1-python

python plot_synthetics_yang.py
