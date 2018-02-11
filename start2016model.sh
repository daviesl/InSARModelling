#!/usr/bin/env bash

module purge
module load python/2.7.11 
#module load python/2.7.11-matplotlib          
module load matplotlib/1.5.1-py2.7
module load gdal/1.11.1-python

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/usr/lib64/libgomp.so.1
env MKL_THREADING_LAYER=GNU python model_2016.py
