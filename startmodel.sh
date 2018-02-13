#!/usr/bin/env bash

module purge
module load python/2.7.11 
#module load python/2.7.11-matplotlib          
module load matplotlib/1.5.1-py2.7
module load gdal/1.11.1-python

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/usr/lib64/libgomp.so.1
env MKL_THREADING_LAYER=GNU python model_argparse.py \
	-i T025D_utme.txt \
	-c T025D_utme.cov \
	-i T131A_utme.txt \
	-c T131A_utme.cov \
	-i T130A_utme.txt \
	-c T130A_utme.cov \
	-d ../T025D/20170831_HH_4rlks_eqa.dem \
	-p ../T025D/20170831_HH_4rlks_eqa.dem.par \
	-o test_argparse_smc_trace.p 
