#!/usr/bin/env bash

module purge
module load python/2.7.11 
#module load python/2.7.11-matplotlib          
module load matplotlib/1.5.1-py2.7
module load gdal/1.11.1-python

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/usr/lib64/libgomp.so.1

export OT025D=T025D_utmf
export OT131A=T131A_utmf
export OT130A=T130A_utmf

env MKL_THREADING_LAYER=GNU python plot_ppd_residuals.py \
	-i $OT025D'.txt' \
	-c $OT025D'.cov' \
	-i $OT131A'.txt' \
	-c $OT131A'.cov' \
	-i $OT130A'.txt' \
	-c $OT130A'.cov' \
	-d ../T025D/20170831_HH_4rlks_eqa.dem \
	-p ../T025D/20170831_HH_4rlks_eqa.dem.par \
	-t ./smc_vd_trace.p \
	-s SMC \
	-b 0 \
	-r V
#	-t test_argparse_smc_trace.p  \
