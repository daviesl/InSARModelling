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
#
#./decompose_gamma.py \
#	-i ../T025D/20170831-20170928_HH_4rlks_eqa.unw \
#	-dp ../T025D/20170831_HH_4rlks_eqa.dem.par \
#	-th ../T025D/20170831_HH_4rlks_eqa.lv_theta \
#	-ph ../T025D/20170831_HH_4rlks_eqa.lv_phi \
#	-cor ../T025D/20170831-20170928_HH_4rlks_flat_eqa.cc \
#	-ct 0.15 \
#	-t 0.5 \
#	-o $OT025D \
#	-covar -dscale 0.001 -noplot 
#./decompose_gamma.py \
#	-i ../T130A/20170727-20170907_HH_4rlks_eqa.unw \
#	-dp ../T130A/20170727_HH_4rlks_eqa.dem.par \
#	-th ../T130A/20170727_HH_4rlks_eqa.lv_theta \
#	-ph ../T130A/20170727_HH_4rlks_eqa.lv_phi \
#	-cor ../T130A/20170727-20170907_HH_4rlks_flat_eqa.cc \
#	-ct 0.15 \
#	-t 0.5 \
#	-o $OT130A \
#	-covar -dscale 0.001 -noplot
#./decompose_gamma.py \
#	-i ../T131A/20170829-20170912_HH_4rlks_eqa.unw \
#	-dp ../T131A/20170829_HH_4rlks_eqa.dem.par \
#	-th ../T131A/20170829_HH_4rlks_eqa.lv_theta \
#	-ph ../T131A/20170829_HH_4rlks_eqa.lv_phi \
#	-cor ../T131A/20170829-20170912_HH_4rlks_flat_eqa.cc \
#	-ct 0.15 \
#	-t 0.5 \
#	-o $OT131A \
#	-covar -dscale 0.001 -noplot

env MKL_THREADING_LAYER=GNU python model_argparse.py \
	-i $OT025D'.txt' \
	-c $OT025D'.cov' \
	-i $OT131A'.txt' \
	-c $OT131A'.cov' \
	-i $OT130A'.txt' \
	-c $OT130A'.cov' \
	-d ../T025D/20170831_HH_4rlks_eqa.dem \
	-p ../T025D/20170831_HH_4rlks_eqa.dem.par \
	-o smc_vd_trace.p \
	-s SMC \
	-r V
