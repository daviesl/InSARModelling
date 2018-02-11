#!/usr/bin/env bash

module purge
module load python/2.7.11 
module load python/2.7.11-matplotlib          
module load gdal/1.11.1-python
./decompose_gamma.py -i ../T025D/20170831-20170928_HH_4rlks_eqa.unw -dp ../T025D/20170831_HH_4rlks_eqa.dem.par -th ../T025D/20170831_HH_4rlks_eqa.lv_theta -ph ../T025D/20170831_HH_4rlks_eqa.lv_phi -cor ../T025D/20170831-20170928_HH_4rlks_flat_eqa.cc -ct 0.15 -t 0.5 -o T025D_utme -covar -dscale 0.001 -noplot 
./decompose_gamma.py -i ../T130A/20170727-20170907_HH_4rlks_eqa.unw -dp ../T130A/20170727_HH_4rlks_eqa.dem.par -th ../T130A/20170727_HH_4rlks_eqa.lv_theta -ph ../T130A/20170727_HH_4rlks_eqa.lv_phi -cor ../T130A/20170727-20170907_HH_4rlks_flat_eqa.cc -ct 0.15 -t 0.5 -o T130A_utme -covar -dscale 0.001 -noplot
./decompose_gamma.py -i ../T131A/20170829-20170912_HH_4rlks_eqa.unw -dp ../T131A/20170829_HH_4rlks_eqa.dem.par -th ../T131A/20170829_HH_4rlks_eqa.lv_theta -ph ../T131A/20170829_HH_4rlks_eqa.lv_phi -cor ../T131A/20170829-20170912_HH_4rlks_flat_eqa.cc -ct 0.15 -t 0.5 -o T131A_utme -covar -dscale 0.001 -noplot

#./decompose_gamma.py -i ../T025D/20170831-20170928_HH_4rlks_eqa.unw -dp ../T025D/20170831_HH_4rlks_eqa.dem.par -th ../T025D/20170831_HH_4rlks_eqa.lv_theta -ph ../T025D/20170831_HH_4rlks_eqa.lv_phi -cor ../T025D/20170831-20170928_HH_4rlks_flat_eqa.cc -ct 0.15 -t 0.5 -o T025D_utm -covar -dscale 0.001 -noplot
#./decompose_gamma.py -i ../T130A/20170727-20170907_HH_4rlks_eqa.unw -dp ../T130A/20170727_HH_4rlks_eqa.dem.par -th ../T130A/20170727_HH_4rlks_eqa.lv_theta -ph ../T130A/20170727_HH_4rlks_eqa.lv_phi -cor ../T130A/20170727-20170907_HH_4rlks_flat_eqa.cc -ct 0.15 -t 0.5 -o T130A_utm -covar -dscale 0.001 -noplot
#./decompose_gamma.py -i ../T131A/20170829-20170912_HH_4rlks_eqa.unw -dp ../T131A/20170829_HH_4rlks_eqa.dem.par -th ../T131A/20170829_HH_4rlks_eqa.lv_theta -ph ../T131A/20170829_HH_4rlks_eqa.lv_phi -cor ../T131A/20170829-20170912_HH_4rlks_flat_eqa.cc -ct 0.15 -t 0.5 -o T131A_utm -covar -dscale 0.001 -noplot
