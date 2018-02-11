#!/usr/bin/env bash


module purge
module load python/2.7.11 
module load python/2.7.11-matplotlib          
module load gdal/1.11.1-python
./decompose_gamma.py -i ../TEST4/T025D/20150205-20160107_HH_4rlks_eqa.unw  -cor ../TEST4/T025D/20150205-20160107_HH_4rlks_flat_eqa.cc -dp ../TEST4/T025D/20150205_HH_4rlks_eqa.dem.par -ph ../TEST4/T025D/20150205_HH_4rlks_eqa.lv_phi -th ../TEST4/T025D/20150205_HH_4rlks_eqa.lv_theta -ct 0.15 -t 0.5 -o T025D_2016_utm -covar -dscale 0.001 -noplot
./decompose_gamma.py -i ../TEST4/T130A/20151217-20160114_HH_4rlks_eqa.unw  -cor ../TEST4/T130A/20151217-20160114_HH_4rlks_flat_eqa.cc -dp ../TEST4/T130A/20151217_HH_4rlks_eqa.dem.par -ph ../TEST4/T130A/20151217_HH_4rlks_eqa.lv_phi -th ../TEST4/T130A/20151217_HH_4rlks_eqa.lv_theta -ct 0.15 -t 0.2 -o T130A_2016_utm -covar -dscale 0.001 -noplot
