import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
from scipy import interpolate
from scipy import signal
import math
import theano.tensor as T
import argparse
#import seaborn as sns
from tempfile import mkdtemp
from gridutils import *
from datautils import * # has varres_data class
import logmgr

import pymc3 as pm
from pymc3.step_methods import smc
import pyproj
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

#######Logger for varrespy
logger = logmgr.logger('pymode')

# set up pymc3
logger.info( "Using PyMC3 version " + str(pm.__version__))


def parse():
	"""
	Command line parser for PyMoDE. Use -h option to get complete list of options.
	Returns: Tuple (arguments,[list of varres_data objects]) 
	"""
	parser = argparse.ArgumentParser(description='Python Modelling of Deformation Events using unwrapped InSAR interferograms')
	parser.add_argument('-i','--input-los', action='append', dest='los', help='<Required> Input line-of-site downsampled list file from Varres', required=True, type=str)
	parser.add_argument('-c','--input-covariance', action='append', dest='cov', help='<Required> Input covariance matrix binary file from Varres', required=True, type=str)
	parser.add_argument('-d','--input-dem', action='store', dest='dem', help='<Required> Input digital elevation model in GAMMA format', required=True, type=str)
	parser.add_argument('-p','--input-dem-par', action='store', dest='par', help='<Required> Input parameter file for digital elevation model in GAMMA format', required=True, type=str)
	parser.add_argument('-o','--output-trace', action='store', dest='traceout', help='<Required> Output file path for pickled PyMC3 trace', required=True, type=str)
	parser.add_argument('-s','--solver', action='store', dest='solver', help='<Required> Input file path for pickled PyMC3 trace', required=True, type=str)
	parser.add_argument('-r','--topo-correction', action='store', dest='topocorrection', help='<Required> Topographic correction. F = First Order Green\'s Function Convolution, V = Varying Depth, N = None', required=True, type=str)
	
	
	inps = parser.parse_args()

	# Ensure number of inputs match number of covariance matrices.
	if len(inps.los) != len(inps.cov):
		logger.error('Number of line-of-site inputs must equal number of corresponding covariance matrix inputs.')
		sys.exit(1)
	
	data = []
	for i in xrange(len(inps.los)):
		data.append(varres_data(inps.los[i],inps.cov[i]))
	
	return inps, data

inps, data_list = parse()

test_folder = mkdtemp(prefix='SMCT')

for v in data_list:
	v.load()

par_D = readpar(inps.par)
dem_D = readgamma(inps.dem, par_D)

demlon = linspaceb(float(par_D['corner_lon']),float(par_D['post_lon']),int(par_D['width']))
demlat = linspaceb(float(par_D['corner_lat']),float(par_D['post_lat']),int(par_D['nlines']))

f = RegularGridInterpolator((demlon,np.flipud(demlat)),np.flipud(dem_D).T)
dem_max = 0.0
dem_medians = []
for v in data_list:
	v.interpolateDEM(f) 
	dem_max = max(dem_max,v.maxDEM())
	dem_medians.append(v.medianDEM())

dem_maxmed = max(dem_medians)
dem_reflevel = 0.75 * dem_max + 0.25 * dem_maxmed # arbitrary
for v in data_list:
	v.set_dem_reflevel(dem_reflevel)

logger.info("DEM reference level = " + str(dem_reflevel))

# TODO create arguments for priors
# get average (centre) location of data
mean_lons = []
mean_lats = []
for v in data_list:
	mean_lons.append(np.mean(v.lon()))
	mean_lats.append(np.mean(v.lat()))

ld = (np.mean(np.array(mean_lons)),np.mean(np.array(mean_lats)))

wgs = pyproj.Proj(init="EPSG:4326")
utm = pyproj.Proj("+proj=utm +zone="+str(long2UTM(ld[0]))+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

# do everything in UTM
ldE, ldN = pyproj.transform(wgs,utm,ld[0],ld[1])

for v in data_list:
	v.reproject(wgs,utm)

# std deviations on priors
hstd = 2000.0 

basic_model = pm.Model()

with basic_model:
	x0 = pm.Uniform('x0',ldE-hstd,ldE+hstd,transform=None)
	y0 = pm.Uniform('y0',ldN-hstd,ldN+hstd,transform=None)
	z0 = pm.Uniform('z0',0,1000,transform=None)
	radius = pm.Uniform('radius',0,500,transform=None)

	import mogi
	mogi.useTheano()
	for v in data_list:
		mogi.add(v.E_utm, v.N_utm, v.dem, v.los_data[:,8], v.los_data[:,9], v.los_data[:,10], v.los_data[:,5], v.los_data[:,6])
	mogi.setDEM(dem_D,par_D,128,100) # 100m padding

	if inps.topocorrection=='F':
		logger.info('Using First Order Green\'s Function Topographic Correction')
		# First order topography correction
		llk = pm.DensityDist('llk',mogi.logpfirstorder,observed={'x0':x0,'y0':y0,'z0':z0,'radius':radius})
		#llk = pm.Potential('llk',logp(x0,y0,z0,radius))
	elif inps.topocorrection=='V':
		logger.info('Using Varying Depth Topographic Correction')
		# Varying depth topography correction
		llk = pm.DensityDist('llk',mogi.logpvaryingdepth,observed={'x0':x0,'y0':y0,'z0':z0,'radius':radius})
	elif inps.topocorrection=='N':
		logger.info('Using No Topographic Correction')
		# No topo correction
		llk = pm.DensityDist('llk',mogi.logpreferencelevel,observed={'x0':x0,'y0':y0,'z0':z0,'radius':radius})
		


	if inps.solver=='SMC':	
		n_chains = 256
		n_steps = 8
		tune_interval = 8
		n_jobs = 1
		trace = smc.sample_smc(
			n_steps=n_steps,
			n_chains=n_chains,
			tune_interval=tune_interval,
			#start=start,
			progressbar=True,
			stage=0,
			n_jobs=n_jobs,
			homepath=test_folder,
			model=basic_model
			)
	elif inps.solver=='NUTS':
		niter = 2000
		start = pm.find_MAP(model=basic_model)
		
		print("Maximum a-posteriori estimate:") 
		print(start)
	
		#step = pm.NUTS(scaling=start)
		#trace = pm.sample(niter, start = start, step = step, nchains=4)

		step = pm.NUTS()
		trace = pm.sample(niter, step = step, tune=2000, cores=4)
	elif inps.solver=='MH':
		# Metropolis hastings
		step = pm.Metropolis()
		trace = pm.sample(niter, step = step, tune=2000)
	else:
		logger.error('Unsupported solver ' + inps.solver)
		sys.exit(1)

	import pickle
	pickle.dump(trace,open(inps.traceout,"wb"))

	#pm.traceplot(trace)
	#ax = plt.gca()
	#ax.get_xaxis().get_major_formatter().set_useOffset(False)
	#ax.get_xaxis().get_major_formatter().set_scientific(False)
	#plt.show()

	print (pm.summary(trace))
	print (trace)

#shutil.rmtree(test_folder)
