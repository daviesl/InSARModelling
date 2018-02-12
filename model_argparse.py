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
import seaborn as sns
from tempfile import mkdtemp
from gridutils import *
import logmgr

import pymc3 as pm
from pymc3.step_methods import smc
import pyproj
import pandas as pd

# set up pymc3
print "Using PyMC3 version " + str(pm.__version__)

#######Logger for varrespy
logger = logmgr.logger('pymode')

class varres_data(object):
	def __init__(self,los,cov):
		self.los_name = los
		self.cov_name = cov
	def diagCov(self,fn):
		c = np.fromfile(fn, dtype=np.float32)
		dim = int(math.sqrt(c.shape[0]))
		c = c.reshape((dim,dim))
		return np.diag(c)
	def load(self):
		# Varres LOS Data Headers:
		# Number xind yind east north data err wgt Elos Nlos Ulos
		# east/north are actually lon and lat
		self.los_data = np.loadtxt(self.los_name,skiprows=2)
		# Varres covariance data is a binary file describing a square matrix
		self.diagcov = self.diagCov(self.cov_name)
		# convert cm to m for los values and set minimum error to 0.01
		self.los_data[:,5] *= 0.01
		# combine error in los data
		self.los_data[:,6] = np.sqrt(self.los_data[:,6]**2 + self.diagcov)
		# convert cm to m for error in los data
		self.los_data[:,6] *= 0.01
	def lat(self, row=False):
		return self.val(4,row)
	def lon(self, row=False):
		return self.val(3,row)
	def losd(self,row=False):
		""" Line of sight displacement """
		return self.val(5,row)
	def downsample_stddev(self,row=False):
		return self.val(6,row)
	def unit_vector(self, row=False):
		""" returns (E, N, U) unit vector from ground to platform of LOS"""
		return (self.val(8,row),self.val(9,row),self,val(10,row))
	def val(self, col, row):
		if row==False:
			return self.los_data[:,col]
		else:
			return self.los_data[row,col]
	def interpolateDEM(self,f):
		self.dem = f(self.los_data[:,3:5]).flatten()
	def maxDEM(self):
		return self.dem.max()
	def medianDEM(self):
		return np.median(self.dem)
	def set_dem_reflevel(self,dem_reflevel):
		self.dem = dem_reflevel - self.dem
	def reproject(self,wgs,utm):
		self.E_utm, self.N_utm = pyproj.transform(wgs,utm,self.lon(),self.lat())


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


#T025D = np.loadtxt('T025D_utme.txt',skiprows=2)
#T130A = np.loadtxt('T130A_utme.txt',skiprows=2)
#T131A = np.loadtxt('T131A_utme.txt',skiprows=2)
#T025Ddiagcov = diagCov('T025D_utme.cov')
#T130Adiagcov = diagCov('T130A_utme.cov')
#T131Adiagcov = diagCov('T131A_utme.cov')
# convert cm to m for los values and set minimum error to 0.01
#T025D[:,5] *= 0.01
#T130A[:,5] *= 0.01
#T131A[:,5] *= 0.01
#T025D[:,6] = np.maximum(T025D[:,6],0.001)
#T130A[:,6] = np.maximum(T130A[:,6],0.001)
#T131A[:,6] = np.maximum(T131A[:,6],0.001)
#T025D[:,6] = np.sqrt(T025D[:,6]**2 + T025Ddiagcov)
#T130A[:,6] = np.sqrt(T130A[:,6]**2 + T130Adiagcov)
#T131A[:,6] = np.sqrt(T131A[:,6]**2 + T131Adiagcov)
#T025D[:,6] *= 0.01
#T130A[:,6] *= 0.01
#T131A[:,6] *= 0.01
#par_D = readpar('../T025D/20170831_HH_4rlks_eqa.dem.par')
#par_A = readpar('../T131A/20170829_HH_4rlks_eqa.dem.par')
#par_A2 = readpar('../T130A/20170727_HH_4rlks_eqa.dem.par')
#dem_A = readgamma('../T131A/20170829_HH_4rlks_eqa.dem', par_A)
#dem_D = readgamma('../T025D/20170831_HH_4rlks_eqa.dem', par_D)

for v in data_list:
	v.load()

par_D = readpar(inps.par)
dem_D = readgamma(inps.dem, par_D)

#cmap1 = plt.set_cmap('gray')
#plotdata(dem_D, par_D, cmap1)


from scipy.interpolate import RegularGridInterpolator

demlon = linspaceb(float(par_D['corner_lon']),float(par_D['post_lon']),int(par_D['width']))
demlat = linspaceb(float(par_D['corner_lat']),float(par_D['post_lat']),int(par_D['nlines']))

#f = interpolate.interp2d(demlon,demlat,dem_D,kind='linear')
#dem_T025D = f(T025D[:,3],T025D[:,4]).flatten()
#dem_T130A = f(T130A[:,3],T130A[:,4]).flatten()
#dem_T131A = f(T131A[:,3],T131A[:,4]).flatten()

print demlon.shape
print demlat.shape
print dem_D.shape
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

print("DEM reference level = " + str(dem_reflevel))

#print T025D.shape
#print dem_T025D.shape
#dem_T025D = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T025D[:,3],T025D[:,4]),method='linear')
#dem_T130A = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T130A[:,3:4]),method='linear')
#dem_T131A = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T131A[:,3:4]),method='linear')

#ld = (129.074,41.299,500)

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

#T025D_E_utm, T025D_N_utm = pyproj.transform(wgs,utm,T025D[:,3],T025D[:,4])
#T131A_E_utm, T131A_N_utm = pyproj.transform(wgs,utm,T131A[:,3],T131A[:,4])
#T130A_E_utm, T130A_N_utm = pyproj.transform(wgs,utm,T130A[:,3],T130A[:,4])

#minlon = min(T025D[:,3].min(),T131A[:,3].min(),T130A[:,3].min())
#maxlon = max(T025D[:,3].max(),T131A[:,3].max(),T130A[:,3].max())
#minlat = min(T025D[:,4].min(),T131A[:,4].min(),T130A[:,4].min())
#maxlat = max(T025D[:,4].max(),T131A[:,4].max(),T130A[:,4].max())

# 500m std deviations on priors
hstd = 2000.0 #/ (3600 * 30)

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
	#mogi.add(T025D_E_utm, T025D_N_utm, dem_T025D, T025D[:,8], T025D[:,9], T025D[:,10], T025D[:,5], T025D[:,6])
	#mogi.add(T131A_E_utm, T131A_N_utm, dem_T131A, T131A[:,8], T131A[:,9], T131A[:,10], T131A[:,5], T131A[:,6])
	#mogi.add(T130A_E_utm, T130A_N_utm, dem_T130A, T130A[:,8], T130A[:,9], T130A[:,10], T130A[:,5], T130A[:,6])
	mogi.setDEM(dem_D,par_D,128,100) # 100m padding

	llk = pm.DensityDist('llk',mogi.logpgridinterpolate,observed={'x0':x0,'y0':y0,'z0':z0,'radius':radius})
	#llk = pm.Potential('llk',logp(x0,y0,z0,radius))

	niter = 2000

	#start = pm.find_MAP(model=basic_model)
	
	#print("Maximum a-posteriori estimate:") 
	#print(start)
	
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

	#t = trace[niter//2:] # discard 50% of values
	#print t
	#x0_ = trace.get_values('x0',burn=niter//2, combine=True)

	#step = pm.NUTS(scaling=start)
	#trace = pm.sample(niter, start = start, step = step)

	#step = pm.NUTS()
	#trace = pm.sample(niter, step = step, tune=2000, cores=4)

	# Metropolis hastings
	#step = pm.Metropolis()
	#trace = pm.sample(niter, step = step, tune=2000)

	#fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
	#ax1.plot(trace['x0'])
	#ax2.plot(trace['y0'])
	#ax3.plot(trace['z0'])
	#ax4.plot(trace['radius'])
	#plt.show()

	import pickle
	pickle.dump(trace,open(inps.traceout,"wb"))

	pm.traceplot(trace)
	ax = plt.gca()
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	ax.get_xaxis().get_major_formatter().set_scientific(False)
	plt.show()

	#pm.traceplot(t, varnames=['x0','y0','z0','radius'])
	#plt.show()

	#pm.plot_posterior(trace)
	#plt.show()

	#pm.plot_posterior(trace, varnames=['x0','y0'],kde_plot=True)
	#plt.show()

	print (pm.summary(trace))
	print (trace)

	def crediblekdeplot(x,y,xlabel,ylabel):
		# Make a 2d normed histogram
		H,xedges,yedges=np.histogram2d(x,y,bins=40,normed=True)
		norm=H.sum() # Find the norm of the sum
		print("Norm of hist = " + str(norm))
		# Set contour levels
		contour2=0.95
		contour3=0.68
		# Set target levels as percentage of norm
		target2 = norm*contour2
		target3 = norm*contour3
		
		# Take histogram bin membership as proportional to Likelihood
		# This is true when data comes from a Markovian process
		def objective(limit, target):
		    w = np.where(H>limit)
		    count = H[w]
		    return count.sum() - target
		
		# Find levels by summing histogram to objective
		level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
		level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

		# For nice contour shading with seaborn, define top level
		level4=H.max()
		levels=[level2,level3,level4]

		print(levels)
		
		# Pass levels to normed kde plot
		fig,ax=plt.subplots()
		sns.kdeplot(x,y,shade=True,ax=ax,n_levels=levels,cmap="Reds_d",normed=True)
		ax.set_aspect('equal')
		plt.show()

	#crediblekdeplot(trace['z0'],trace['radius'],r"Depth $m$",r"Volume $m^3$")
	#crediblekdeplot(trace['x0'],trace['y0'],r"Easting $m$",r"Northing $m$")

	def crediblejointplot(x,y,xlabel,ylabel,clr,aspectequal=False):
		# Make a 2d normed histogram
		H,xedges,yedges=np.histogram2d(x,y,bins=40,normed=True)
		norm=H.sum() # Find the norm of the sum
		print("Norm of hist = " + str(norm))
		# Set contour levels
		contour2=0.95
		contour3=0.68
		# Set target levels as percentage of norm
		target2 = norm*contour2
		target3 = norm*contour3
		
		# Take histogram bin membership as proportional to Likelihood
		# This is true when data comes from a Markovian process
		def objective(limit, target):
		    w = np.where(H>limit)
		    count = H[w]
		    return count.sum() - target
		
		# Find levels by summing histogram to objective
		level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
		level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

		# For nice contour shading with seaborn, define top level
		level4=H.max()
		levels=[level2,level3,level4]

		print(levels)
		
		# Pass levels to normed kde plot
		df = pd.DataFrame(np.column_stack((x,y)),columns=[xlabel,ylabel])
		#graph = sns.jointplot(x=xlabel,y=ylabel,data=df,kind='kde',color=clr,n_levels=levels,normed=True) #.set_axis_labels(xlabel,ylabel)
		graph = sns.jointplot(x=xlabel,y=ylabel,data=df,color=clr,marker='.').plot_joint(sns.kdeplot,n_levels=levels,normed=True,linewidth=2,linestyle=':') 
		ax = plt.gca()
		ax.get_xaxis().get_major_formatter().set_useOffset(False)
		ax.get_xaxis().get_major_formatter().set_scientific(False)
		if aspectequal:
			ax.set_aspect('equal')
		plt.show()

	crediblejointplot(trace['z0'],trace['radius'],r"Depth $m$",r"Cavity Radius $m$",'r')
	crediblejointplot(trace['x0'],trace['y0'],r"Easting $m$",r"Northing $m$",'g',True)

	#sns.jointplot(x=trace['z0'],y=trace['radius'],kind='kde',color='r').set_axis_labels(r"Depth $m$",r"Volume $m^3$")
	#sns.jointplot(x=trace['x0'],y=trace['y0'],kind='kde',color='green').set_axis_labels(r"Easting $m$",r"Northing $m$")

	#sns.kdeplot(trace['x0'],trace['y0'])
	#plt.xlabel(r"Easting $m$")
	#plt.ylabel(r"Northing $m$")
	#plt.ticklabel_format(useOffset=False)
	#plt.show()

	#sns.kdeplot(trace['z0'],trace['radius'])
	#plt.xlabel(r"Depth $m$")
	#plt.ylabel(r"Volume $m^3$")
	#plt.ticklabel_format(useOffset=False)
	#plt.show()
	

	#pm.energyplot(trace)
	#plt.show()

	#pm.autocorrplot(t, varnames=['x0'])
	#pm.autocorrplot(trace)
	#plt.show()

	print (pm.summary(trace))
	#print (pm.summary(t))
	


#shutil.rmtree(test_folder)
