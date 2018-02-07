import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
from scipy import interpolate
from scipy import signal
import math
import theano.tensor as T
import seaborn as sns
from tempfile import mkdtemp
from gridutils import *

test_folder = mkdtemp(prefix='SMCT')

# Headers:
# Number xind yind east north data err wgt Elos Nlos Ulos
# east/north are actually lon and lat
T025D = np.loadtxt('T025D_utme.txt',skiprows=2)
T130A = np.loadtxt('T130A_utme.txt',skiprows=2)
T131A = np.loadtxt('T131A_utme.txt',skiprows=2)

def diagCov(fn):
	c = np.fromfile(fn, dtype=np.float32)
	dim = int(math.sqrt(c.shape[0]))
	c = c.reshape((dim,dim))
	return np.diag(c)

T025Ddiagcov = diagCov('T025D_utme.cov')
T130Adiagcov = diagCov('T130A_utme.cov')
T131Adiagcov = diagCov('T131A_utme.cov')

# convert cm to m for los values and set minimum error to 0.01
T025D[:,5] *= 0.01
T130A[:,5] *= 0.01
T131A[:,5] *= 0.01


#T025D[:,6] = np.maximum(T025D[:,6],0.001)
#T130A[:,6] = np.maximum(T130A[:,6],0.001)
#T131A[:,6] = np.maximum(T131A[:,6],0.001)

T025D[:,6] = np.sqrt(T025D[:,6]**2 + T025Ddiagcov)
T130A[:,6] = np.sqrt(T130A[:,6]**2 + T130Adiagcov)
T131A[:,6] = np.sqrt(T131A[:,6]**2 + T131Adiagcov)

T025D[:,6] *= 0.01
T130A[:,6] *= 0.01
T131A[:,6] *= 0.01



def readpar(file):
    """Function to read a GAMMA parameter (metadata) file into a dictionary"""
    par={}
    with open(file) as f:
        next(f) # skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue # skip blank lines in the par file
            (key, val) = line.split(':')
            par[str(key)] = val.split()[0]
    return par

def readgamma(datafile, par):
    """Function to read GAMMA format float data files"""
    ct = int(par['width']) * int(par['nlines'])
    print("Number of elements in the file is",ct)
    dt = np.dtype('>f4') # GAMMA files are big endian 32 bit float
    d = np.fromfile(datafile, dtype=dt, count=ct)
    d = d.reshape(int(par['nlines']), int(par['width']))
    print("Number of elements and size of the array is",d.size, d.shape)
    d[d==0]= np.nan # convert zeros to nan
    return d


par_D = readpar('../T025D/20170831_HH_4rlks_eqa.dem.par')
par_A = readpar('../T131A/20170829_HH_4rlks_eqa.dem.par')
par_A2 = readpar('../T130A/20170727_HH_4rlks_eqa.dem.par')
dem_A = readgamma('../T131A/20170829_HH_4rlks_eqa.dem', par_A)
dem_D = readgamma('../T025D/20170831_HH_4rlks_eqa.dem', par_D)

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
dem_T025D = f(T025D[:,3:5]).flatten()
dem_T025D_max = dem_T025D.max()
dem_T130A = f(T130A[:,3:5]).flatten()
dem_T130A_max = dem_T130A.max()
dem_T131A = f(T131A[:,3:5]).flatten()
dem_T131A_max = dem_T131A.max()
dem_max = max(dem_T025D_max,dem_T130A_max,dem_T131A_max)
dem_maxmed = max(np.median(dem_T025D),np.median(dem_T130A),np.median(dem_T131A_max))
dem_reflevel = 0.75 * dem_max + 0.25 * dem_maxmed # arbitrary
dem_T025D = dem_reflevel - dem_T025D
dem_T130A = dem_reflevel - dem_T130A
dem_T131A = dem_reflevel - dem_T131A

print("DEM reference level = " + str(dem_reflevel))

print T025D.shape
print dem_T025D.shape
#dem_T025D = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T025D[:,3],T025D[:,4]),method='linear')
#dem_T130A = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T130A[:,3:4]),method='linear')
#dem_T131A = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T131A[:,3:4]),method='linear')

ld = (129.074,41.299,500)
ustc = (129.074200,41.298200,0)
uri = (129.067000,41.309000,700)


# set up pymc3
import pymc3 as pm
from pymc3.step_methods import smc
import pyproj
import pandas as pd

wgs = pyproj.Proj(init="EPSG:4326")
utm = pyproj.Proj("+proj=utm +zone="+str(long2UTM(ld[0]))+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

# do everything in UTM
ldE, ldN = pyproj.transform(wgs,utm,ld[0],ld[1])
T025D_E_utm, T025D_N_utm = pyproj.transform(wgs,utm,T025D[:,3],T025D[:,4])
T131A_E_utm, T131A_N_utm = pyproj.transform(wgs,utm,T131A[:,3],T131A[:,4])
T130A_E_utm, T130A_N_utm = pyproj.transform(wgs,utm,T130A[:,3],T130A[:,4])

minlon = min(T025D[:,3].min(),T131A[:,3].min(),T130A[:,3].min())
maxlon = max(T025D[:,3].max(),T131A[:,3].max(),T130A[:,3].max())
minlat = min(T025D[:,4].min(),T131A[:,4].min(),T130A[:,4].min())
maxlat = max(T025D[:,4].max(),T131A[:,4].max(),T130A[:,4].max())

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
	mogi.add(T025D_E_utm, T025D_N_utm, dem_T025D, T025D[:,8], T025D[:,9], T025D[:,10], T025D[:,5], T025D[:,6])
	mogi.add(T131A_E_utm, T131A_N_utm, dem_T131A, T131A[:,8], T131A[:,9], T131A[:,10], T131A[:,5], T131A[:,6])
	#mogi.add(T130A_E_utm, T130A_N_utm, dem_T130A, T130A[:,8], T130A[:,9], T130A[:,10], T130A[:,5], T130A[:,6])
	mogi.setDEM(dem_D,par_D,128,100) # 100m padding

	llk = pm.DensityDist('llk',mogi.logpgridinterpolate,observed={'x0':x0,'y0':y0,'z0':z0,'radius':radius})
	#llk = pm.Potential('llk',logp(x0,y0,z0,radius))

	niter = 5000

	start = pm.find_MAP(model=basic_model)
	
	print("Maximum a-posteriori estimate:") 
	print(start)
	
	#n_chains = 200
	#n_steps = 50
	#tune_interval = 10
	#n_jobs = 1
	#trace = smc.sample_smc(
	#	n_steps=n_steps,
	#	n_chains=n_chains,
	#	tune_interval=tune_interval,
	#	n_jobs=n_jobs,
	#	#start=start,
	#	progressbar=True,
	#	stage=0,
	#	homepath=test_folder,
	#	model=basic_model
	#	)

	#t = trace[niter//2:] # discard 50% of values
	#print t
	#x0_ = trace.get_values('x0',burn=niter//2, combine=True)

	#step = pm.NUTS(scaling=start)
	#trace = pm.sample(niter, start = start, step = step)

	#step = pm.NUTS()
	#trace = pm.sample(niter, step = step, tune=2000)

	# Metropolis hastings
	step = pm.Metropolis()
	trace = pm.sample(niter, step = step, tune=2000)

	#fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
	#ax1.plot(trace['x0'])
	#ax2.plot(trace['y0'])
	#ax3.plot(trace['z0'])
	#ax4.plot(trace['radius'])
	#plt.show()

	import pickle
	pickle.dump(trace,open("trace_metro.p","wb"))

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
