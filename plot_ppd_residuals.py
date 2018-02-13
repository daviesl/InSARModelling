import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import interpolate
from scipy import signal
import math
from gridutils import *
from datautils import *
from plotutils import *
import pymc3 as pm
from pymc3.step_methods import smc
import pyproj
import pickle
import theano.tensor as T
import mogi
from scipy.interpolate import griddata
import logmgr
import argparse
from scipy.interpolate import RegularGridInterpolator

#######Logger for varrespy
logger = logmgr.logger('pymode')

# set up pymc3
logger.info( "Using PyMC3 version " + str(pm.__version__))

def parse():
	"""
	Command line parser for plotting routines for PyMoDE. Use -h option to get complete list of options.
	Returns: Tuple (arguments,[list of varres_data objects]) 
	"""
	parser = argparse.ArgumentParser(description='Plot output from PyMoDE')
	parser.add_argument('-i','--input-los', action='append', dest='los', help='<Required> Input line-of-site downsampled list file from Varres', required=True, type=str)
	parser.add_argument('-c','--input-covariance', action='append', dest='cov', help='<Required> Input covariance matrix binary file from Varres', required=True, type=str)
	parser.add_argument('-d','--input-dem', action='store', dest='dem', help='<Required> Input digital elevation model in GAMMA format', required=True, type=str)
	parser.add_argument('-p','--input-dem-par', action='store', dest='par', help='<Required> Input parameter file for digital elevation model in GAMMA format', required=True, type=str)
	parser.add_argument('-t','--input-trace', action='store', dest='tracein', help='<Required> Input file path for pickled PyMC3 trace', required=True, type=str)
	parser.add_argument('-s','--solver', action='store', dest='solver', help='<Required> Input file path for pickled PyMC3 trace', required=True, type=str)
	parser.add_argument('-b','--burnin', action='store', dest='burnin', help='<Required> Number of samples to discard for burnin', required=True, type=int)
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

for v in data_list:
	v.load()

par_D = readpar(inps.par)
dem_D = readgamma(inps.dem, par_D)

demlon = linspaceb(float(par_D['corner_lon']),float(par_D['post_lon']),int(par_D['width']))
demlat = linspaceb(float(par_D['corner_lat']),float(par_D['post_lat']),int(par_D['nlines']))

#print demlon.shape
#print demlat.shape
#print dem_D.shape
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

# 500m std deviations on priors
hstd = 1000.0 #/ (3600 * 30)
vstd = 1500.0

trace = pickle.load(open(inps.tracein,"rb"))

print len(trace)

burnin = inps.burnin # TODO make this argparsed

# Print and plot for debug
print (pm.summary(trace))
pm.traceplot(trace)
#pm.plot_traces(trace)

if inps.solver=='SMC':
	argmaxlike = trace.get_values('like',burn=burnin).argmax()
	x0 = trace.get_values('x0',burn=burnin)[argmaxlike]
	y0 = trace.get_values('y0',burn=burnin)[argmaxlike]
	z0 = trace.get_values('z0',burn=burnin)[argmaxlike]
	radius = trace.get_values('radius',burn=burnin)[argmaxlike]
else:
	x0 = np.median(trace.get_values('x0',burn=burnin))
	y0 = np.median(trace.get_values('y0',burn=burnin))
	z0 = np.median(trace.get_values('z0',burn=burnin))
	radius = np.median(trace.get_values('radius',burn=burnin))
	
print "y0: %f x0: %f radius %f z0 %f"%(y0,x0,radius,z0)

x0ll, y0ll = pyproj.transform(utm,wgs,x0,y0)

mogi.useNumpy()
for v in data_list:
	mogi.add(v.E_utm, v.N_utm, v.dem, v.los_data[:,8], v.los_data[:,9], v.los_data[:,10], v.los_data[:,5], v.los_data[:,6])

gridsize = 128 # TODO argparse this
padding = 100 # 100m padding
mogi.setDEM(dem_D,par_D,gridsize,padding) 

syntheticdata = []
residualdata = []

if inps.topocorrection=='F':
	(fo_dE, fo_dN, fo_dZ) = mogi.mogiTopoCorrection(x0,y0,z0,radius,mogi.utm_EE,mogi.utm_NN,mogi.utm_dem,1.0,False)
	dEf = mogi.Tgridinterp((mogi.utm_E,mogi.utm_N),fo_dE.T)
	dNf = mogi.Tgridinterp((mogi.utm_E,mogi.utm_N),fo_dN.T)
	dZf = mogi.Tgridinterp((mogi.utm_E,mogi.utm_N),fo_dZ.T)

	for o in mogi.observeddata:
		# interpolate first order correction
		dE1, dN1, dZ1 = mogi.obinterpolate(dEf,dNf,dZf,o)
		# Compute zeroth order correction more precisely using utm coordinates instead of interpolation
		(ux,uxx,uxxx) = mogi.mogiZerothOrder(x0,y0,z0,radius,mogi.mu,mogi.nu,mogi.P0,o.E_utm,o.N_utm)
		dEc = ux[0]+dE1
		dNc = ux[1]+dN1
		dZc = ux[2]+dZ1
		syn = o.uE*dEc + o.uN*dNc + o.uZ*dZc
		syntheticdata.append(syn)
		residualdata.append(syn-o.dLOS)

elif inps.topocorrection=='V':
	for o in mogi.observeddata:
		(ux,uxx,uxxx) = mogi.mogiZerothOrder(x0,y0,z0,radius,mogi.mu,mogi.nu,mogi.P0,o.E_utm,o.N_utm,o.DEM)
		dEc = ux[0]
		dNc = ux[1]
		dZc = ux[2]
		syn = o.uE*dEc + o.uN*dNc + o.uZ*dZc
		syntheticdata.append(syn)
		residualdata.append(syn-o.dLOS)
elif inps.topocorrection=='N':
	for o in mogi.observeddata:
		(ux,uxx,uxxx) = mogi.mogiZerothOrder(x0,y0,z0,radius,mogi.mu,mogi.nu,mogi.P0,o.E_utm,o.N_utm)
		dEc = ux[0]
		dNc = ux[1]
		dZc = ux[2]
		syn = o.uE*dEc + o.uN*dNc + o.uZ*dZc
		syntheticdata.append(syn)
		residualdata.append(syn-o.dLOS)
	

def makegaussian(xs,ys,sigma,mu):
	x, y = np.meshgrid(np.linspace(-1,1,xs), np.linspace(-1,1,ys))
	d = np.sqrt(x*x+y*y)
	return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

#mindim = min(dem_D_utm.shape[1],dem_D_utm.shape[0])
#dem_D_utm = makegaussian(mindim,mindim,1.0,0.0) * 500

#fig = plt.figure()
#fa1 = fig.add_subplot(131)
#cfa1 = fa1.imshow(np.real(dem_dE))
#fa2 = fig.add_subplot(132)
#cfa2 = fa2.imshow(np.real(dem_dN))
#fa3 = fig.add_subplot(133)
#cfa3 = fa3.imshow(np.real(dem_dZ))
#plt.colorbar(cfa1, ax=fa1)
#plt.colorbar(cfa2, ax=fa2)
#plt.colorbar(cfa3, ax=fa3)
#plt.show()



#
#(T025D_dE1, T025D_dN1, T025D_dZ1) = mogiDEM__(ldE,ldN,ld[2],1e6,T025D_E_utm,T025D_N_utm,dem_T025D)
#T025Dlos1 = T025D[:,8]*T025D_dE1 + T025D[:,9]*T025D_dN1 + T025D[:,10]*T025D_dZ1
#mf1 = (((T025Dlos1 - T025D[:,5]) / T025D[:,6])**2).sum()
#print "Initial misfit = " + str(mf1)
#
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax3d1 = fig.add_subplot(211, projection='3d')
#ax3d1.scatter(T025D[:,3],T025D[:,4],T025D_los)
#ax3d2 = fig.add_subplot(212, projection='3d')
#ax3d2.scatter(T025D[:,3],T025D[:,4],T025D[:,5])
#plt.show()

npasses = len(residualdata)

fig, axarr = plt.subplots(3,npasses, sharex=True, sharey=True)

pad=5

maxdiff = max([d.max() for d in residualdata])

# Y-Axis labels
axarr[0,0].annotate("Synthetic LOS", xy=(0, 0.5), xytext=(-axarr[0,0].yaxis.labelpad - pad, 0),xycoords=axarr[0,0].yaxis.label, textcoords='offset points',size='large', ha='right', va='center')
axarr[1,0].annotate("InSAR Data", xy=(0, 0.5), xytext=(-axarr[1,0].yaxis.labelpad - pad, 0),xycoords=axarr[1,0].yaxis.label, textcoords='offset points',size='large', ha='right', va='center')
axarr[2,0].annotate("Residuals", xy=(0, 0.5), xytext=(-axarr[2,0].yaxis.labelpad - pad, 0),xycoords=axarr[2,0].yaxis.label, textcoords='offset points',size='large', ha='right', va='center')

for col in xrange(npasses):
	# X-Axis Label
	axarr[0,col].annotate(inps.los[col],xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')
	# Synthetic
	axarr[0,col].set_axis_bgcolor('black')
	plot_varres_scatter(axarr[0,col],par_D,data_list[col].lon(),data_list[col].lat(),syntheticdata[col],7.5,cmap='gist_rainbow_r')
	axarr[0,col].scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)
	# Data
	axarr[1,col].set_axis_bgcolor('black')
	plot_varres_scatter(axarr[1,col],par_D,data_list[col].lon(),data_list[col].lat(),data_list[col].losd(),7.5,None,cmap='gist_rainbow_r')
	axarr[1,col].scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)
	# Residuals
	axarr[2,col].set_axis_bgcolor('black')
	plot_varres_scatter(axarr[2,col],par_D,data_list[col].lon(),data_list[col].lat(),residualdata[col],7.5,None,0,maxdiff,'CMRmap')
	axarr[2,col].scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)
	
	plt.setp(axarr[0,col].get_xticklabels(), visible=False)
	plt.setp(axarr[1,col].get_xticklabels(), visible=False)
	if col > 0:
		plt.setp(axarr[0,col].get_yticklabels(), visible=False)
		plt.setp(axarr[1,col].get_yticklabels(), visible=False)
		plt.setp(axarr[2,col].get_yticklabels(), visible=False)

plt.show()

#crediblejointplot(trace.get_values('z0'),trace.get_values('semimajor'),r"Depth $m$",r"Semi-major axis $m$",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('aspectratio'),r"Depth $m$",r"Apsect Ratio",'r')
#crediblejointplot(trace.get_values('semimajor'),trace.get_values('aspectratio'),r"Semi-major axis $m$",r"Aspect Ratio",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('DP_mu'),r"Depth $m$",r"Excess Pressure",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('mu'),r"Depth $m$",r"Shear Modulus $Pa$",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('nu'),r"Depth $m$",r"Poisson's Ratio $m^3$",'r')
#crediblejointplot(trace.get_values('x0'),trace.get_values('y0'),r"Easting $m$",r"Northing $m$",'g',True)


z0raw = trace.get_values('z0', burn=burnin)
x0raw = trace.get_values('x0', burn=burnin)
y0raw = trace.get_values('y0', burn=burnin)
radiusraw = trace.get_values('radius', burn=burnin)

crediblejointplot(radiusraw,z0raw,r"Radius $m$",r"Depth $m$",'c')
crediblejointplot(x0raw-x0raw.mean(),y0raw-y0raw.mean(),r"Easting Offset$m$",r"Northing Offset $m$",'c',True)
crediblejointplot(z0raw,x0raw-x0raw.mean(),r"Depth $m$",r"Easting Offset $m$",'c',True)
crediblejointplot(z0raw,y0raw-y0raw.mean(),r"Depth $m$",r"Northing Offset$m$",'c',True)
crediblejointplot(radiusraw,x0raw-x0raw.mean(),r"Radius $m$",r"Easting Offset $m$",'c',True)
crediblejointplot(radiusraw,y0raw-y0raw.mean(),r"Radiis $m$",r"Northing Offset $m$",'c',True)


