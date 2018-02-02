import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import interpolate
from scipy import signal
import math

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

T025D[:,6] = np.sqrt(T025D[:,6]**2 + T025Ddiagcov)
T130A[:,6] = np.sqrt(T130A[:,6]**2 + T130Adiagcov)
T131A[:,6] = np.sqrt(T131A[:,6]**2 + T131Adiagcov)

T025D[:,6] *= 0.01
T130A[:,6] *= 0.01
T131A[:,6] *= 0.01

# interpolate the dem for each of the varres grids
def linspaceb(start,step,num):
    return np.linspace(float(start),float(start)+float(step)*(num-1),float(num))

def plotdata(d, par, cmap):
    """Function to make a simple plot of a data array"""
    lon = linspaceb(float(par['corner_lon']),float(par['post_lon']),int(par['width']))
    lat = linspaceb(float(par['corner_lat']),float(par['post_lat']),int(par['nlines']))
    dd = ma.masked_where(np.isnan(d),d)
    axisformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    plt.pcolormesh(lon,lat,dd,cmap=cmap,vmin=dd.min(),vmax=dd.max())
    ax = plt.gca()
    ax.yaxis.set_major_formatter(axisformatter)
    ax.xaxis.set_major_formatter(axisformatter)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    #print(dd.min(),dd.max())
    #plt.matshow(d, cmap)
    plt.colorbar()
    plt.show()

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

T131Acor = readgamma('../T131A/20170829-20170912_HH_4rlks_flat_eqa.cc', par_A)
T130Acor = readgamma('../T130A/20170727-20170907_HH_4rlks_flat_eqa.cc', par_A2)
T025Dcor = readgamma('../T025D/20170831-20170928_HH_4rlks_flat_eqa.cc', par_D)

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

print("DEM offset = " + str(-dem_max))

print T025D.shape
print dem_T025D.shape
#dem_T025D = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T025D[:,3],T025D[:,4]),method='linear')
#dem_T130A = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T130A[:,3:4]),method='linear')
#dem_T131A = interpolate.griddata((demlon,demlat),dem_D.flatten(),(T131A[:,3:4]),method='linear')

ld = (129.074,41.299,500)
ustc = (129.074200,41.298200,0)
uri = (129.067000,41.309000,700)

def long2UTM(lon):
        return (math.floor((float(lon) + 180)/6) % 60) + 1

# set up pymc3
import pyproj

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
hstd = 1000.0 #/ (3600 * 30)
vstd = 1500.0

from yang import *

#dV = 2499994.04188822
#y0 = 14572103.930080593
#x0 = 505083.11674481846
#z0 = 3368.625140496614

#{'y0_interval__': array(2.027965335976719), 'dV_interval__': array(-2.56124060440146), 'z0_interval__': array(-0.4475748581034097), 'x0_interval__': array(0.9861819631341313), 'dV': array(358374.75741076434), 'y0': array(14572719.8071666), 'x0': array(506651.9703539055), 'z0': array(169.81255920567833)}

dV = 358374.75741076434
y0 = 14572719.8071666
x0 = 506651.9703539055
z0 = 169.81255920567833

dV = 356800
x0 = 506707.43
y0 = 14572628.5
z0 = 131.5 

dV = 1440550
z0 = 583.5
y0 = 14572290.8
x0 = 506430.6

dV = 1440478.9942677645
y0 = 14572291.053122176
x0 = 506430.6330203693
z0 = 583.4407811820657

dV = 1395894.597
y0 = 14572290.963 
x0 = 506421.515
z0 = 412.539

dV = 2499996.9679989633 
y0 = 14572238.46200912 
x0 = 505936.39546445553
z0 = 2386.058383115763

dV=829123.922894638
y0=14572601.689286707
x0=506316.24905746605 
z0=893.939494952472

semimajor = np.power(dV / (4*math.pi/3),1.0/3.0) 
DP_mu = 3 # dimensionless excess pressure / shear modulus
mu = (2500 ** 2) * 2500 # Vs^2 * density
nu = 0.25

semimajor = 50.015588190175336
DP_mu = 5.051962086133976
theta = 44.993704312584974
y0 = 14571952.402948132
x0 = 506195.304077638
z0 = 1499.999759279775
aspectratio = 0.99 #1.4935265197066823


x0 = 506394.425 
y0 = 14572444.497 
z0 = 452.465
semimajor = 151.566
aspectratio = 0.044
DP_mu = 52.315
mu = 48547815436.490 
nu = 0.299
theta = 71.178    
phi = 174.278   


x0 = 506258.499   
y0 = 14572471.607   
z0 = 575.095   
semimajor = 108.572 
aspectratio = 0.072
DP_mu = 55.492 
mu = 64305636318.070 
nu = 0.219     
theta = 71.310     
phi = 172.936 

x0= 506366.141    
y0= 14572321.057  
z0= 465.641 
radius= 71.884 

modeltype='Mogi'

#import pickle
#import theano.tensor as T
#import seaborn as sns
#import pymc3 as pm

#trace = pickle.load(open("trace.p","rb"))

if modeltype=='Yang':
	# last Yang
	x0 = 506387.798
	y0 = 14572395.005
	z0 = 340.288
	semimajor = 51.691
	aspectratio = 0.104
	DP_mu = 256.723
	mu = 46786837013.641
	nu = 0.300
	theta = 63.087
	phi = 191.166
else:
	# last Mogi
	x0 = 506366.147    #   0.637            0.005            [506364.907, 506367.379]
	y0 = 14572321.053  #   0.568            0.005            [14572319.977, 14572322.201]
	z0 = 400 #z0 = 465.648       #   0.540            0.004            [464.592, 466.715]
	radius = 55 #radius = 71.884    #       0.018            0.000            [71.848, 71.918]
	#print "{'y0': array(14572347.107446698), 'x0': array(506379.4341817431), 'radius': array(70.35783911344616), 'z0': array(455.17884772392574)}" # 
	#print(pm.summary(trace)) 
	#argmaxlike = trace.get_values('like',burn=2000).argmax()
	#x0 = trace.get_values('x0',burn=2000)[argmaxlike]
	#y0 = trace.get_values('y0',burn=2000)[argmaxlike]
	#z0 = trace.get_values('z0',burn=2000)[argmaxlike]
	#radius = trace.get_values('radius',burn=2000)[argmaxlike]
	print "y0: %f x0: %f radius %f z0 %f"%(y0,x0,radius,z0)

x0ll, y0ll = pyproj.transform(utm,wgs,x0,y0)

print("semimajor = ",semimajor)

if modeltype=='Yang':
	(T025D_dE, T025D_dN, T025D_dZ) = yangmodel(x0,y0,z0,semimajor,aspectratio,DP_mu,mu,nu,theta,phi,T025D_E_utm,T025D_N_utm,dem_T025D)
	(T131A_dE, T131A_dN, T131A_dZ) = yangmodel(x0,y0,z0,semimajor,aspectratio,DP_mu,mu,nu,theta,phi,T131A_E_utm,T131A_N_utm,dem_T131A)
	(T130A_dE, T130A_dN, T130A_dZ) = yangmodel(x0,y0,z0,semimajor,aspectratio,DP_mu,mu,nu,theta,phi,T130A_E_utm,T130A_N_utm,dem_T130A)
else:
	import mogi
	#(T025D_dE, T025D_dN, T025D_dZ) = mogiTopoCorrected(x0,y0,z0,radius,T025D_E_utm,T025D_N_utm,dem_T025D)
	#(T131A_dE, T131A_dN, T131A_dZ) = mogiTopoCorrected(x0,y0,z0,radius,T131A_E_utm,T131A_N_utm,dem_T131A)
	#(T130A_dE, T130A_dN, T130A_dZ) = mogiTopoCorrected(x0,y0,z0,radius,T130A_E_utm,T130A_N_utm,dem_T130A)
	""" Interpolate a UTM grid from the above DEM"""
	""" First get range of utm coords, 
	    then interp using 7.5m grid spacing """
	
	deltaE = float(par_D['post_lon'])*3600*30 * 4
	deltaN = float(par_D['post_lat'])*3600*30 * 4
	min_Eo = min(T025D_E_utm.min(), T130A_E_utm.min(), T131A_E_utm.min()) - abs(deltaE)
	max_Eo = max(T025D_E_utm.max(), T130A_E_utm.max(), T131A_E_utm.max()) + abs(deltaE)
	min_No = min(T025D_N_utm.min(), T130A_N_utm.min(), T131A_N_utm.min()) - abs(deltaN)
	max_No = max(T025D_N_utm.max(), T130A_N_utm.max(), T131A_N_utm.max()) + abs(deltaN)
	# Instead of constraining to the minimum area, construct a 2^n x 2^n grid that covers the area
	# This is to make the FFT most efficient.
	# We already have desirable area. Choose the largest dimension, expand the other d to it and then grid by 512x512
	smallestd = min(max_Eo-min_Eo,max_No-min_No)
	addE = 0.5*(smallestd - (max_Eo-min_Eo))
	addN = 0.5*(smallestd - (max_No-min_No))
	print "Add E,N = " + str((addE,addN))
	min_E = min_Eo - addE
	max_E = max_Eo + addE
	min_N = min_No - addN
	max_N = max_No + addN
	# Grid it
	gridspacing = 512
	deltaEN = smallestd / gridspacing 
	print (min_E,max_E,max_E-min_E)
	print (min_N,max_N,max_N-min_N)
	#demE = np.arange(min_E,max_E,deltaE)
	#demN = np.arange(max_N,min_N,deltaN) # should be 7.5 metres
	demE = np.linspace(min_E,max_E,gridspacing)
	demN = np.linspace(min_N,max_N,gridspacing)
	#demN = np.flipud(np.linspace(min_N,max_N,gridspacing))
	demEE,demNN = np.meshgrid(demE,demN)
	demLon, demLat = pyproj.transform(utm,wgs,demEE,demNN)
	dem_utm_wgs_grid = np.column_stack((demLon.flatten(),demLat.flatten()))
	dem_D_utm = f(dem_utm_wgs_grid)
	gridshape = (demN.shape[0],demE.shape[0])
	dem_D_utm = dem_D_utm.reshape(gridshape)
	
	#utm_f = RegularGridInterpolator((demE,demN)),np.flipud(dem_D_utm).T)
	
	def makegaussian(xs,ys,sigma,mu):
		x, y = np.meshgrid(np.linspace(-1,1,xs), np.linspace(-1,1,ys))
		d = np.sqrt(x*x+y*y)
		return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
	
	#dem_D_utm = np.ones_like(dem_D_utm)
	foxyminx = min(dem_D_utm.shape[1],dem_D_utm.shape[0])
	#dem_D_utm = np.zeros_like(dem_D_utm)
	# roffl = int((dem_D_utm.shape[0]-foxyminx)/2)
	# coffl = int((dem_D_utm.shape[1]-foxyminx)/2)
	# roffr = dem_D_utm.shape[0]-roffl-foxyminx
	# coffr = dem_D_utm.shape[1]-coffl-foxyminx
	# print(roffl,roffr,coffl,coffr)
	# print dem_D_utm.shape
	dem_D_utm = makegaussian(foxyminx,foxyminx,1.0,0.0) * 500
	# compare
	fig = plt.figure()
	fa1 = fig.add_subplot(211)
	fa1.imshow(dem_D)
	fa2 = fig.add_subplot(212)
	fa2.imshow(dem_D_utm)
	plt.show()
	
	(dem_dE, dem_dN, dem_dZ) = mogi.mogiTopoCorrected(x0,y0,z0,radius,demEE.reshape(gridshape),demNN.reshape(gridshape),dem_D_utm,deltaEN)
	#print dem_dE
	#print dem_dN
	#print dem_dZ
	fig = plt.figure()
	fa1 = fig.add_subplot(131)
	cfa1 = fa1.imshow(np.real(dem_dE))
	fa2 = fig.add_subplot(132)
	cfa2 = fa2.imshow(np.real(dem_dN))
	fa3 = fig.add_subplot(133)
	cfa3 = fa3.imshow(np.real(dem_dZ))
	plt.colorbar(cfa1, ax=fa1)
	plt.colorbar(cfa2, ax=fa2)
	plt.colorbar(cfa3, ax=fa3)
	plt.show()

	T025D_utmloc = np.column_stack((T025D_E_utm,T025D_N_utm))
	T130A_utmloc = np.column_stack((T130A_E_utm,T130A_N_utm))
	T131A_utmloc = np.column_stack((T131A_E_utm,T131A_N_utm))

	print (min_E,max_E)
	print (min_N,max_N)
	print (demE.min(),demE.max())
	print (demN.min(),demN.max())
	print (T025D_utmloc[:,0].min(),T025D_utmloc[:,0].max())
	print (T025D_utmloc[:,1].min(),T025D_utmloc[:,1].max())
	utm_synthetic_dE = RegularGridInterpolator((demE,demN),dem_dE.T)
	utm_synthetic_dN = RegularGridInterpolator((demE,demN),dem_dN.T)
	utm_synthetic_dZ = RegularGridInterpolator((demE,demN),dem_dZ.T)
	#utm_synthetic_dE = RegularGridInterpolator((demE,np.flipud(demN)),np.flipud(dem_dE).T)
	#utm_synthetic_dN = RegularGridInterpolator((demE,np.flipud(demN)),np.flipud(dem_dN).T)
	#utm_synthetic_dZ = RegularGridInterpolator((demE,np.flipud(demN)),np.flipud(dem_dZ).T)
	T025D_dE = utm_synthetic_dE(T025D_utmloc)
	T025D_dN = utm_synthetic_dN(T025D_utmloc)
	T025D_dZ = utm_synthetic_dZ(T025D_utmloc)
	T130A_dE = utm_synthetic_dE(T130A_utmloc)
	T130A_dN = utm_synthetic_dN(T130A_utmloc)
	T130A_dZ = utm_synthetic_dZ(T130A_utmloc)
	T131A_dE = utm_synthetic_dE(T131A_utmloc)
	T131A_dN = utm_synthetic_dN(T131A_utmloc)
	T131A_dZ = utm_synthetic_dZ(T131A_utmloc)
	



T025D_los = (T025D[:,8]*T025D_dE + T025D[:,9]*T025D_dN + T025D[:,10]*T025D_dZ)
T025D[:,5]
T025D[:,6]
		
T131A_los = (T131A[:,8]*T131A_dE + T131A[:,9]*T131A_dN + T131A[:,10]*T131A_dZ)
T131A[:,5]
T131A[:,6]

T130A_los = (T130A[:,8]*T130A_dE + T130A[:,9]*T130A_dN + T130A[:,10]*T130A_dZ)
T130A[:,5]
T130A[:,6]

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

cormask = 0.3 # 0.15

T025Dmask = np.zeros_like(T025Dcor)
T025Dmask[T025Dcor < cormask] = 1
T131Amask = np.zeros_like(T131Acor)
T131Amask[T131Acor < cormask] = 1
T130Amask = np.zeros_like(T130Acor)
T130Amask[T130Acor < cormask] = 1

from scipy.interpolate import griddata

def plot_varres_utm(ax,x,y,v,size,mask=None):
	xx = np.arange(x.min(),x.max(),size)
	yy = np.arange(y.min(),y.max(),size)
	vv = griddata((x,y),v,(xx[None,:],yy[:,None]),method='linear')
	if mask is not None:
		vv[mask==1] = np.nan
	axisformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
	cax = ax.pcolormesh(xx,yy,vv,vmin=vv.min(),vmax=vv.max(), cmap='gist_rainbow_r')
	ax.yaxis.set_major_formatter(axisformatter)
	ax.xaxis.set_major_formatter(axisformatter)
	for tick in ax.get_xticklabels():
	    tick.set_rotation(45)
	plt.colorbar(cax,ax=ax)

#fig = plt.figure()
#ax1 = fig.add_subplot(231)
#plot_varres(ax1,T025D_E_utm,T025D_N_utm,T025D_los,7.5)
#ax2 = fig.add_subplot(234,sharex=ax1)
#plot_varres(ax2,T025D_E_utm,T025D_N_utm,T025D[:,5],7.5,T025Dmask)
#ax3 = fig.add_subplot(232,sharey=ax1)
#plot_varres(ax3,T131A_E_utm,T131A_N_utm,T131A_los,7.5)
#ax4 = fig.add_subplot(235,sharex=ax3,sharey=ax2)
#plot_varres(ax4,T131A_E_utm,T131A_N_utm,T131A[:,5],7.5,T131Amask)
#ax5 = fig.add_subplot(233,sharey=ax1)
#plot_varres(ax5,T130A_E_utm,T130A_N_utm,T130A_los,7.5)
#ax6 = fig.add_subplot(236,sharex=ax5,sharey=ax2)
#plot_varres(ax6,T130A_E_utm,T130A_N_utm,T130A[:,5],7.5,T130Amask)
#plt.show()

def plot_varres(ax,par,x,y,v,size,mask=None,vmin=-0.36,vmax=0.2,cmap='gist_rainbow_r'):
	xx = linspaceb(float(par['corner_lon']),float(par['post_lon']),int(par['width']))
	yy = linspaceb(float(par['corner_lat']),float(par['post_lat']),int(par['nlines']))
	vv = griddata((x,y),v,(xx[None,:],yy[:,None]),method='linear')
	if mask is not None:
		vv[mask==1] = np.nan
	vvm = ma.masked_where(np.isnan(vv),vv)
	axisformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
	#cax = ax.pcolormesh(xx,yy,vvm,vmin=np.ma.minimum(vvm),vmax=np.ma.maximum(vvm), cmap='gist_rainbow_r')
	cax = ax.pcolormesh(xx,yy,vvm,vmin=vmin,vmax=vmax, cmap=cmap)
	ax.yaxis.set_major_formatter(axisformatter)
	ax.xaxis.set_major_formatter(axisformatter)
	for tick in ax.get_xticklabels():
	    tick.set_rotation(45)
	plt.colorbar(cax,ax=ax)

def plot_varres_scatter(ax,par,x,y,v,size,mask=None,vmin=-0.36,vmax=0.36,cmap='gist_rainbow_r'):
	axisformatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
	cax = ax.scatter(x,y,s=8.0,c=v,marker=(0,3,0),linewidth=0,vmin=vmin,vmax=vmax,cmap=cmap) # s=<markersize>
	ax.yaxis.set_major_formatter(axisformatter)
	ax.xaxis.set_major_formatter(axisformatter)
	for tick in ax.get_xticklabels():
	    tick.set_rotation(45)
	plt.colorbar(cax,ax=ax)

fig = plt.figure()

pad=5

#diff1 = np.absolute((T025D_los - T025D[:,5])/T025D[:,6])
diff1 = np.absolute((T025D_los - T025D[:,5]))
#diff2 = np.absolute((T131A_los - T131A[:,5])/T131A[:,6])
diff2 = np.absolute((T131A_los - T131A[:,5]))
#diff3 = np.absolute((T130A_los - T130A[:,5])/T130A[:,6])
diff3 = np.absolute((T130A_los - T130A[:,5]))
maxdiff = max(diff1.max(),diff2.max(),diff3.max())

ax1 = fig.add_subplot(331,axisbg='black')
ax1.annotate("T025D",xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')
ax1.annotate("Synthetic LOS", xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - pad, 0),xycoords=ax1.yaxis.label, textcoords='offset points',size='large', ha='right', va='center')

plot_varres_scatter(ax1,par_D,T025D[:,3],T025D[:,4],T025D_los,7.5,cmap='gist_rainbow_r')
ax1.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax2 = fig.add_subplot(334,sharex=ax1,axisbg='black')
ax2.annotate("InSAR Data", xy=(0, 0.5), xytext=(-ax2.yaxis.labelpad - pad, 0),xycoords=ax2.yaxis.label, textcoords='offset points',size='large', ha='right', va='center')
plot_varres_scatter(ax2,par_D,T025D[:,3],T025D[:,4],T025D[:,5],7.5,T025Dmask,cmap='gist_rainbow_r')
ax2.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax2b = fig.add_subplot(337,sharex=ax1,axisbg='black')
ax2b.annotate("Residuals", xy=(0, 0.5), xytext=(-ax2b.yaxis.labelpad - pad, 0),xycoords=ax2b.yaxis.label, textcoords='offset points',size='large', ha='right', va='center')
plot_varres_scatter(ax2b,par_D,T025D[:,3],T025D[:,4],diff1,7.5,None,0,maxdiff,'CMRmap')
ax2b.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax3 = fig.add_subplot(332,sharey=ax1,axisbg='black')
ax3.annotate("T131A",xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')
plot_varres_scatter(ax3,par_A,T131A[:,3],T131A[:,4],T131A_los,7.5,cmap='gist_rainbow_r')
ax3.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax4 = fig.add_subplot(335,sharex=ax3,sharey=ax2,axisbg='black')
plot_varres_scatter(ax4,par_A,T131A[:,3],T131A[:,4],T131A[:,5],7.5,T131Amask,cmap='gist_rainbow_r')
ax4.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax4b = fig.add_subplot(338,sharex=ax3,sharey=ax2b,axisbg='black')
plot_varres_scatter(ax4b,par_D,T131A[:,3],T131A[:,4],diff2,7.5,None,0,maxdiff,'CMRmap')
ax4b.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax5 = fig.add_subplot(333,sharey=ax1,axisbg='black')
ax5.annotate("T130A",xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')
plot_varres_scatter(ax5,par_A2,T130A[:,3],T130A[:,4],T130A_los,7.5,cmap='gist_rainbow_r')
ax5.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax6 = fig.add_subplot(336,sharex=ax5,sharey=ax2,axisbg='black')
plot_varres_scatter(ax6,par_A2,T130A[:,3],T130A[:,4],T130A[:,5],7.5,T130Amask,cmap='gist_rainbow_r')
ax6.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

ax6b = fig.add_subplot(339,sharex=ax5,sharey=ax2b,axisbg='black')
plot_varres_scatter(ax6b,par_D,T130A[:,3],T130A[:,4],diff3,7.5,None,0,maxdiff,'CMRmap')
ax6b.scatter(x0ll,y0ll,s=128,c='white',marker='+',linewidth=1)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax6.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax4b.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)
plt.setp(ax6b.get_yticklabels(), visible=False)

plt.show()

#crediblejointplot(trace.get_values('z0'),trace.get_values('semimajor'),r"Depth $m$",r"Semi-major axis $m$",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('aspectratio'),r"Depth $m$",r"Apsect Ratio",'r')
#crediblejointplot(trace.get_values('semimajor'),trace.get_values('aspectratio'),r"Semi-major axis $m$",r"Aspect Ratio",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('DP_mu'),r"Depth $m$",r"Excess Pressure",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('mu'),r"Depth $m$",r"Shear Modulus $Pa$",'r')
#crediblejointplot(trace.get_values('z0'),trace.get_values('nu'),r"Depth $m$",r"Poisson's Ratio $m^3$",'r')
#crediblejointplot(trace.get_values('x0'),trace.get_values('y0'),r"Easting $m$",r"Northing $m$",'g',True)

import scipy
import pandas as pd

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

z0raw = trace.get_values('z0', burn=2000)
x0raw = trace.get_values('x0', burn=2000)
y0raw = trace.get_values('y0', burn=2000)
radiusraw = trace.get_values('radius', burn=2000)

crediblejointplot(radiusraw,z0raw,r"Radius $m$",r"Depth $m$",'c')
crediblejointplot(x0raw-x0raw.mean(),y0raw-y0raw.mean(),r"Easting Offset$m$",r"Northing Offset $m$",'c',True)
crediblejointplot(z0raw,x0raw-x0raw.mean(),r"Depth $m$",r"Easting Offset $m$",'c',True)
crediblejointplot(z0raw,y0raw-y0raw.mean(),r"Depth $m$",r"Northing Offset$m$",'c',True)
crediblejointplot(radiusraw,x0raw-x0raw.mean(),r"Radius $m$",r"Easting Offset $m$",'c',True)
crediblejointplot(radiusraw,y0raw-y0raw.mean(),r"Radiis $m$",r"Northing Offset $m$",'c',True)


