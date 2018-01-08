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
T025D = np.loadtxt('T025D_utm.txt',skiprows=2)
T130A = np.loadtxt('T130A_utm.txt',skiprows=2)
T131A = np.loadtxt('T131A_utm.txt',skiprows=2)

# convert cm to m for los values and set minimum error to 0.01
T025D[:,5] *= 0.01
T130A[:,5] *= 0.01
T131A[:,5] *= 0.01

T025D[:,6] *= 0.01
T130A[:,6] *= 0.01
T131A[:,6] *= 0.01

T025D[:,6] = np.maximum(T025D[:,6],0.001)
T130A[:,6] = np.maximum(T130A[:,6],0.001)
T131A[:,6] = np.maximum(T131A[:,6],0.001)


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


def mogi(coeffs,x,y):
    """evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,dV)"""
    x0,y0,z0,dV = coeffs
    dx = x - x0
    dy = y - y0
    dz = z0
    c = dV * 3. / (4. * math.pi)
    # or equivalently c= (3/4) a^3 dP / rigidity
    # where a = sphere radius, dP = delta Pressure
    r2 = dx*dx + dy*dy + dz*dz
    C = c / pow(r2, 1.5)
    return (C*dx,C*dy,C*dz)



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
dem_T025D -= dem_max
dem_T130A -= dem_max
dem_T131A -= dem_max

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

# 500m std deviations on priors
hstd = 1000.0 #/ (3600 * 30)
vstd = 1500.0

def mogiDEM(x0_,y0_,z0_,dV_,x,y,dem):
	"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,dV)"""
	#x0,y0,z0,dV = coeffs
	dx = x - x0_
	dy = y - y0_
	dz = (dem + z0_).clip(0,1e4) # max 1e4m depth
	c = dV_ * 3. / (4. * math.pi)
	# or equivalently c= (3/4) a^3 dP / rigidity
	# where a = sphere radius, dP = delta Pressure
	r2 = dx*dx + dy*dy + dz*dz
	C = c / (r2 ** 1.5)
	return (C*dx,C*dy,C*dz)

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
print("semimajor = ",semimajor)
excesspressure = 3 # dimensionless excess pressure / shear modulus
mu = (2500 ** 2) * 2500 # Vs^2 * density
nu = 0.25

(T025D_dE, T025D_dN, T025D_dZ) = yangmodel(x0,y0,z0,semimajor,1,excesspressure,mu,nu,90,0,T025D_E_utm,T025D_N_utm,dem_T025D)
T025D_los = (T025D[:,8]*T025D_dE + T025D[:,9]*T025D_dN + T025D[:,10]*T025D_dZ)
T025D[:,5]
T025D[:,6]
		
(T131A_dE, T131A_dN, T131A_dZ) = yangmodel(x0,y0,z0,semimajor,1,excesspressure,mu,nu,90,0,T131A_E_utm,T131A_N_utm,dem_T131A)
T131A_los = (T131A[:,8]*T131A_dE + T131A[:,9]*T131A_dN + T131A[:,10]*T131A_dZ)
T131A[:,5]
T131A[:,6]

(T130A_dE, T130A_dN, T130A_dZ) = yangmodel(x0,y0,z0,semimajor,1,excesspressure,mu,nu,90,0,T130A_E_utm,T130A_N_utm,dem_T130A)
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

T025Dmask = np.zeros_like(T025Dcor)
T025Dmask[T025Dcor < 0.15] = 1
T131Amask = np.zeros_like(T131Acor)
T131Amask[T131Acor < 0.15] = 1
T130Amask = np.zeros_like(T130Acor)
T130Amask[T130Acor < 0.15] = 1

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
	cax = ax.scatter(x,y,s=1.5,c=v,marker=(0,3,0),linewidth=0,vmin=vmin,vmax=vmax,cmap=cmap) # s=<markersize>
	ax.yaxis.set_major_formatter(axisformatter)
	ax.xaxis.set_major_formatter(axisformatter)
	for tick in ax.get_xticklabels():
	    tick.set_rotation(45)
	plt.colorbar(cax,ax=ax)

fig = plt.figure()

pad=5

ax1 = fig.add_subplot(331,axisbg='black')
ax1.annotate("T025D",xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')
ax1.annotate("Synthetic LOS", xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - pad, 0),xycoords=ax1.yaxis.label, textcoords='offset points',size='large', ha='right', va='center')

plot_varres_scatter(ax1,par_D,T025D[:,3],T025D[:,4],T025D_los,7.5,cmap='gist_rainbow_r')
ax2 = fig.add_subplot(334,sharex=ax1,axisbg='black')
ax2.annotate("InSAR Data", xy=(0, 0.5), xytext=(-ax2.yaxis.labelpad - pad, 0),xycoords=ax2.yaxis.label, textcoords='offset points',size='large', ha='right', va='center')
plot_varres_scatter(ax2,par_D,T025D[:,3],T025D[:,4],T025D[:,5],7.5,T025Dmask,cmap='gist_rainbow_r')
ax2b = fig.add_subplot(337,sharex=ax1,axisbg='black')
ax2b.annotate("Residuals", xy=(0, 0.5), xytext=(-ax2b.yaxis.labelpad - pad, 0),xycoords=ax2b.yaxis.label, textcoords='offset points',size='large', ha='right', va='center')
#diff1 = np.absolute((T025D_los - T025D[:,5])/T025D[:,6])
diff1 = np.absolute((T025D_los - T025D[:,5]))
plot_varres_scatter(ax2b,par_D,T025D[:,3],T025D[:,4],diff1,7.5,None,0,diff1.max(),'CMRmap')

ax3 = fig.add_subplot(332,sharey=ax1,axisbg='black')
ax3.annotate("T131A",xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')

plot_varres_scatter(ax3,par_A,T131A[:,3],T131A[:,4],T131A_los,7.5,cmap='gist_rainbow_r')
ax4 = fig.add_subplot(335,sharex=ax3,sharey=ax2,axisbg='black')
plot_varres_scatter(ax4,par_A,T131A[:,3],T131A[:,4],T131A[:,5],7.5,T131Amask,cmap='gist_rainbow_r')
ax4b = fig.add_subplot(338,sharex=ax3,sharey=ax2b,axisbg='black')
#diff2 = np.absolute((T131A_los - T131A[:,5])/T131A[:,6])
diff2 = np.absolute((T131A_los - T131A[:,5]))
plot_varres_scatter(ax4b,par_D,T131A[:,3],T131A[:,4],diff2,7.5,None,0,diff2.max(),'CMRmap')

ax5 = fig.add_subplot(333,sharey=ax1,axisbg='black')
ax5.annotate("T130A",xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction', textcoords='offset points',size='large', ha='center', va='baseline')

plot_varres_scatter(ax5,par_A2,T130A[:,3],T130A[:,4],T130A_los,7.5,cmap='gist_rainbow_r')
ax6 = fig.add_subplot(336,sharex=ax5,sharey=ax2,axisbg='black')
plot_varres_scatter(ax6,par_A2,T130A[:,3],T130A[:,4],T130A[:,5],7.5,T130Amask,cmap='gist_rainbow_r')
ax6b = fig.add_subplot(339,sharex=ax5,sharey=ax2b,axisbg='black')
#diff3 = np.absolute((T130A_los - T130A[:,5])/T130A[:,6])
diff3 = np.absolute((T130A_los - T130A[:,5]))
plot_varres_scatter(ax6b,par_D,T130A[:,3],T130A[:,4],diff3,7.5,None,0,diff3.max(),'CMRmap')

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


