import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
from scipy import interpolate
from scipy import signal
import math
import theano
import theano.tensor as T
import seaborn as sns
from tempfile import mkdtemp

#solver='MAPNUTS'
solver='SMC'

#theano.config.mode='FAST_COMPILE'
theano.config.int_division='raise'

test_folder = mkdtemp(prefix='SMCT')

# Headers:
# Number xind yind east north data err wgt Elos Nlos Ulos
# east/north are actually lon and lat
T025D = np.loadtxt('T025D_utme.txt',skiprows=2)
T130A = np.loadtxt('T130A_utme.txt',skiprows=2)
T131A = np.loadtxt('T131A_utme.txt',skiprows=2)

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

# 500m std deviations on priors
hstd = 1000.0 #/ (3600 * 30)

basic_model = pm.Model()

with basic_model:
	x0 = pm.Uniform('x0',ldE-hstd,ldE+hstd,transform=None)
	y0 = pm.Uniform('y0',ldN-hstd,ldN+hstd,transform=None)
	z0 = pm.Uniform('z0',0,3000,transform=None)
	semimajor = pm.Uniform('semimajor',0,5e2,transform=None)
	aspectratio = pm.Uniform('aspectratio',0.01,0.99,transform=None)
	DP_mu = pm.Uniform('DP_mu',0,1e3,transform=None)
	mu = pm.Uniform('mu',1e10,1e11,transform=None)
	nu = pm.Uniform('nu',0.2,0.3,transform=None)
	theta = pm.Uniform('theta',0,89.99,transform=None)
	phi = pm.Uniform('phi',0,359.99,transform=None)
	#theta = 0 #89.999
	#mu = theano.shared(2500**3)
	#nu = theano.shared(0.25)

	def mogiDEM(x0_,y0_,z0_,dV_,x,y,dem):
		"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,dV)"""
		#x0,y0,z0,dV = coeffs
		dx = x - x0_
		dy = y - y0_
		dz = (dem + z0_).clip(0,1e5) # max 1e4m depth
		#dz = z0 # max 1e4m depth
		c = dV_ * 3. / (4. * math.pi)
		# or equivalently c= (3/4) a^3 dP / rigidity
		# where a = sphere radius, dP = delta Pressure
		r2 = dx*dx + dy*dy + dz*dz
		C = c / (r2 ** 1.5)
		return (C*dx,C*dy,C*dz)
		
	from yang import *

	def synthetic1(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_):
		(T025D_dE, T025D_dN, T025D_dZ) = yangmodel(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_,T025D_E_utm,T025D_N_utm,dem_T025D)
		#T025D_dE_p = T.printing.Print('T025D_dE')(T025D_dE)
		return (((((T025D[:,8]*T025D_dE + T025D[:,9]*T025D_dN + T025D[:,10]*T025D_dZ) - T025D[:,5]) / T025D[:,6])**2).sum())
		
	def synthetic2(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_):
		(T131A_dE, T131A_dN, T131A_dZ) = yangmodel(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_,T131A_E_utm,T131A_N_utm,dem_T131A)
		return (((((T131A[:,8]*T131A_dE + T131A[:,9]*T131A_dN + T131A[:,10]*T131A_dZ) - T131A[:,5]) / T131A[:,6])**2).sum())

	def synthetic3(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_):
		(T130A_dE, T130A_dN, T130A_dZ) = yangmodel(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_,T130A_E_utm,T130A_N_utm,dem_T130A)
		#T130A_dE_p = T.printing.Print('T130A_dE')(T130A_dE)
		#T130A_dN_p = T.printing.Print('T130A_dN')(T130A_dN)
		#T130A_dZ_p = T.printing.Print('T130A_dZ')(T130A_dZ)
		return (((((T130A[:,8]*T130A_dE + T130A[:,9]*T130A_dN + T130A[:,10]*T130A_dZ) - T130A[:,5]) / T130A[:,6])**2).sum())

	def logp(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_):
		#return (synthetic1(x0_,y0_,z0_,dV_)+synthetic2(x0_,y0_,z0_,dV_)) * (-0.5) # for DensityDist: * (-0.5)
		s1 = synthetic1(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_)
		#s1_p = T.printing.Print('s1')(s1)
		s2 = synthetic2(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_)
		#s2_p = T.printing.Print('s2')(s2)
		s3 = synthetic3(x0_,y0_,z0_,semimajor_,aspectratio_,DP_mu_,mu_,nu_,theta_,phi_)
		#s3_p = T.printing.Print('s3')(s3)
		return (s1 + s2) * (-0.5) # for DensityDist: * (-0.5)

	llk = pm.DensityDist('llk',logp,observed={'x0_':x0,'y0_':y0,'z0_':z0,'semimajor_':semimajor,'aspectratio_':aspectratio,'DP_mu_':DP_mu,'mu_':mu,'nu_':nu,'theta_':theta,'phi_':phi})
	#llk = pm.Potential('llk',logp(x0,y0,z0,dV))


	if solver=='SMC':	
		#start = pm.find_MAP(model=basic_model)
		#print("Maximum a-posteriori estimate:") 
		#print(start)
		n_chains = 256
		n_steps = 16
		tune_interval = 8
		n_jobs = 1
		trace = smc.sample_smc(
			n_steps=n_steps,
			n_chains=n_chains,
			tune_interval=tune_interval,
			n_jobs=n_jobs,
			#start=start,
			progressbar=True,
			stage=0,
			homepath=test_folder,
			model=basic_model
			)
	elif solver=='MAPNUTS':
		niter = 5000
	
		# with find_MAP()
		start = pm.find_MAP(model=basic_model)
		print("Maximum a-posteriori estimate:") 
		print(start)
		step = pm.NUTS(scaling=start)
		trace = pm.sample(niter, start = start, step = step)

	elif solver=='NUTS':
		niter = 5000
		# Without find_MAP()
		step = pm.NUTS()
		trace = pm.sample(niter, step = step)

	pm.traceplot(trace)
	ax = plt.gca()
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	ax.get_xaxis().get_major_formatter().set_scientific(False)
	plt.show()

	#pm.traceplot(t, varnames=['x0','y0','z0','dV'])
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

	#crediblekdeplot(trace['z0'],trace['dV'],r"Depth $m$",r"Volume $m^3$")
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

	print trace

	crediblejointplot(trace.get_values('z0'),trace.get_values('semimajor'),r"Depth $m$",r"Semi-major axis $m$",'r')
	crediblejointplot(trace.get_values('z0'),trace.get_values('aspectratio'),r"Depth $m$",r"Apsect Ratio",'r')
	crediblejointplot(trace.get_values('semimajor'),trace.get_values('aspectratio'),r"Semi-major axis $m$",r"Aspect Ratio",'r')
	crediblejointplot(trace.get_values('z0'),trace.get_values('DP_mu'),r"Depth $m$",r"Excess Pressure",'r')
	crediblejointplot(trace.get_values('z0'),trace.get_values('mu'),r"Depth $m$",r"Shear Modulus $Pa$",'r')
	crediblejointplot(trace.get_values('z0'),trace.get_values('nu'),r"Depth $m$",r"Poisson's Ratio $m^3$",'r')
	crediblejointplot(trace.get_values('x0'),trace.get_values('y0'),r"Easting $m$",r"Northing $m$",'g',True)

	#sns.jointplot(x=trace.get_values('z0'),y=trace.get_values('dV'),kind='kde',color='r').set_axis_labels(r"Depth $m$",r"Volume $m^3$")
	#sns.jointplot(x=trace.get_values('x0'),y=trace.get_values('y0'),kind='kde',color='green').set_axis_labels(r"Easting $m$",r"Northing $m$")

	#sns.kdeplot(trace.get_values('x0'),trace.get_values('y0'))
	#plt.xlabel(r"Easting $m$")
	#plt.ylabel(r"Northing $m$")
	#plt.ticklabel_format(useOffset=False)
	#plt.show()

	#sns.kdeplot(trace.get_values('z0'),trace.get_values('dV'))
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
