import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import interpolate
from scipy import signal
import math
import theano.tensor as T
from tempfile import mkdtemp

test_folder = mkdtemp(prefix='SMCT')

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

T025D[:,6] = np.maximum(T025D[:,6],0.01)
T130A[:,6] = np.maximum(T130A[:,6],0.01)
T131A[:,6] = np.maximum(T131A[:,6],0.01)

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
dem_T130A = f(T130A[:,3:5]).flatten()
dem_T131A = f(T131A[:,3:5]).flatten()

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

wgs = pyproj.Proj(init="EPSG:4326")
utm = pyproj.Proj("+proj=utm +zone="+str(long2UTM(ld[0]))+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

# do everything in UTM
ldE, ldN = pyproj.transform(wgs,utm,ld[0],ld[1])
T025D_E_utm, T025D_N_utm = pyproj.transform(wgs,utm,T025D[:,3],T025D[:,4])
T131A_E_utm, T131A_N_utm = pyproj.transform(wgs,utm,T131A[:,3],T131A[:,4])
T130A_E_utm, T130A_N_utm = pyproj.transform(wgs,utm,T130A[:,3],T130A[:,4])

# 500m std deviations on priors
hstd = 1000.0 / (3600 * 30)
vstd = 1000.0

basic_model = pm.Model()


with basic_model:
	#x0 = pm.Normal('x0',mu=ld[0],sd=hstd)
	#y0 = pm.Normal('y0',mu=ld[1],sd=hstd)
	#z0 = pm.Normal('z0',mu=ld[2],sd=vstd)
	#dV = pm.Normal('dV',mu=8e6,sd=2e6)

	x0 = pm.Uniform('x0',ld[0]-hstd,ld[0]+hstd)
	y0 = pm.Uniform('y0',ld[1]-hstd,ld[1]+hstd)
	z0 = pm.Uniform('z0',ld[2]-vstd,ld[2]+vstd)
	dV = pm.Uniform('dV',6e6,1e7)


	x0_print = T.printing.Print('x0')(x0)
	y0_print = T.printing.Print('y0')(y0)
	z0_print = T.printing.Print('z0')(z0)
	dV_print = T.printing.Print('dV')(dV)

	def mogiDEM(x0_,y0_,z0_,dV_,x,y,dem):
		"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,dV)"""
		#x0,y0,z0,dV = coeffs
		dx = x - x0_
		dy = y - y0_
		dz = dem - z0_
		c = dV_ * 3. / (4. * math.pi)
		# or equivalently c= (3/4) a^3 dP / rigidity
		# where a = sphere radius, dP = delta Pressure
		r2 = dx*dx + dy*dy + dz*dz
		C = c / (r2 ** 1.5)
		return (C*dx,C*dy,C*dz)

	# misfit function - assume diagonal variance matrix for now.
	def loglikelihood(x0,y0,z0,dV):
		(T025D_dE, T025D_dN, T025D_dZ) = mogiDEM(x0,y0,z0,dV,T025D_E_utm,T025D_N_utm,dem_T025D)
		(T130A_dE, T130A_dN, T130A_dZ) = mogiDEM(x0,y0,z0,dV,T130A_E_utm,T130A_N_utm,dem_T130A)
		(T131A_dE, T131A_dN, T131A_dZ) = mogiDEM(x0,y0,z0,dV,T131A_E_utm,T131A_N_utm,dem_T131A)
		#T025D_dE_p = T.printing.Print('T025D_dE')(T025D_dE)
		# rotate each to LOS
		# E:8, N:9, U:10
		T025D_dlos = T025D[:,8]*T025D_dE + T025D[:,9]*T025D_dN + T025D[:,10]*T025D_dZ
		T130A_dlos = T130A[:,8]*T130A_dE + T130A[:,9]*T130A_dN + T130A[:,10]*T130A_dZ
		T131A_dlos = T131A[:,8]*T131A_dE + T131A[:,9]*T131A_dN + T131A[:,10]*T131A_dZ
		#T025D_dlos_p = T.printing.Print('T025D_dlos')(T025D_dlos)
		# weighted squared differences
		# data:5, err:6
		T025D_l2 = (((T025D_dlos - T025D[:,5])/T025D[:,6])**2).sum()
		T130A_l2 = (((T130A_dlos - T130A[:,5])/T130A[:,6])**2).sum()
		T131A_l2 = (((T131A_dlos - T131A[:,5])/T131A[:,6])**2).sum()
		#T025D_l2_p = T.printing.Print('T025D_l2')(T025D_l2)
		return -0.5 * (T025D_l2 + T131A_l2) # - 0.5 * T131A_l2
		
	#X = pm.DensityDist('X',logp,observed={'x0':x0,'y0':y0,'z0':z0,'dV':dV})

	#llk = loglikelihood(x0_print,y0_print,z0_print,dV_print)
	#llk = loglikelihood(x0,y0,z0,dV)
	#llk_print = T.printing.Print('llk')(llk)
	#llkp = pm.Potential('llk_potential',llk_print)
	#llk_potential = pm.Potential('llk_potential',llk)
	#llk = pm.DensityDist('llk',loglikelihood,observed={'x0':x0,'y0':y0,'z0':z0,'dV':dV})

	def synthetic1(x0,y0,z0,dV):
		(T025D_dE, T025D_dN, T025D_dZ) = mogiDEM(x0,y0,z0,dV,T025D_E_utm,T025D_N_utm,dem_T025D)
		return T025D[:,8]*T025D_dE + T025D[:,9]*T025D_dN + T025D[:,10]*T025D_dZ
		
	def synthetic2(x0,y0,z0,dV):
		(T131A_dE, T131A_dN, T131A_dZ) = mogiDEM(x0,y0,z0,dV,T131A_E_utm,T131A_N_utm,dem_T131A)
		return T131A[:,8]*T131A_dE + T131A[:,9]*T131A_dN + T131A[:,10]*T131A_dZ

	def synthetic3(x0,y0,z0,dV):
		(T130A_dE, T130A_dN, T130A_dZ) = mogiDEM(x0,y0,z0,dV,T130A_E_utm,T130A_N_utm,dem_T130A)
		return T130A[:,8]*T130A_dE + T130A[:,9]*T130A_dN + T130A[:,10]*T130A_dZ

	data1 = T025D[:,5]
	cov1 = np.diag(T025D[:,6]**2)
	llk1 = pm.MvNormal('llk1',mu=synthetic1(x0,y0,z0,dV),cov=cov1,observed=data1)
	data2 = T131A[:,5]
	cov2 = np.diag(T131A[:,6]**2)
	llk2 = pm.MvNormal('llk2',mu=synthetic2(x0,y0,z0,dV),cov=cov2,observed=data2)
	data3 = T130A[:,5]
	cov3 = np.diag(T130A[:,6]**2)
	llk3 = pm.MvNormal('llk3',mu=synthetic3(x0,y0,z0,dV),cov=cov3,observed=data3)

	niter = 1000

	start = pm.find_MAP(model=basic_model)
	step = pm.NUTS(scaling=start)
	trace = pm.sample(niter, start = start, step = step)
	
	#n_chains = 100
	#n_steps = 50
	#tune_interval = 10
	#n_jobs = 1
	#trace = smc.sample_smc(
	#	n_steps=n_steps,
	#	n_chains=n_chains,
	#	tune_interval=tune_interval,
	#	n_jobs=n_jobs,
	#	#start=start,
	#	progressbar=False,
	#	stage=0,
	#	homepath=test_folder,
	#	model=basic_model
	#	)
	t = trace[niter//2:] # discard 50% of values
	#print t
	#x0_ = trace.get_values('x0',burn=niter//2, combine=True)
	#fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,sharex=True)
	fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
	ax1.plot(trace['x0'])
	ax2.plot(trace['y0'])
	ax3.plot(trace['z0'])
	ax4.plot(trace['dV'])
	#ax5.plot(trace['llk1'])
	#ax6.plot(trace['llk2'])
	#ax7.plot(trace['llk3'])
	plt.show()
	pm.traceplot(trace)
	plt.show()
	#pm.traceplot(t, varnames=['x0','y0','z0','dV','llk1','llk2','llk3'])
	pm.traceplot(t, varnames=['x0','y0','z0','dV'])
	plt.show()
	#pm.autocorrplot(t, varnames=['x0'])
	#plt.show()
	print (pm.summary(trace))
	print (pm.summary(t))

#shutil.rmtree(test_folder)
