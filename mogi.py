import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import pyproj
from gridutils import *
import numpy.fft as npfft
import theano
import theano.tensor as T
import theano.tensor.fft as Tfft
	

class Observed(object):
	def __init__(self,E_utm_,N_utm_,DEM_,uE_,uN_,uZ_,dLOS_,sigma_):
		self.E_utm = E_utm_
		self.N_utm = N_utm_
		self.DEM = DEM_
		self.uE = uE_
		self.uN = uN_
		self.uZ = uZ_
		self.dLOS = dLOS_
		self.sigma = sigma_

observeddata = []
utm_dem = 0
utm_E = 0
utm_N = 0
utm_EE = 0
utm_NN = 0

# granite
Vs = 2500 
nu = 0.25 
rho = 2500
Vp = Vs * math.sqrt((2.0*nu-2.0)/(2.0*nu-1.0))
mu = Vs**2 * rho 
lam = Vp**2 * rho - 2 * mu
P0 = mu / (1 - nu)

def add(E_utm,N_utm,DEM,uE,uN,uZ,dLOS,sigma):
	global observeddata
	observeddata.append(Observed(E_utm,N_utm,DEM,uE,uN,uZ,dLOS,sigma))

def setDEM(gammadem,par,gridsize,padding):
	"""
	Call this method AFTER all data has been added so that it can crop the DEM appropriately.
	gammadem is a 2D array
	par is the dem parameter file
	gridsize must be a power of 2
	padding in metres
	"""
	global utm_dem
	global utm_EE
	global utm_NN
	global utm_E
	global utm_N
	demlon = linspaceb(float(par['corner_lon']),float(par['post_lon']),int(par['width']))
	demlat = linspaceb(float(par['corner_lat']),float(par['post_lat']),int(par['nlines']))
	# get median boundary fill value for padding
	meanfill = 0.25 * (np.median(gammadem[:,0]) + np.median(gammadem[:,-1]) + np.median(gammadem[0,:]) + np.median(gammadem[-1,:]))
	# interpolate for easting and northing
	f = RegularGridInterpolator((demlon,np.flipud(demlat)),np.flipud(gammadem).T,method='linear',bounds_error=False,fill_value=None) #meanfill)
	#deltaE = float(par['post_lon'])*3600*30 * 4
	#deltaN = float(par['post_lat'])*3600*30 * 4
	min_Eo = min([o.E_utm.min() for o in observeddata]) #- abs(deltaE)
	max_Eo = max([o.E_utm.max() for o in observeddata]) #+ abs(deltaE)
	min_No = min([o.N_utm.min() for o in observeddata]) #- abs(deltaN)
	max_No = max([o.N_utm.max() for o in observeddata]) #+ abs(deltaN)
	# Instead of constraining to the minimum area, construct a 2^n x 2^n grid that covers the area
	# This is to make the FFT most efficient.
	# We already have desirable area. Choose the largest dimension, expand the other d to it and then grid by 512x512
	smallestd = min(max_Eo-min_Eo,max_No-min_No)
	addE = 0.5*(smallestd - (max_Eo-min_Eo))
	addN = 0.5*(smallestd - (max_No-min_No))
	#print "Add E,N = " + str((addE,addN))
	min_E = min_Eo - addE - padding
	max_E = max_Eo + addE + padding
	min_N = min_No - addN - padding
	max_N = max_No + addN + padding

	# Grid it
	#deltaEN = smallestd / gridsize 
	#print (min_E,max_E,max_E-min_E)
	#print (min_N,max_N,max_N-min_N)
	demE = np.linspace(min_E,max_E,gridsize)
	demN = np.linspace(min_N,max_N,gridsize)
	#demN = np.flipud(np.linspace(min_N,max_N,gridsize))
	demEE,demNN = np.meshgrid(demE,demN)
	wgs = pyproj.Proj(init="EPSG:4326")
	utm = pyproj.Proj("+proj=utm +zone="+str(long2UTM(float(par['corner_lon'])))+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
	demLon, demLat = pyproj.transform(utm,wgs,demEE,demNN)
	dem_utm_wgs_grid = np.column_stack((demLon.flatten(),demLat.flatten()))
	# interpolate
	utm_dem = f(dem_utm_wgs_grid)
	gridshape = (gridsize,gridsize) #(demN.shape[0],demE.shape[0])
	utm_dem = utm_dem.reshape(gridshape)
	utm_EE = demEE # TODO ensure this assign works
	utm_NN = demNN
	utm_E = demE
	utm_N = demN
	

def FourierGreensCorrections(s,t,mu,nu):
	s2 = s**2
	t2 = t**2
	s2t23r2 = (s2 + t2)**1.5
	G11= (s2 + t2 - nu * s2) / (2 * math.pi * mu * s2t23r2)
	G22= (s2 + t2 - nu * t2) / (2 * math.pi * mu * s2t23r2)
	G12= ( - nu * s * t) / (2 * math.pi * mu * s2t23r2)
	G13= ( - (1 - 2*nu) * s * 1.0j) / (2 * math.pi * mu * (s2 + t2))
	G23= ( - (1 - 2*nu) * t * 1.0j) / (2 * math.pi * mu * (s2 + t2))
	G11[np.isnan(G11)]=0
	G22[np.isnan(G22)]=0
	G12[np.isnan(G12)]=0
	G13[np.isnan(G13)]=0
	G23[np.isnan(G23)]=0
	G21= G12
	return [[G11, G12, G13],[G21, G22, G23]]

def stressCoeffs(lam,mu,nu,uxx):
	c = lam*(1-2*nu)/(1-nu)
	truxx = uxx[0][0]+uxx[1][1]
	s11 = c*truxx + 2*mu*uxx[0][0]
	s22 = c*truxx + 2*mu*uxx[1][1]
	#s111 = c*uxx[0][0] + 2*mu*uxx[0][0]
	#s221 = c*uxx[0][0] + 2*mu*uxx[1][1]
	#s112 = c*uxx[1][1] + 2*mu*uxx[0][0]
	#s222 = c*uxx[1][1] + 2*mu*uxx[1][1]
	s12 = mu*(uxx[0][1]+uxx[1][0])
	#import matplotlib.pyplot as plt
	#fig = plt.figure()
	#fa1 = fig.add_subplot(131)
	#cfa1 = fa1.imshow(s11)
	#fa2 = fig.add_subplot(132)
	#cfa2 = fa2.imshow(s12)
	#fa3 = fig.add_subplot(133)
	#cfa3 = fa3.imshow(s22)
	#plt.colorbar(cfa1, ax=fa1)
	#plt.colorbar(cfa2, ax=fa2)
	#plt.colorbar(cfa3, ax=fa3)
	#plt.show()
	#
	#return (s111,s112,s221,s222,s12)
	return (s11,s22,s12)


# Numpy version
def npfft2(a):
	#return npfft.fft2(a)
	#return npfft.fftshift(npfft.fft(a),axes=1)
	""" assumes len(a.shape)==2, does not assert this """
	A = np.zeros_like(a,dtype=np.complex_)
	S = a.shape[1]//2+1
	A[:,:S] = npfft.rfftn(a) 
	# copy left half to right half
	A[:,S:] = np.conj(npfft.fftshift(A))[:,S:]
	#A[:,S:] = np.conj(np.fliplr(A[:,1:S-1]))
	return A

# numpy version
def npifft2(A):
	#return npfft.ifft2(A)
	#return npfft.ifft(npfft.ifftshift(A,axes=1))
	s = A.shape[0]
	S = s//2 + 1
	return npfft.irfftn(A[:,:S]) #,A.shape)

# Theano version
def Tfft2(a):
	""" assumes len(a.shape)==2, does not assert this """
	#A = np.zeros_like(a,dtype=np.complex_)
	s = a.shape[0]
	S = s//2+1
	#aa = T.stack([a],axis=0)
	aa = a.reshape((1,a.shape[0],a.shape[1]))
	AA = Tfft.rfft(aa)
	B = AA[...,0] + 1.j * AA[...,1] # Theano rfft stores complex values in separate array 
	# get first output
	C = B[0,...]
	CC = C[:,1:S-1]
	A = T.zeros_like(a)
	Afront = A[:,:S]
	A = T.set_subtensor(Afront,C)
	Aback = A[:,S:]
	A = T.set_subtensor(Aback,T.conj(Tfftshift(A))[:,S:])
	return A

# Theano version
def Tfft2old(a):
	""" assumes len(a.shape)==2, does not assert this """
	#A = np.zeros_like(a,dtype=np.complex_)
	s = a.shape[1]
	S = s//2+1
	aa = T.stack([a],axis=0)
	AA = Tfft.rfft(aa)
	B = AA[...,0] + 1.j * AA[...,1] # Theano rfft stores complex values in separate array 
	#return AA[0,...,0]
	#A[:,:S] = B 
	# copy left half to right half
	#A[:,S:] = np.conj(np.fliplr(A[:,1:S-1]))
	#below no worky
	C = B[0,...]
	CC = C[:,1:S-1]
	return  T.concatenate([C,T.conj(CC[:,::-1])],axis=1)
	A = T.zeros_like(a)
	#Alookup = theano.shared(A)
	Afront = A[:,:S]
	Aback = A[:,S:]
	A = T.set_subtensor(Afront,B)
	A = T.set_subtensor(Aback,T.conj(Tfftshift(A))[:,S:])
	#A[:,:S] = B
	#A[:,S:] = T.conj(npfft.fftshift(A))[:,S:]
	return A

# Theano version
def Tifft2(A):
	s = A.shape[1]
	S = s//2 + 1
	#B = T.tensor3() # np.zeros((s,S,2),dtype=np.float_)
	#B[:,:,0] = T.real(A)
	#B[:,:,1] = T.imag(A)
	B = T.zeros((1,A.shape[1],S,2))
	Breal = B[0,:,:,0]
	B = T.set_subtensor(Breal,T.real(A[:,:S]))
	Bimag = B[0,:,:,1]
	B = T.set_subtensor(Bimag,T.imag(A[:,:S]))
	#B = T.stack([T.real(A[:,:S]),T.imag(A[:,:S])],axis=2)
	#BB = T.stack([B],axis=0)
	return Tfft.irfft(B)[0,:,:] #,A.shape)

def Tfftshift(x):
	xb = x
	for a in range(x.ndim):
		shift = x.shape[a] // 2
		xb = T.roll(xb,shift,a)	
	return xb

def Tifftshift(x):
	xb = x
	for a in range(x.ndim):
		shift = -x.shape[a] // 2
		xb = T.roll(xb,shift,a)	
	return xb

def mogiTopoCorrection(x0,y0,z0,radius,x,y,refdem,spacing,include_zeroth_order=False):
	"""
	Assumes x,y are equivalent to np.meshgrid(easting,northing) for all coordinates in the refdem.
	All expressions are taken from Williams and Wadge (2000)
	This method applies the first order topography correction using a uniform meshgrid x,y as input.
	"""
	(ux,uxx,uxxx) = mogiZerothOrder(x0,y0,z0,radius,mu,nu,P0,x,y)
	nx = refdem.shape[1]
	ny = refdem.shape[0]
	(s11,s22,s12) = stressCoeffs(lam,mu,nu,uxx)
	F11 = fft2(refdem*s11)
	F12 = fft2(refdem*s12)
	F22 = fft2(refdem*s22)
	s = npfft.fftfreq(nx,spacing)
	t = npfft.fftfreq(ny,spacing)
	#s = npfft.fftshift(npfft.fftfreq(nx,spacing))
	#t = np.linspace(0.0,spacing/ny,ny)
	#s = np.flipud(s)
	#print "s "
	#print s
	#print "t"
	#print t
	ss,tt = np.meshgrid(s,t)
	#import matplotlib.pyplot as plt
	#fig = plt.figure()
	#fa1 = fig.add_subplot(121)
	#cfa1 = fa1.imshow(ss)
	#fa2 = fig.add_subplot(122)
	#cfa2 = fa2.imshow(tt)
	#plt.colorbar(cfa1, ax=fa1)
	#plt.colorbar(cfa2, ax=fa2)
	#plt.show()
	
	#tt,ss = np.meshgrid(t,s)
	G = FourierGreensCorrections(ss,tt,mu,nu)
	#tt = np.flipud(tt)
	#ss = np.fliplr(ss)
	#testmult = F11 * G1
	#testmult2 = G1 * ss
	F11ss = F11 * ss
	F12ss = F12 * ss
	F12tt = F12 * tt
	F22tt = F22 * tt
	U1 = -2.0j * math.pi * (F11ss * G[0][0] + F12tt * G[0][0] + F12ss * G[1][0] + F22tt * G[1][0] )
	U2 = -2.0j * math.pi * (F11ss * G[0][1] + F12tt * G[0][1] + F12ss * G[1][1] + F22tt * G[1][1] )
	U3 = -2.0j * math.pi * (F11ss * G[0][2] + F12tt * G[0][2] + F12ss * G[1][2] + F22tt * G[1][2] )
	#U1 = F11 * tt
	#U2 = F12 * tt
	#U3 = F22 * tt
	#U1 = -2.0j * math.pi * (F11ss * G[0][0] )
	#U2 = -2.0j * math.pi * (F11ss * G[0][1] )
	#U3 = -2.0j * math.pi * (F11ss * G[0][2] )
	#U1 = -2.0j * math.pi * (F12tt * G[0][0] )
	#U2 = -2.0j * math.pi * (F12tt * G[0][1] )
	#U3 = -2.0j * math.pi * (F12tt * G[0][2] )
	#U1 = -2.0j * math.pi * (F12ss * G[1][0] )
	#U2 = -2.0j * math.pi * (F12ss * G[1][1] )
	#U3 = -2.0j * math.pi * (F12ss * G[1][2] )
	#U1 = -1.0j *  (F22tt * G[1][0] )
	#U2 = -1.0j * (F22tt * G[1][1] )
	#U3 = -1.0j * (F22tt * G[1][2] )
	#return (ux[0],ux[1],ux[2])
	if include_zeroth_order:
		return (ux[0] + ifft2(U1), ux[1] + ifft2(U2), ux[2] + ifft2(U3))
	else:
		return (ifft2(U1),ifft2(U2),ifft2(U3))
	#return (ifft2(fft2(s11)*ss),ifft2(fft2(s22)*tt),ifft2(fft2(s12)*ss))

# Pre-compute the Greens functions ffts for the DEM
# First compute zeroth order mogi
# Compute the fi from Appendix 1 using the zeroth order soln (as FFTs)
# Use A4 and A9 for displacement and displacement gradient ffts

def mogiZerothOrder(x0,y0,z0,radius,mu,nu,P0,x,y):
	"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,radius)
	"""
	#dx = (x - x0,y - y0, np.zeros(x.shape) + z0)
	dx = (x-x0,y - y0, z0)
	K = (P0 * (1 - nu) / mu) * (radius**3) # reduces to radius**3 for granite
	# or equivalently c= (3/4) a^3 dP / rigidity
	# where a = sphere radius, dP = delta Pressure
	R2 = (dx[0]**2 + dx[1]**2 + dx[2]**2)
	R3 = R2 ** 1.5
	R5 = R2 ** 2.5
	kr=np.identity(3)
	#ux=np.zeros(3)
	#uxx=np.zeros((3,3))
	#uxxx=np.zeros((3,3,3))
	ux=[0 for i in xrange(3)]
	uxx=[[0 for i in xrange(3)] for j in xrange(3)]
	uxxx=[[[0 for i in xrange(3)] for j in xrange(3)] for k in xrange(3)]
	for i in xrange(3):
		ux[i]=dx[i]*K/R3
	for j in xrange(2):
		for i in xrange(2):
			uxx[i][j]=(kr[i,j]-3.0*dx[i]*dx[j]/R2)*K/R3
		uxx[2][j]=(-3.0*K*dx[2]*dx[j])/R5
	#print uxxx[0][0][0]
	for j in xrange(2):
		for k in xrange(2):
			for i in xrange(2):
				uxxx[i][j][k]=((5.0*dx[i]*dx[j]*dx[k])/R2-kr[i,j]*dx[k]-kr[i,k]*dx[j]-kr[j,k]*dx[i])*3.0*K/R5
		uxxx[2][j][k]=((5.0*dx[j]*dx[k])/R2-kr[j,k])*3.0*K*dx[2]/R5
	#import matplotlib.pyplot as plt
	#fig = plt.figure()
	#fa1 = fig.add_subplot(131)
	#fa1.imshow(uxx[0][0])
	#fa2 = fig.add_subplot(132)
	#fa2.imshow(uxx[0][1])
	#fa3 = fig.add_subplot(133)
	#fa3.imshow(uxx[1][1])
	#plt.show()
	return (ux,uxx,uxxx)
	
	# u1=dx*K/R3
	# u2=dy*K/R3
	# u3=dz*K/R3
	# u11=(1.0-3.0*dx*dx/R2)*K/R3
	# u12=(0.0-3.0*dx*dy/R2)*K/R3
	# u22=(1.0-3.0*dy*dy/R2)*K/R3
	# u21=u12
	# u31=(-3.0*K*dz*dx)/R5
	# u32=(-3.0*K*dz*dy)/R5
	# u111=((5.0*dx*dx*dx)/R2-dx-dx-dx)*3.0*K/R5
	# u112=((5.0*dx*dx*dy)/R2-dy-dx-dx)*3.0*K/R5
	# u111=((5.0*dx*dx*dx)/R2-dx-dx-dx)*3.0*K/R5
	# u111=((5.0*dx*dx*dx)/R2-dx-dx-dx)*3.0*K/R5
	# u111=((5.0*dx*dx*dx)/R2-dx-dx-dx)*3.0*K/R5
	# u111=((5.0*dx*dx*dx)/R2-dx-dx-dx)*3.0*K/R5
	
def mogiVaryingDepth(x0,y0,z0,radius,x,y,dem):
	"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,radius)
	Assumes the dem argument has been subtracted from the reference level,
	i.e. negative values are above the reference level, positive values are below.
	"""
	#x0,y0,z0,radius = coeffs
	dx = x - x0
	dy = y - y0
	dz = (dem + z0).clip(0,1e5) # max 1e4m depth
	#dz = z0 # max 1e4m depth
	dV = 4.0/3.0 * math.pi * (radius ** 3)
	c = dV * 3. / (4. * math.pi)
	# or equivalently c= (3/4) a^3 dP / rigidity
	# where a = sphere radius, dP = delta Pressure
	r2 = dx*dx + dy*dy + dz*dz
	C = c / (r2 ** 1.5)
	return (C*dx,C*dy,C*dz)

def mogiDEMGBIS(x0,y0,z0,radius,x,y,dem):
	C=(radius ** 3) / 4.0
	nu=0.25 # approximate for granite
	dx=x-x0;
	dy=y-y0;
	dz=dem;
	d1=z0-dz;
	d2=z0+dz;
	R12=dx**2+dy**2+d1**2;
	R22=dx**2+dy**2+d2**2;
	R13=R12**1.5;
	R23=R22**1.5;
	R15=R12**2.5;
	R25=R22**2.5;
	
	#Calculate displacements
	
	ddx = C*( (3 - 4*nu)*dx/R13 + dx/R23 + 6*d1*dx*dz/R15 );
	ddy = C*( (3 - 4*nu)*dy/R13 + dy/R23 + 6*d1*dy*dz/R15 );
	ddz = C*( (3 - 4*nu)*d1/R13 + d2/R23 - 2*(3*d1**2 - R12)*dz/R15);
	
	return (ddx,ddy,ddz)

def obinterpolate(dEf,dNf,dZf,ob):
	dEi = dEf(np.column_stack((ob.E_utm,ob.N_utm)))
	dNi = dNf(np.column_stack((ob.E_utm,ob.N_utm)))
	dZi = dZf(np.column_stack((ob.E_utm,ob.N_utm)))
	return (dEi,dNi,dZi)

def logpob(dE,dN,dZ,ob):
	return (((((ob.uE*dE + ob.uN*dN + ob.uZ*dZ) - ob.dLOS) / ob.sigma)**2).sum())

#Numpy
def npgridinterp(coordgrid,A):
	return RegularGridInterpolator(coordgrid,A)

def Tgridinterp(coordgrid,A):
	""" Performs a vectorised bilinear interpolation
	    Assumes equi-linearly spaced coordgrid
	"""
	dx = coordgrid[0][1]-coordgrid[0][0]
	dy = coordgrid[1][1]-coordgrid[1][0]
	nx = coordgrid[0].shape[0]
	ny = coordgrid[1].shape[0]
	scale = np.array([1.0/dx,1.0/dy])
	topleft = np.array([coordgrid[0].min(),coordgrid[1].min()])
	print topleft
	def bilinear(inpcoordpairs):
		icp = (inpcoordpairs - topleft) * scale
		ax = icp[:,0] - np.floor(icp[:,0])
		ay = icp[:,1] - np.floor(icp[:,1])
		xl = np.maximum(0,np.minimum(nx-1,np.floor(icp[:,0]))).astype(np.int)
		xu = np.maximum(0,np.minimum(nx-1,np.ceil(icp[:,0]))).astype(np.int)
		yl = np.maximum(0,np.minimum(ny-1,np.floor(icp[:,1]))).astype(np.int)
		yu = np.maximum(0,np.minimum(ny-1,np.ceil(icp[:,1]))).astype(np.int)
		print len(xl)
		print xu 
		print yu 
		val = (1-ay) * ((1-ax) * A[xl,yl] + ax * A[xu,yl]) + ay * ((1-ax) * A[xl,yu] + ax * A[xu,yu])
		return val
	return bilinear


def logpgridinterpolate(x0,y0,z0,radius):
	global utm_E
	global utm_N
	global utm_EE
	global utm_NN
	ll = 0
	#dE,dN,dZ = gridsynthetic(x0,y0,z0,radius)
	dE,dN,dZ = mogiTopoCorrection(x0,y0,z0,radius,utm_EE,utm_NN,utm_dem,1.0,False) # second last arg is unity spacing.
	dEf = Tgridinterp((utm_E,utm_N),dE.T)
	dNf = Tgridinterp((utm_E,utm_N),dN.T)
	dZf = Tgridinterp((utm_E,utm_N),dZ.T)
	for o in observeddata:
		# interpolate first order correction
		dE1, dN1, dZ1 = obinterpolate(dEf,dNf,dZf,o)
		# Compute zeroth order correction more precisely using utm coordinates instead of interpolation
		(ux,uxx,uxxx) = mogiZerothOrder(x0,y0,z0,radius,mu,nu,P0,o.E_utm,o.N_utm)
		dEc = ux[0]+dE1
		dNc = ux[1]+dN1
		dZc = ux[2]+dZ1
		ll += logpob(dEc,dNc,dZc,o)
	return ll * (-0.5)

def synthetic(x0,y0,z0,radius,ob):
	(dE, dN, dZ) = mogiDEM(x0,y0,z0,radius,ob.E_utm,ob.N_utm,ob.DEM)
	return (((((ob.uE*dE + ob.uN*dN + ob.uZ*dZ) - ob.dLOS) / ob.sigma)**2).sum())
	
def logp(x0,y0,z0,radius):
	ll = 0
	for o in observeddata:
		ll += synthetic(x0,y0,z0,radius,o)
	return ll * (-0.5) 

	
def useTheano():
	global fft2
	global ifft2
	global gridinterp
	fft2 = Tfft2
	ifft2 = Tifft2
	gridinterp = Tgridinterp # npgridinterp

def useNumpy():
	global fft2
	global ifft2
	global gridinterp
	fft2 = npfft2
	ifft2 = npifft2
	gridinterp = Tgridinterp # npgridinterp

useNumpy()

def test_fft_numpy_equals_theano():
	#a = np.linspace(0.0,1.0,512)
	#xx,yy = np.meshgrid(a,a)
	#zz = xx + yy
	#import matplotlib.pyplot as plt
	#fig = plt.figure()
	#fa1 = fig.add_subplot(131)
	#cfa1 = fa1.imshow(xx)
	#fa2 = fig.add_subplot(132)
	#cfa2 = fa2.imshow(yy)
	#fa3 = fig.add_subplot(133)
	#cfa3 = fa3.imshow(zz)
	#plt.colorbar(cfa1, ax=fa1)
	#plt.colorbar(cfa2, ax=fa2)
	#plt.colorbar(cfa3, ax=fa3)
	#plt.show()
	zz = np.zeros((16,16))
	zz[0,0] = 10.0
	zz[8,8] = -10.0
	print "zz"
	print zz
	ZZ = npfft2(zz)
	zz1 = np.asarray([zz])
	print "zz1"
	print zz1
	tzz = theano.shared(zz)
	tZZ_ = Tfft2(tzz)
	tZZ = tZZ_.eval()
	import matplotlib.pyplot as plt
	fig = plt.figure()
	fa1 = fig.add_subplot(131)
	cfa1 = fa1.imshow(np.abs(ZZ))
	fa2 = fig.add_subplot(132)
	cfa2 = fa2.imshow(np.abs(tZZ))
	fa3 = fig.add_subplot(133)
	cfa3 = fa3.imshow(np.abs(tZZ-ZZ))
	plt.colorbar(cfa1, ax=fa1)
	plt.colorbar(cfa2, ax=fa2)
	plt.colorbar(cfa3, ax=fa3)
	plt.show()
	iZZ = npifft2(ZZ)
	itZZ = Tifft2(tZZ).eval()
	fig = plt.figure()
	fa1 = fig.add_subplot(131)
	cfa1 = fa1.imshow(np.abs(iZZ))
	fa2 = fig.add_subplot(132)
	cfa2 = fa2.imshow(np.abs(itZZ))
	fa3 = fig.add_subplot(133)
	cfa3 = fa3.imshow(np.abs(itZZ-iZZ))
	plt.colorbar(cfa1, ax=fa1)
	plt.colorbar(cfa2, ax=fa2)
	plt.colorbar(cfa3, ax=fa3)
	plt.show()

	# Test interpolation	
	a = np.linspace(0.0,1.0,512)
	dsto = 61
	b = np.linspace(0.0,1.0,dsto) # a prime number
	xx,yy = np.meshgrid(a,a)
	xxyy = xx*yy + (1-xx)*(1-yy)
	bx,by = np.meshgrid(b,b)
	ftgi = Tgridinterp((a,a),xxyy)
	fnpgi = npgridinterp((a,a),xxyy)
	bxby = np.column_stack((bx.flatten(),by.flatten()))
	tgi = ftgi(bxby).reshape((dsto,dsto))
	npgi = fnpgi(bxby).reshape((dsto,dsto))
	fig = plt.figure()
	fa1 = fig.add_subplot(141)
	cfa1 = fa1.imshow(tgi)
	fa2 = fig.add_subplot(142)
	cfa2 = fa2.imshow(npgi)
	fa3 = fig.add_subplot(143)
	cfa3 = fa3.imshow(tgi-npgi)
	fa4 = fig.add_subplot(144)
	cfa4 = fa4.imshow(xxyy)
	plt.colorbar(cfa1, ax=fa1)
	plt.colorbar(cfa2, ax=fa2)
	plt.colorbar(cfa3, ax=fa3)
	plt.colorbar(cfa4, ax=fa4)
	plt.show()
	
	
	
	
	
	

# Add some unit tests.
if __name__=="__main__":
	test_fft_numpy_equals_theano()
	

#class MogiModel(object):
#	def __init__(self):
#		self.observeddata = []
#
#	def add(self,E_utm,N_utm,DEM,uE,uN,uZ,dLOS,sigma):
#		self.observeddata.append(Observed(E_utm,N_utm,DEM,uE,uN,uZ,dLOS,sigma))
#	
#	def mogiDEM(self,x0,y0,z0,radius,x,y,dem):
#		"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,radius)"""
#		#x0,y0,z0,radius = coeffs
#		dx = x - x0
#		dy = y - y0
#		dz = (dem + z0).clip(0,1e5) # max 1e4m depth
#		#dz = z0 # max 1e4m depth
#		dV = 4.0/3.0 * math.pi * (radius ** 3)
#		c = dV * 3. / (4. * math.pi)
#		# or equivalently c= (3/4) a^3 dP / rigidity
#		# where a = sphere radius, dP = delta Pressure
#		r2 = dx*dx + dy*dy + dz*dz
#		C = c / (r2 ** 1.5)
#		return (C*dx,C*dy,C*dz)
#	
#	def synthetic(self,x0,y0,z0,radius,ob):
#		(dE, dN, dZ) = self.mogiDEM(x0,y0,z0,radius,ob.E_utm,ob.N_utm,ob.DEM)
#		return (((((ob.uE*dE + ob.uN*dN + ob.uZ*dZ) - ob.dLOS) / ob.sigma)**2).sum())
#		
#	def logp(self,x0,y0,z0,radius):
#		ll = 0
#		for o in self.observeddata:
#			ll += self.synthetic(x0,y0,z0,radius,o)
#		return ll * (-0.5) 
