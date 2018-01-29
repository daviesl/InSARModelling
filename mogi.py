import numpy as np
import math

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

def add(E_utm,N_utm,DEM,uE,uN,uZ,dLOS,sigma):
	observeddata.append(Observed(E_utm,N_utm,DEM,uE,uN,uZ,dLOS,sigma))


def FourierGreensCorrections(s,t,mu,nu):
	s2 = s**2
	t2 = t**2
	G1= (s2 + t2 - nu * s2) / (2 * math.pi * mu * (s2 + t2)**1.5)
	G2= (s2 + t2 - nu * t2) / (2 * math.pi * mu * (s2 + t2)**1.5)
	G3= ( - nu * s * t) / (2 * math.pi * mu * (s2 + t2)**1.5)
	G4= ( - (1 - 2*nu) * s * 1j) / (4 * math.pi * mu * (s2 + t2))
	G5= ( - (1 - 2*nu) * t * 1j) / (4 * math.pi * mu * (s2 + t2))

def stressCoeffs(lam,mu,nu,uxx):
	c = lam*(1-2*nu)/(1-nu)
	truxx = uxx[0,0]+uxx[1,1]
	s11 = c*truxx + 2*mu*uxx[0,0]
	s22 = c*truxx + 2*mu*uxx[1,1]
	s12 = mu*(uxx[0,1]+uxx[1,0]
	return (s11,s22,s12)

def mogiTopoCorrected(x0,y0,z0,radius,x,y,refdem):
	"""
	Assumes x,y are equivalent to np.meshgrid(easting,northing) for all coordinates in the refdem.
	All expressions are taken from Williams and Wadge (2000)
	"""
	# granite
	Vs = 2500 
	nu = 0.25 
	rho = 2500
	Vp = Vs * math.sqrt((2.0*nu-2.0)/(2.0*nu-1.0))
	mu = Vs**2 * rho 
	lam = Vp**2 * rho - 2 * mu
	P0 = mu / (1 - nu)
	(ux,uxx,uxxx) = mogiZerothOrder(x0,y0,z0,radius,mu,nu,P0,x,y)
	# FIXME define range of s and t
	s = np.fft.fftfreq(n,spacing)
	t = np.fft.fftfreq(n,spacing)
	ss,tt = np.meshgrid((s,t))
	(G1, G2, G3, G4, G5) = FourierGreensCorrections(ss,tt,mu,nu)
	(s11,s22,s12) = stressCoeffs(lam,mu,nu,uxx)
	U1 = np.fft.fft2(refdem*(s11 + s12)) * G1 + np.fft.fft2(refdem*(s12 + s22)) * G3
	U2 = np.fft.fft2(refdem*(s11 + s12)) * G2 + np.fft.fft2(refdem*(s12 + s22)) * G3
	U3 = np.fft.fft2(refdem*(s11 + s12)) * G4 + np.fft.fft2(refdem*(s12 + s22)) * G5
	return (np.fft.ifft2(U1), np.fft.ifft2(U2), np.fft.ifft2(U3))
	
	
	

# Pre-compute the Greens functions ffts for the DEM
# First compute zeroth order mogi
# Compute the fi from Appendix 1 using the zeroth order soln (as FFTs)
# Use A4 and A9 for displacement and displacement gradient ffts

def mogiZerothOrder(x0,y0,z0,radius,mu,nu,P0,x,y):
	"""evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts, where coeffs = (x0,y0,z0,radius)
	"""
	dx = (x - x0,y - y0, np.zeros(x.shape[0]) + z0)
	K = (P0 * (1 - nu) / mu) * (radius**3) # reduces to radius**3 for granite
	# or equivalently c= (3/4) a^3 dP / rigidity
	# where a = sphere radius, dP = delta Pressure
	R2 = (dx[0]**2 + dy[1]**2 + dz[2]**2)
	R3 = R2 ** 1.5
	R5 = R2 ** 2.5
	kr=np.identity(3)
	#ux=np.zeros(3)
	#uxx=np.zeros((3,3))
	#uxxx=np.zeros((3,3,3))
	ux=[np.array() for i in xrange(3)]
	uxx=[ux[:] for i in xrange(3)]
	uxxx=[uxx[:] for i in xrange(3)]
	for i in xrange(3):
		ux[i]=dx[i]*K/R3
	for j in xrange(2):
		for i in xrange(2):
			uxx[i,j]=(kr[i,j]-3.0*dx[i]*dx[j]/R2)*K/R3
		uxx[2,j]=(-3.0*K*dx[2]*dx[j])/R5
	for j in xrange(2):
		for k in xrange(2):
			for i in xrange(2):
				uxxx[i,j,k]=((5.0*dx[i]*dx[j]*dx[k])/R2-kr[i,j]*dx[k]-kr[i,k]*dx[j]-kr[j,k]*dx[i])*3.0*K/R5
		uxxx[2,j,k]=((5.0**dx[j]*dx[k])/R2-kr[j,k])*3.0*K*dx[2]/R5
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

def synthetic(x0,y0,z0,radius,ob):
	(dE, dN, dZ) = mogiDEM(x0,y0,z0,radius,ob.E_utm,ob.N_utm,ob.DEM)
	return (((((ob.uE*dE + ob.uN*dN + ob.uZ*dZ) - ob.dLOS) / ob.sigma)**2).sum())
	
def logp(x0,y0,z0,radius):
	ll = 0
	for o in observeddata:
		ll += synthetic(x0,y0,z0,radius,o)
	return ll * (-0.5) 

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
