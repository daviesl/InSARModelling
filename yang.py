import numpy as np
import math

def yangmodel(x0,y0,z0,a,A,P_G,mu,nu,theta,phi,x,y,z):
	# 3D Green's function for a spheroidal source 
	# all parameters are in SI (MKS) units
	#
	# OUTPUT
	# u         horizontal (East component) deformation
	# v         horizontal (North component) deformation
	# w         vertical (Up component) deformation
	# dwdx      ground tilt (East component)
	# dwdy      ground tilt (North component)
	# eea       areal strain
	# gamma1    shear strain
	# gamma2    shear strain
	#
	# SOURCE PARAMETERS
	# a         semimajor axis
	# A         geometric aspect ratio [dimensionless]
	# P_G       dimennsionless excess pressure (pressure/shear modulus) 
	# x0,y0     surface coordinates of the center of the prolate spheroid
	# z0        depth of the center of the sphere (positive downward and
	#              defined as distance below the reference surface)
	# theta     plunge (dip) angle [deg] [90 = vertical spheroid]
	# phi       trend (strike) angle [deg] [0 = aligned to North]
	#
	# CRUST PARAMETERS
	# mu        shear modulus
	# nu        Poisson's ratio 
	#
	# BENCHMARKS
	# x,y       benchmark location
	# z         depth within the crust (z=0 is the free surface)
	#
	# Reference ***************************************************************
	#
	# Note ********************************************************************
	# compute the displacement due to a pressurized ellipsoid 
	# using the finite prolate spheroid model by from Yang et al (JGR,1988)
	# and corrections to the model by Newmann et al (JVGR, 2006).
	# The equations by Yang et al (1988) and Newmann et al (2006) are valid for a
	# vertical prolate spheroid only. There is and additional typo at pg 4251 in 
	# Yang et al (1988), not reported in Newmann et al. (2006), that gives an error 
	# when the spheroid is tilted (plunge different from 90 degrees):
	#           C0 = y0*cos(theta) + z0*sin(theta)
	# The correct equation is 
	#           C0 = z0/sin(theta)
	# This error has been corrected in this script.
	# *************************************************************************
	#==========================================================================
	# USGS Software Disclaimer 
	# The software and related documentation were developed by the U.S. 
	# Geological Survey (USGS) for use by the USGS in fulfilling its mission. 
	# The software can be used, copied, modified, and distributed without any 
	# fee or cost. Use of appropriate credit is requested. 
	#
	# The USGS provides no warranty, expressed or implied, as to the correctness 
	# of the furnished software or the suitability for any purpose. The software 
	# has been tested, but as with any complex software, there could be undetected 
	# errors. Users who find errors are requested to report them to the USGS. 
	# The USGS has limited resources to assist non-USGS users; however, we make 
	# an attempt to fix reported problems and help whenever possible. 
	#==========================================================================
	
	
	# SINGULARITIES ***********************************************************
	if theta >= 89.99:
		theta = 89.99                                                          # solution is singular when theta = 90 degrees
	if A >= 0.99:
		A = 0.99
	# *************************************************************************
	
	
	# DISPLACEMENT ************************************************************
	# define parameters used to compute the displacement
	b = A*a                                                                    # semi-minor axis
	clambda = 2*mu*nu/(1-2*nu)                                                  # first Lame's elatic modulus
	P = P_G*mu                                                                 # excess pressure
	theta = math.pi*theta/180                                                       # dip angle in rad
	phi = math.pi*phi/180                                                           # strike angle in rad
	
	# compute 3D displacements
	(u, v, w) = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y,z)
	#return (u, v, w, dwdx, dwdy, eea, gamma1, gamma2)
	return (u, v, w)
	# *************************************************************************
	
	
	# # TILT ********************************************************************
	# h = 0.001*abs(max(x)-min(x));                                              # finite difference step
	# 
	# # East comonent
	# [~, ~, wp] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x+h,y,z);
	# [~, ~, wm] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x-h,y,z);
	# dwdx = 0.5*(wp - wm)/h;
	# 
	# # North component
	# [~, ~, wp] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y+h,z);
	# [~, ~, wm] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y-h,z);
	# dwdy = 0.5*(wp - wm)/h;
	# # *************************************************************************
	# 
	# 
	# # STRAIN ******************************************************************
	# # Displacement gradient tensor
	# [up , ~, ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x+h,y,z);
	# [um , ~, ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x-h,y,z);
	# dudx = 0.5*(up - um)/h;
	# 
	# [up , ~, ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y+h,z);
	# [um , ~, ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y-h,z);
	# dudy = 0.5*(up - um)/h;
	# 
	# [~, vp , ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x+h,y,z);
	# [~, vm , ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x-h,y,z);
	# dvdx = 0.5*(vp - vm)/h;
	# 
	# [~, vp , ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y+h,z);
	# [~, vm , ~] = yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y-h,z);
	# dvdy = 0.5*(vp - vm)/h;
	# 
	# # Strains
	# eea = dudx + dvdy;                                                          # areal strain
	# gamma1 = dudx - dvdy;                                                       # shear strain
	# gamma2 = dudy + dvdx;                                                       # shear strain
	# # *************************************************************************

def yangpar(a,b,clambda,mu,nu,P):
	# compute the parameters for the spheroid model
	# formulas from [1] Yang et al (JGR,1988)
	# corrections from [2] Newmann et al (JVGR, 2006), Appendix
	#
	# IN
	# a         semimajor axis [m]
	# b         semiminor axis [m]
	# clambda    Lame's constant [Pa]
	# mu        shear modulus [Pa]
	# nu        Poisson's ratio 
	# P         excess pressure (stress intensity on the surface) [pressure units]
	#
	# OUT
	# a1, b1    pressure (stress) [units of P] from [1]
	# c         prolate ellipsoid focus [m]
	# Pdila     pressure (proportional to double couple forces) [units of P] from [1]
	# Pstar     pressure [units of P]
	#
	# Notes:
	# [-]   : dimensionless
	#==========================================================================
	# USGS Software Disclaimer 
	# The software and related documentation were developed by the U.S. 
	# Geological Survey (USGS) for use by the USGS in fulfilling its mission. 
	# The software can be used, copied, modified, and distributed without any 
	# fee or cost. Use of appropriate credit is requested. 
	#
	# The USGS provides no warranty, expressed or implied, as to the correctness 
	# of the furnished software or the suitability for any purpose. The software 
	# has been tested, but as with any complex software, there could be undetected 
	# errors. Users who find errors are requested to report them to the USGS. 
	# The USGS has limited resources to assist non-USGS users; however, we make 
	# an attempt to fix reported problems and help whenever possible. 
	#==========================================================================
	
	epsn = 1E-10
	
	c = np.sqrt(a**2-b**2)                                  # prolate ellipsoid focus [m]
	
	a2 = a**2
	a3 = a**3 
	b2 = b**2
	c2 = c**2 
	c3 = c**3 
	c4 = c**4 
	c5 = c**5
	ac = (a-c)/(a+c)                                   # [-]
	coef1 = 2*math.pi*a*b2                                  # [m^3]
	den1 = 8*math.pi*(1-nu)                                 # [-]
	
	Q = 3/den1                                         # [-]       - parameter from [1]
	R = (1-2*nu)/den1                                  # [-]       - parameter from [1]
	Ia = -coef1*(2/(a*c2) + np.log(ac)/c3)                # [-]       - parameter from [1]
	Iaa = -coef1*(2/(3*a3*c2) + 2/(a*c4) + np.log(ac)/c5) # [1/m^2]  - parameter from [1]
	
	a11 = 2*R*(Ia-4*math.pi)                                # [-]        - (A-1) from [2]
	a12 = -2*R*(Ia+4*math.pi)                               # [-]        - (A-2) from [2]
	a21 = Q*a2*Iaa + R*Ia - 1                          # [-]        - (A-3) from [2]
	a22 = -Q*a2*Iaa - Ia*(2*R-Q)                       # [-]        - (A-4) from [2]
	
	den2 = 3*clambda+2*mu                               # [Pa]
	num2 = 3*a22-a12                                   # [-]
	den3 = a11*a22-a12*a21                             # [-]
	num3 = a11-3*a21                                   # [-]
	
	Pdila = P*(2*mu/den2)*(num2-num3)/den3                     # [units of P]  - (A-5) from [2]
	Pstar = P*(1/den2)*(num2*clambda+2*(clambda+mu)*num3)/den3   # [units of P]  - (A-6) from [2]
	
	a1 = - 2*b2*Pdila                                  # [m^2*Pa]  - force from [1]
	b1 = 3*(b2/c2)*Pdila + 2*(1-2*nu)*Pstar            # [Pa]      - pressure from [1]
	return (a1,b1,c,Pdila,Pstar)

def yangint(x,y,z,z0,theta,a1,b1,a,b,csi,mu,nu,Pdila):
	# compute the primitive of the displacement for a prolate ellipsoid
	# equation (1)-(8) from Yang et al (JGR, 1988) 
	# corrections to some parameters from Newmann et al (JVGR, 2006)
	# 
	# IN
	# x,y,x     coordinates of the point(s) where the displacement is computed [m]
	# y0,z0     coordinates of the center of the prolate spheroid (positive downward) [m]
	# theta     plunge angle [rad]
	# a1,b1     pressure (stress) (output from yangpar.m) [units of P]
	# a         semimajor axis [m]
	# b         semiminor axis [m]
	# c         focus of the prolate spheroid (output from yangpar.m) [m]
	# mu        shear modulus [Pa]
	# nu        Poisson's ratio
	# Pdila     pressure (proportional to double couple forces) [units of P]
	# 
	#
	# OUT
	# U1,U2,U3 : displacement in local coordinates [m] - see Figure 3 of Yang et al (1988)
	#
	# Notes:
	# The location of the center of the prolate spheroid is (x0,y0,z0)
	#     with x0=0 and y0=0;
	# The free surface is z=0;
	#==========================================================================
	# USGS Software Disclaimer 
	# The software and related documentation were developed by the U.S. 
	# Geological Survey (USGS) for use by the USGS in fulfilling its mission. 
	# The software can be used, copied, modified, and distributed without any 
	# fee or cost. Use of appropriate credit is requested. 
	#
	# The USGS provides no warranty, expressed or implied, as to the correctness 
	# of the furnished software or the suitability for any purpose. The software 
	# has been tested, but as with any complex software, there could be undetected 
	# errors. Users who find errors are requested to report them to the USGS. 
	# The USGS has limited resources to assist non-USGS users; however, we make 
	# an attempt to fix reported problems and help whenever possible. 
	#==========================================================================
	
	
	# precalculate parameters that are used often
	sint = np.sin(theta)
	cost = np.cos(theta)                                       # y0 = 0;
	
	# new coordinates and parameters from Yang et al (JGR, 1988), p. 4251
	# dimensions [m]
	csi2 = csi*cost 
	csi3 = csi*sint                                           # see Figure 3 of Yang et al (1988)
	x1 = x  
	x2 = y 
	x3 = z - z0 
	xbar3 = z + z0                               # x2 = y - y0;
	y1 = x1 
	y2 = x2 - csi2 
	y3 = x3 - csi3 
	ybar3 = xbar3 + csi3
	r2 = x2*sint - x3*cost 
	q2 = x2*sint + xbar3*cost
	r3 = x2*cost + x3*sint 
	q3 = -x2*cost + xbar3*sint
	rbar3 = r3 - csi 
	qbar3 = q3 + csi
	R1 = np.sqrt(y1**2+y2**2+y3**2) 
	R2 = np.sqrt(y1**2+y2**2+ybar3**2)
	#########################################################################
	#C0 = y0*cost + z0*sint
	C0= z0/sint                                                                # correction base on test by FEM by P. Tizzani IREA-CNR Napoli
	#########################################################################
	beta = (q2*cost + (1+sint)*(R2+qbar3))/(cost*y1 + 1E-15)                  # add 1E-15 to avoid a Divide by Zero warning at the origin
	
	# precalculate parameters that are used often
	drbar3 = R1+rbar3 
	dqbar3 = R2+qbar3 
	dybar3 = R2+ybar3
	lrbar3 = np.log(R1+rbar3) 
	lqbar3 = np.log(R2+qbar3) 
	lybar3 = np.log(R2+ybar3)
	atanb = np.arctan(beta)
	
	# primitive parameters from Yang et al (1988), p. 4252
	Astar1 = a1/(R1*drbar3) + b1*(lrbar3+(r3+csi)/drbar3)
	Astarbar1 = -a1/(R2*dqbar3) - b1*(lqbar3+(q3-csi)/dqbar3)
	
	A1 = csi/R1 + lrbar3 
	Abar1 = csi/R2 - lqbar3
	A2 = R1 - r3*lrbar3 
	Abar2 = R2 - q3*lqbar3
	A3 = csi*rbar3/R1 + R1 
	Abar3 = csi*qbar3/R2 - R2
	
	Bstar = (a1/R1+2*b1*A2) + (3-4*nu)*(a1/R2+2*b1*Abar2)
	B = csi*(csi+C0)/R2 - Abar2 - C0*lqbar3
	
	# the 4 equations below have been changed to improve the fit to internal deformation
	Fstar1 = 0 
	Fstar2 = 0 
	F1 = 0 
	F2 = 0 
	
	f1 = csi*y1/dybar3 + (3/cost**2)*(y1*sint*lybar3 - y1*lqbar3 + 2*q2*atanb) + 2*y1*lqbar3 - 4*xbar3*atanb/cost
	f2 = csi*y2/dybar3 + (3/cost**2)*(q2*sint*lqbar3 - q2*lybar3 + 2*y1*sint*atanb + cost*(R2-ybar3)) - 2*cost*Abar2 + (2/cost)*(xbar3*lybar3 - q3*lqbar3)                      # correction after Newmann et al (2006), eq (A-9)
	f3 = (1/cost)*(q2*lqbar3 - q2*sint*lybar3 + 2*y1*atanb) + 2*sint*Abar2 + q3*lybar3 - csi
	
	
	# precalculate coefficients that are used often
	cstar = (a*b**2/csi**3)/(16*mu*(1-nu)) 
	cdila = 2*cstar*Pdila
	
	# displacement components (2) to (7): primitive of equation (1) from Yang et al (1988)
	Ustar1 = cstar*(Astar1*y1 + (3-4*nu)*Astarbar1*y1 + Fstar1*y1)          # equation (2) from Yang et al (1988)
	# U2star and U3star changed to improve fit to internal deformation
	Ustar2 = cstar*(sint*(Astar1*r2 + (3-4*nu)*Astarbar1*q2 + Fstar1*q2) + cost*(Bstar-Fstar2))                                              # equation (3) from Yang et al (1988)
	
	# The formula used in the script by Fialko and Andy is different from
	# equation (4) of Yang et al (1988)
	# I use the same to continue to compare the results 2009 07 23
	# Ustar3 = cstar*(-cost*(Astarbar1*r2 + (3-4*nu)*Astarbar1*q2 - Fstar1*q2) + ...
	#         sint*(Bstar+Fstar2) + 2*cost^2*z*Astarbar1)           
	###################################################################################
	# The equation below is correct - follows equation (4) from Yang et al (1988)
	Ustar3 = cstar*(-cost*(Astar1*r2 + (3-4*nu)*Astarbar1*q2 - Fstar1*q2) + sint*(Bstar+Fstar2))                                              # equation (4) from Yang et al (1988)
	####################################################################################
	
	Udila1 = cdila*((A1*y1 + (3-4*nu)*Abar1*y1 + F1*y1) - 4*(1-nu)*(1-2*nu)*f1)             # equation (5) from Yang et al (1988)
	Udila2 = cdila*(sint*(A1*r2 + (3-4*nu)*Abar1*q2 + F1*q2) - 4*(1-nu)*(1-2*nu)*f2 + 4*(1-nu)*cost*(A2+Abar2) + cost*(A3-(3-4*nu)*Abar3 - F2))                         # equation (6) from Yang et al (1988)
	Udila3 = cdila*(cost*(-A1*r2 + (3-4*nu)*Abar1*q2 + F1*q2) + 4*(1-nu)*(1-2*nu)*f3 + 4*(1-nu)*sint*(A2+Abar2) + sint*(A3+(3-4*nu)*Abar3 + F2 - 2*(3-4*nu)*B))          # equation (7) from Yang et al (1988)
	
	    
	# displacement: equation (8) from Yang et al (1988) - see Figure 3
	U1 = Ustar1 + Udila1                                                                       # local x component
	U2 = Ustar2 + Udila2                                                                       # local y component
	U3 = Ustar3 + Udila3                                                                       # local z component
	return (U1,U2,U3)

def yangdisp(x0,y0,z0,a,b,clambda,mu,nu,P,theta,phi,x,y,z):
	# compute the 3D displacement due to a pressurized ellipsoid  
	#
	# IN
	# a         semimajor axis [m]
	# b         semiminor axis [m]
	# clambda   Lame's constant [Pa]
	# mu        shear modulus [Pa]
	# nu        Poisson's ratio 
	# P         excess pressure (stress intensity on the surface) [pressure units]
	# x,y,x     coordinates of the point(s) where the displacement is computed [m]
	# x0,y0,z0  coordinates of the center of the prolate spheroid (positive downward) [m]
	# theta     plunge angle [rad]
	# phi       trend angle [rad]
	#
	# OUT
	# Ux,Uy,Uz  displacement 
	#
	# Note ********************************************************************
	# compute the displacement due to a pressurized ellipsoid 
	# using the finite prolate spheroid model by from Yang et al (JGR,1988)
	# and corrections to the model by Newmann et al (JVGR, 2006).
	# The equations by Yang et al (1988) and Newmann et al (2006) are valid for a
	# vertical prolate spheroid only. There is and additional typo at pg 4251 in 
	# Yang et al (1988), not reported in Newmann et al. (2006), that gives an error 
	# when the spheroid is tilted (plunge different from 90 degrees):
	#           C0 = y0*cos(theta) + z0*sin(theta)
	# The correct equation is 
	#           C0 = z0/sin(theta)
	# This error has been corrected in this script.
	# *************************************************************************
	#==========================================================================
	# USGS Software Disclaimer 
	# The software and related documentation were developed by the U.S. 
	# Geological Survey (USGS) for use by the USGS in fulfilling its mission. 
	# The software can be used, copied, modified, and distributed without any 
	# fee or cost. Use of appropriate credit is requested. 
	#
	# The USGS provides no warranty, expressed or implied, as to the correctness 
	# of the furnished software or the suitability for any purpose. The software 
	# has been tested, but as with any complex software, there could be undetected 
	# errors. Users who find errors are requested to report them to the USGS. 
	# The USGS has limited resources to assist non-USGS users; however, we make 
	# an attempt to fix reported problems and help whenever possible. 
	#==========================================================================
	
	
	# testing parameters ######################################
	# clear all; close all; clc
	# a = 1000; b = 0.99*a
	# clambda = 1; mu = clambda; nu = 0.25; P = 0.01
	# theta = pi*89.99/180; phi = 0
	# x = linspace(0,2E4,7)
	# y = linspace(0,1E4,7)
	# x0 = 0; y0 = 0; z0 = 5E3
	############################################################
	
	# compute the parameters for the spheroid model
	a1,b1,c,Pdila,Pstar = yangpar(a,b,clambda,mu,nu,P)
	
	# translate the coordinates of the points where the displacement is computed
	# in the coordinates systen centered in (x0,0)
	xxn = x - x0
	yyn = y - y0
	
	# rotate the coordinate system to be coherent with the model coordinate
	# system of Figure 3 (Yang et al., 1988)
	xxp  = np.cos(phi)*xxn - np.sin(phi)*yyn
	yyp  = np.sin(phi)*xxn + np.cos(phi)*yyn
	
	# compute displacement for a prolate ellipsoid at csi = c
	U1p,U2p,U3p = yangint(xxp,yyp,z,z0,theta,a1,b1,a,b,c,mu,nu,Pdila)
	# compute displacement for a prolate ellipsoid at csi = -c
	U1m,U2m,U3m = yangint(xxp,yyp,z,z0,theta,a1,b1,a,b,-c,mu,nu,Pdila)
	Upx = -U1p-U1m
	Upy = -U2p-U2m
	Upz  =  U3p+U3m
	# rotate horizontal displacement back (strike)
	Ux = np.cos(phi)*Upx + np.sin(phi)*Upy
	Uy = -np.sin(phi)*Upx + np.cos(phi)*Upy
	Uz = Upz
	return (Ux, Uy, Uz)
