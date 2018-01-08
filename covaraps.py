#####################################################################
#  Approximating Atmospheric covariance function                    #
#  with an exponential.						    #
#  Translated from variable_res.m                                   #
#  Written by Piyush Agram.					    #
#  Date: Jan 2, 2012                                                #
#####################################################################

import numpy as np
import utils
import matplotlib.pyplot as plt
import scipy.optimize as sp
import logmgr
import math

logger=logmgr.logger('varres')

#Structure function
def model_fn(t,sig,lam):
	return sig*(1-np.exp(-t/lam))


# TODO write a new rampfn that does not assume cartesian distances.
def ramp_fn(t,a,b):
	dx = t[:,0]
	dy = t[:,1]
	d = np.sqrt(dx*dx+dy*dy) # doesn't even seem to be used
	return a*dx*dx + b*dy*dy + 2*a*b*dx*dy


#Estimation of sigma and lambda
# covariance of atmosphere = sigma^2 * exp(-dist/lambda)
#### Could definitely implement this better.
#### Plotting and histograms can be improved.
def aps_param(phs,frac,scale,plotflag):
	[ii,jj] = np.where(np.isnan(phs) == False)
	val = phs[ii,jj]
	num = len(ii)
	Nsamp = np.floor(frac*num)
	samp = np.random.random_integers(0,num-1,size=(Nsamp,2))
	s1 = samp[:,0]
	s2 = samp[:,1]
	dx = scale*(ii[s1]-ii[s2])
	dy = scale*(jj[s1]-jj[s2])
	dist = np.sqrt(dx*dx+dy*dy)
	ind = dist.nonzero()
	dist = dist[ind]
	dx = dx[ind]
	dy = dy[ind]
	Nsamp = len(dist)
	dv = val[s1]-val[s2]
	dv = dv*dv
	dv = dv[ind]

	mask = (dist < 1)
	dist = dist[mask]
	dx = dx[mask]
	dy = dy[mask]
	dv = dv[mask]
	Nsamp = len(dist)

	Amat = np.zeros((Nsamp,2))
	Amat[:,0] = dx
	Amat[:,1] = dy
	print(ramp_fn)
	print(Amat)
	print(dv)
	opt_pars, pars_cov = sp.curve_fit(ramp_fn,Amat,dv)
	a = opt_pars[0]
	b = opt_pars[1]
	logger.info('RAMP: %f %f'%(a,b))
	dv = dv - ramp_fn(Amat,a,b)

	opt_pars, pars_cov = sp.curve_fit(model_fn,dist,dv)
	sig = opt_pars[0]
	lam = opt_pars[1]

        logger.info('SIGMA   : %f'%(sig))
        logger.info('LAMBDA  : %f'%(lam))

	if plotflag:
	    plt.figure('Structure')
	    plt.hold(True)
	    plt.scatter(dist,dv,s=1,c='k')
	    x = np.arange(100)*0.1*dist.max()/100.0
	    y = model_fn(x,sig,lam)
	    plt.plot(x,y)
	    plt.xlabel('Normlized Distance')
	    plt.ylabel('log(phase var)')
	    plt.show()
		
	return sig,lam

def long2UTM(lon):
        return (math.floor((float(lon) + 180)/6) % 60) + 1

import pyproj
#from geographiclib.geodesic import Geodesic
#Estimation of sigma and lambda
# covariance of atmosphere = sigma^2 * exp(-dist/lambda)
#### UPDATED 2017/12/08 Laurence Davies
####         Distances computed using pyproj
#### Could definitely implement this better.
#### Plotting and histograms can be improved.
#### TODO input meshgrid of coordinate pairs for sampling
#### Scale parameter used such that all distances < 1 are included in covar calc
####       e.g. to use all <1km distance pairs, use scale = 0.001
def aps_param_proj(phs,frac,xg,yg,scale,plotflag):
	[ii,jj] = np.where(np.isnan(phs) == False)
	val = phs[ii,jj]
	num = len(ii)
	Nsamp = np.floor(frac*num)
	samp = np.random.random_integers(0,num-1,size=(Nsamp,2))
	s1 = samp[:,0]
	s2 = samp[:,1]
	# remove all pairs where s1==s2
	delidx = np.where(np.equal(s1,s2))
	s1 = np.delete(s1,delidx)
	s2 = np.delete(s2,delidx)
	
        geod = pyproj.Geod(ellps='WGS84')
	xx, yy = np.meshgrid(xg,yg)
	xx = xx.flatten()
	yy = yy.flatten()
	#print xx
	#print yy
        _az12, _az21, dist = geod.inv(xx[s1],yy[s1],xx[s2],yy[s2]) #,radians=True)
	#Geo = Geodesic.WGS84
	#dist = Geo.Inverse(yy[s1],xx[s1],yy[s2],xx[s2])
	wgs = pyproj.Proj(init="EPSG:4326")
	utm = pyproj.Proj("+proj=utm +zone="+str(long2UTM(xx[0]))+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
	xxutm, yyutm = pyproj.transform(wgs,utm,xx,yy)
	# NOTE: dx, dy are approximate because it is not a cartesian grid.
        dx = scale * (xxutm[s1]-xxutm[s2])
        dy = scale * (yyutm[s1]-yyutm[s2])
	dist *= scale
  	#dx = scale*(ii[s1]-ii[s2])
	#dy = scale*(jj[s1]-jj[s2])
	#dist = np.sqrt(dx*dx+dy*dy)
	ind = dist.nonzero()
	dist = dist[ind]
	dx = dx[ind]
	dy = dy[ind]
	Nsamp = len(dist)
	dv = val[s1]-val[s2]
	dv = dv*dv
	dv = dv[ind]

	mask = (dist < 1)
	dist = dist[mask]
	dx = dx[mask]
	dy = dy[mask]
	dv = dv[mask]
	Nsamp = len(dist)

	# What is this doing? Why are dx and dy the columns of amat? What are we solving for?
	Amat = np.zeros((Nsamp,2))
	Amat[:,0] = dx
	Amat[:,1] = dy
	print(ramp_fn)
	print(Amat)
	print(dv)
	opt_pars, pars_cov = sp.curve_fit(ramp_fn,Amat,dv)
	a = opt_pars[0]
	b = opt_pars[1]
	logger.info('RAMP: %f %f'%(a,b))
	dv = dv - ramp_fn(Amat,a,b)

	opt_pars, pars_cov = sp.curve_fit(model_fn,dist,dv)
	sig = opt_pars[0]
	lam = opt_pars[1]

        logger.info('SIGMA   : %f'%(sig))
        logger.info('LAMBDA  : %f'%(lam))

	if plotflag:
	    plt.figure('Structure')
	    plt.hold(True)
	    plt.scatter(dist,dv,s=1,c='k')
	    x = np.arange(100)*0.1*dist.max()/100.0
	    y = model_fn(x,sig,lam)
	    plt.plot(x,y)
	    plt.xlabel('Normlized Distance')
	    plt.ylabel('log(phase var)')
	    plt.show()
		
	return sig,lam
			

############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
