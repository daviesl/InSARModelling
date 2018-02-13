import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import theano.tensor as T
import seaborn as sns
from scipy.interpolate import griddata
import scipy
import pandas as pd

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
