import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
import math

def long2UTM(lon):
        return (math.floor((float(lon) + 180)/6) % 60) + 1

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
