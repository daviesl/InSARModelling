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
