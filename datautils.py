import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
import math
import pyproj

class varres_data(object):
	def __init__(self,los,cov):
		self.los_name = los
		self.cov_name = cov
	def diagCov(self,fn):
		c = np.fromfile(fn, dtype=np.float32)
		dim = int(math.sqrt(c.shape[0]))
		c = c.reshape((dim,dim))
		return np.diag(c)
	def load(self):
		# Varres LOS Data Headers:
		# Number xind yind east north data err wgt Elos Nlos Ulos
		# east/north are actually lon and lat
		self.los_data = np.loadtxt(self.los_name,skiprows=2)
		# Varres covariance data is a binary file describing a square matrix
		self.diagcov = self.diagCov(self.cov_name)
		# convert cm to m for los values and set minimum error to 0.01
		self.los_data[:,5] *= 0.01
		# combine error in los data
		self.los_data[:,6] = np.sqrt(self.los_data[:,6]**2 + self.diagcov)
		# convert cm to m for error in los data
		self.los_data[:,6] *= 0.01
	def lat(self, row=False):
		return self.val(4,row)
	def lon(self, row=False):
		return self.val(3,row)
	def losd(self,row=False):
		""" Line of sight displacement """
		return self.val(5,row)
	def downsample_stddev(self,row=False):
		return self.val(6,row)
	def unit_vector(self, row=False):
		""" returns (E, N, U) unit vector from ground to platform of LOS"""
		return (self.val(8,row),self.val(9,row),self,val(10,row))
	def val(self, col, row):
		if row==False:
			return self.los_data[:,col]
		else:
			return self.los_data[row,col]
	def interpolateDEM(self,f):
		self.dem = f(self.los_data[:,3:5]).flatten()
	def maxDEM(self):
		return self.dem.max()
	def medianDEM(self):
		return np.median(self.dem)
	def set_dem_reflevel(self,dem_reflevel):
		self.dem = dem_reflevel - self.dem
	def reproject(self,wgs,utm):
		self.E_utm, self.N_utm = pyproj.transform(wgs,utm,self.lon(),self.lat())
