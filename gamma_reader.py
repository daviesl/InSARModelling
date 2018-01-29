import numpy as np
import math
import os
import sys
import logmgr
import loaddata 
import utils
from scipy import constants
import pyproj

logger = logmgr.logger('varres')

class gamma_reader:
    def __init__(self, demparname, iname, thetaname, phiname, corname):
        '''Class for reading in the geometry data and setting it up.'''
        
        if(os.path.isfile(iname) == False):
            logger.error('IFG file: %s not found.'%(iname))
            sys.exit(1)

        if(os.path.isfile(demparname) == False):
            logger.error('IFG DEM paramter file: %s not found.'%(demparname))
            sys.exit(1)

        if phiname is not None:
            if(os.path.isfile(phiname) == False):
                logger.error('Azimuth geometry file: %s not found.'%(phiname))
                sys.exit(1)

        if thetaname is not None:
            if(os.path.isfile(thetaname) == False):
                logger.error('Elevation geometry file: %s not found.'%(thetaname))
                sys.exit(1)

        if corname is not None:
            if(os.path.isfile(corname) == False):
                logger.error('Coherence file: %s not found.'%(corname))
                sys.exit(1)


        self.rdict = self.readpar(demparname)
        self.nx = np.int(self.rdict['width'])
        self.ny = np.int(self.rdict['nlines'])
        self.dlon = np.float(self.rdict['post_lon'])
        self.dlat = np.float(self.rdict['post_lat'])
        self.TL_lon = np.float(self.rdict['corner_lon'])
        #self.TL_east = np.float(self.rdict['corner_lon'])
        self.TL_lat = np.float(self.rdict['corner_lat'])
        #self.TL_north = np.float(self.rdict['corner_lat'])
	# TODO parse the SLC par file and extract the radar frequency property from it.
        #self.wvl = 100.0 * np.float(self.rdict['WAVELENGTH'])
	self.wvl = 100.0 * self.radar_freq_to_wavelength(1.2575002e+09) # appears to be in cm
       
        self.iname = iname
        self.thetaname = thetaname
	self.phiname = phiname
        self.corname = corname
        self.phs = None
        self.geom = []
        self.x = None
        self.y = None

    def radar_freq_to_wavelength(self,radar_frequency):
        return constants.speed_of_light / radar_frequency # wavelength in metres    

    def convert_radians_millimetres(self,data, radar_frequency):
        wavelength = constants.speed_of_light / radar_frequency # wavelength in metres    
        return data * ( wavelength * 1000 / ( 4 * constants.pi ))

    def readpar(self,file):
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

    def readgamma(self,datafile, par):
        """Function to read GAMMA format float data files"""
        ct = int(par['width']) * int(par['nlines'])
        print("Number of elements in the file is",ct)
    
        dt = np.dtype('>f4') # GAMMA files are big endian 32 bit float
    
        d = np.fromfile(datafile, dtype=dt, count=ct)
    
        d = d.reshape(int(par['nlines']), int(par['width']))
        print("Number of elements and size of the array is",d.size, d.shape)
	(goodi,goodj) = np.where(d!=0)
	ngood = len(goodi)
        d[d==0]= np.nan # convert zeros to nan
        return (d,ngood)

    def read_coherence(self,flip):
        (cor,ngood) = self.readgamma(self.corname,self.rdict)
        if flip:
            self.cor = np.flipud(cor)
        else:
            self.cor = cor

    def mask_igram_by_threshold(self,ct,ref=2):
        #self.phs[self.cor < ct] = np.nan
	mask = np.full_like(self.phs,1)
	mask[self.cor < ct] = 0
        import island
        c = island.Counter()
        numislands = c.numIslands(mask)
	print "Number of islands = " + str(numislands)
        maxisland, msize = c.maxIsland()
	print "Size of island " + str(maxisland) + " is " + str(msize)
        self.phs[c.grid!=maxisland] = np.nan

    def read_igram(self, scale=True, flip=False, mult=1.0):
        fact = np.choose(scale,[1.0,self.wvl/(4*np.pi)])
        fact = fact*mult

        #(phs,ngood) = loaddata.load_igram(self.iname,self.nx,self.ny,fact)
        (phs,ngood) = self.readgamma(self.iname,self.rdict)
	phs *= fact # convert radians to .... cm?
	# de-median
	mphs = np.ma.masked_where(np.isnan(phs),phs)
	print("Subtracting median ifg value = " + str(np.ma.median(mphs)))
	phs -= np.ma.median(mphs)
        if flip:
            self.phs = np.flipud(phs)
        else:
            self.phs = phs

        logger.info('Original number of data points: %d'%(ngood))

    def unit_vector(self,theta, phi):
        """Function to calculate unit vector for each pixel using GAMMA theta and phi files"""
        u = -np.sin(theta) # Negative in order to convert range shortening to positive in an 'up' frame
        e = -np.cos(phi)*np.cos(theta) # I think this is the correct sign
        n = -np.sin(phi)*np.cos(theta) # Not sure I have the correct sign here or not
        return e, n, u

    def long2UTM(self,lon):
        return (math.floor((float(lon) + 180)/6) % 60) + 1

    def read_geom(self, flip=False):
        """ assume input is lat/lon and desired is UTM """
	wgs = pyproj.Proj(init="EPSG:4326")
	utm = pyproj.Proj("+proj=utm +zone="+str(self.long2UTM(self.TL_lon))+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        #myProj = Proj("+proj=utm +zone="+self.long2UTM(self.TL_lon)+", +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        self.TL_east, self.TL_north = pyproj.transform(wgs,utm,float(self.TL_lon),float(self.TL_lat))
        self.x = self.clon = self.TL_lon + np.arange(self.nx) * self.dlon
        self.y = self.clat = self.TL_lat + np.arange(self.ny) * self.dlat
	#print self.clon.shape
	#print self.clat.shape
	#self.cEast, self.cNorth = pyproj.transform(wgs,utm,self.clon,self.clat)
        if flip:
            self.y = self.y[::-1]

        if self.thetaname is not None and self.phiname is not None:
            (self.theta,ngood) = self.readgamma(self.thetaname,self.rdict)
            (self.phi,ngood) = self.readgamma(self.phiname,self.rdict)
            self.geom = self.unit_vector(self.theta,self.phi)
            if flip:
                for k in xrange(3):
                    self.geom[k] = np.flipud(self.geom[k])

        else:
            for k in xrange(3):
                self.geom[k] = 0





############################################################
# Program is an extension to varres to allow GAMMA support #
# Copyright 2017 Geoscience Australia                      #
# Contact: earthdef@gps.caltech.edu                        #
############################################################
