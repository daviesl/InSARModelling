import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import interpolate
from scipy import signal
import math


from yang import *

mu = (2500 ** 2) * 2500 # Vs^2 * density
nu = 0.25

(Y,X)=np.meshgrid(np.linspace(-1000,1000,100),np.linspace(-1000,1000,100));
(dE, dN, dZ) = yangmodel(0,0,1000,100,0.99,1,mu,nu,89.99,0,X.flatten(),Y.flatten(),np.zeros(X.flatten().shape[0]))

np.savetxt('X.txt',X.flatten())
np.savetxt('Y.txt',Y.flatten())

np.savetxt('yangpython.txt',np.column_stack((dE,dN,dZ)))
C = np.loadtxt('yangmatlab5.txt')

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(C[:,0]-dE)
ax2 = fig.add_subplot(312)
ax2.plot(C[:,1]-dN)
ax3 = fig.add_subplot(313)
ax3.plot(C[:,2]-dZ)

plt.show()
