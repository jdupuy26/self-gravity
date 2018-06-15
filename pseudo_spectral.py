#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn 
from scipy.integrate import trapz


#===================================================================
#
#  Code: pseudo_spectral.py
#
#  Purpose: Test the pseudo spectral method for self-gravity in
#           cylindrical coordinates. The goal of this code is simply 
#           to learn how the method works and apply it to test problems.  
#
#  References: Jung, M., Illenseer, T.F., Duschl, W.J. A&A (2018)
#                "Multi-scale simulations of black hole accretion in 
#                 barred galaxies: Numerical methods and tests."
#              Li, S., Buoni, M.J., Li, H., ApJ (2009)
#                "A fast potential and self-gravity solver for
#                 nonaxisymmetric disks" 
#              Chan, C., Psaltis, D., Ozel, F. (2005)
#                "Spectral methods for time-dependent studies of 
#                 accretion flow II..." 
#
#  Keywords: python pseudo_spectral.py -h   
#
#  Usage: python pseudo_spectral.py args   
#
#  Author: John Dupuy & Boyu Xue 
#          UNC Chapel Hill
#  Date:    06/15/18 
#  Updated: 06/15/18 
#====================================================================

#---------------- GLOBAL CONSTANTS ----------------------------------
G = 1.0 # gravitational constant 

#---------------- FUNCTION DEFINITIONS ------------------------------
# \func get_Gfunc() 
# Given, Z(r,z) this returns the corresponding Green's function 
def get_Gfunc(zfunc):
    if   zfunc == 'delta':
        Gfunc = Grazor
    elif zfunc == 'gauss':
        Gfunc = Ggauss  
    else:
        print("[get_Gfunc]: zfunc not understood, got %s, exiting..." %(zfunc))
        quit() 
    return Gfunc 

# \func Grazor() 
# Green's function for razor thin disk 
#   Z(r,z) = \delta (z) 
# pmpp = \phi - \phi' 
def Grazor(r, rp, pmpp): 
    R2  = r**2. + rp**2. - 2.*r*rp*np.cos(pmpp)
    return R2**(-0.5)  

# \func Ggauss()
# Green's function for Gaussian vertical structure
#   Z(r,z) \propto exp(-z^2/( 2 H^2 (r) ))
def Ggauss(r, rp, pmpp):
    # here H is the scale height function 
    R2  = r**2. + rp**2. - 2.*r*rp*np.cos(pmpp) 
    fac = -np.exp(R2/4.0)/(np.sqrt(2.*np.pi) * H(rp) )
    # here kn(0,x) is the modified bessel function of the 2nd kind 
    return fac*kn(0,R2/4.0)  

# \func H()
# Scale height function, currently just assumed to be constant  
def H(r):
    return 1.0  

# \func Ifunc()
def Ifunc(Gfunc, r, rp, pmpp):
    return 2.0*np.pi*G*rp*Gfunc(r,rp,pmpp) 

# \func init_data()
# Initialize the density data based on test problem  
def init_data(prob,nx1,nx2): 
    if prob == 'constant':
        data = np.ones( (nx2, nx1) ) 
    else:
        print('[init_data]: Prob not understood, got %s, exiting...' %(prob))
        quit() 
    return data 

# \func init_grid()
# Initialize the 2D polar grid 
# x1: R, x2: \phi 
def init_grid(x1min,x1max,nx1,nx2,x2min=0.0,x2max=2.0*np.pi,log=False): 

    # First get the face-centered grid  
    if log: # for a log grid 
        lx1f = np.linspace(np.log(x1min),
                           np.log(x1max), nx1+1)
        x1f  = np.exp(lx1f) 
    else:
        x1f  = np.linspace(x1min,
                           x1max, nx1+1)
    x2f = np.linspace(x2min,
                      x2max, nx2+1) 
    # Now get the cell-centered grid
    x1c = 0.5*(x1f[1:] + x1f[:-1])
    x2c = 0.5*(x2f[1:] + x2f[:-1]) 
    # So that x1c and x1f are the same shape
    # exclude the last point 
    x1f = x1f[:-1]
    x2f = x2f[:-1] 

    return x1f, x1c, x2f, x2c 


# \func fft()
# Given a 2D array (\phi, R), this computes 
# and returns the Fourier transform of the data over \phi
# Note that the \phi axis is assumed to be the 0 axis 
def fft(data): 
    fdata = np.fft.fft(data, axis=0)  
    return fdata 

# \func ifft()
# Given a 2D array (\phi, R), this computes 
# and returns the inverse Fourier transform of the data over \phi
# Note that the \phi axis is assumed to be the 0 axis 
def ifft(data): 
    fdata = np.fft.ifft(data, axis=0)  
    return fdata 



def main():
    nx1 = 512
    nx2 = nx1

    x1f, x1c, x2f, x2c = init_grid(1.,20.,nx1,nx2)
    Gfunc = get_Gfunc('delta')  

    pmpp  = x2c-x2f
    RP, PMPP, R = np.meshgrid(x1c,pmpp,x1f)  

    # Get Idata 
    Idata = Ifunc(Gfunc, R, RP, PMPP) 
    # Get data 
    data = init_data('constant', nx1, nx2) 
        
    # Take fourier transform of data and Idata 
    fdata  = fft(data)
    fIdata = fft(Idata) 

    # Now compute potential in Fourier space 
    Phim = np.zeros( (nx2, nx1), dtype='complex') 
    dr   = x1f[1]-x1f[0]
    for j in range(nx2):
        for i in range(nx1):
            # here the last axis is the rp axis (the axis of integration)  
            Phim[j,i] = trapz(fdata[j,:] * fIdata[j,i,:], dx = (x1f[1]-x1f[0]) )
              
    # Now transfer the potential back to real space
    # Note that Phi is defined at cell walls 
    Phi = ifft(Phim) 

    # Now plot the resultant potential  
    fig = plt.figure()
    ax1 = fig.add_subplot(111) 

    x1f, x2f   = np.meshgrid(x1f, x2f, indexing='xy')
    x1cf, x2cf = x1f*np.cos(x2f), x1f*np.sin(x2f) 
    
    im  = ax1.pcolorfast(x1cf, x2cf, Phi[:-1,:-1]) 
    ax1.set_aspect('equal') 
    cbar = fig.colorbar(im,label='$\Phi$')
    # Compuate gr 
    gr = (Phi[0,1:] - Phi[0,:-1])/dr 

    
    plt.figure()
    plt.semilogy(np.abs(gr)) 
    plt.show() 
     






main() 
