#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
# Scipy imports
from scipy.special import kn 
from scipy.special import i0, i1, ellipe, ellipk  
from scipy.integrate import trapz, simps
from scipy.fftpack import dct 
# Argparse
import argparse
from argparse import RawTextHelpFormatter

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
    elif zfunc == 'cons':
        Gfunc = Glog 
    else:
        print("[get_Gfunc]: zfunc not understood, got %s, exiting..." %(zfunc))
        quit() 
    return Gfunc 

# \func Grazor() 
# Green's function for razor thin disk 
#   Z(r,z) = \delta (z) 
# pmpp = \phi - \phi' 
def Grazor(r, rp, pmpp): 
    eps = 0.0 
    R2  = r**2. + rp**2. - 2.*r*rp*np.cos(pmpp) + eps**2. 
    return -R2**(-0.5)  

# \func Glog() 
# Green's function for infinite disk 
#   Z(r,z) = const 
# pmpp = \phi - \phi' 
def Glog(r, rp, pmpp): 
    eps = 0.0 
    R2  = r**2. + rp**2. - 2.*r*rp*np.cos(pmpp) + eps**2. 
    return np.log(R2)  



# \func Ggauss()
# Green's function for Gaussian vertical structure
#   Z(r,z) \propto exp(-z^2/( 2 H^2 (r) ))
def Ggauss(r, rp, pmpp):
    # here H is the scale height function 
    eps = 0.0 
    R2  = (r**2. + rp**2. - 2.*r*rp*np.cos(pmpp) + eps**2.)/(H(rp))**2. 
    fac = -np.exp(R2/4.0)/(np.sqrt(2.*np.pi) * H(rp) )
    # here kn(0,x) is the modified bessel function of the 2nd kind 
    fac *= kn(0,R2/4.) 
    print(kn(0,1.))
    return fac  

# \func H()
# Scale height function, currently just assumed to be constant  
def H(r):
    return 100

# \func Ifunc()
def Ifunc(Gfunc, r, rp, pmpp):
    return 2.0*np.pi*G*rp*Gfunc(r,rp,pmpp) 

# \func init_data()
# Initialize the density data based on test problem  
def init_data(prob,R,p): 
    nx1, nx2 = len(R), len(p) 
    R, p = np.meshgrid(R,p, indexing='xy') 
    if prob == 'constant':
        sig0 = 2.e-3
        data = sig0*np.ones( (nx2, nx1) ) 
    elif prob == 'exp':
        Rd   = 1.
        data = np.exp(-R/Rd)
    elif prob == 'mestel':
        v0   = 1.
        data = v0**2./(2.*np.pi*G*R) 
    elif prob == 'kuzmin':
        a = 1.
        M = 1.
        data = a*M/(2.*np.pi*(R**2. + a**2.)**(1.5)) 
    elif prob == 'cylinders':
        sig = 1.0
        Rk   = lambda rk,pk: np.sqrt( R**2. + rk**2. - 2.*R*rk*np.cos(p-pk) )
        dens = lambda rk,pk: np.exp(-Rk(rk,pk)/sig)/(2.*np.pi*sig**2.)
        data = 2.*dens(1.,1e-3) + 0.5*dens(1.,np.pi+1e-3) + dens(0.9,0.75*np.pi) 
    elif prob == 'rotcyl':
        sig = 0.1
        Rk   = lambda rk,pk: np.sqrt( R**2. + rk**2. - 2.*R*rk*np.cos(p-pk) )
        dens = lambda rk,pk: np.exp(-Rk(rk,pk)**2./(2.*sig**2.))/(2.*np.pi*sig**2.)
        data = dens(1., np.pi/4. + 1e-3) + dens(1.,5.*np.pi/4. + 1e-3) + 2e-2/(3.2*np.pi) 
    else:
        print('[init_data]: Prob not understood, got %s, exiting...' %(prob))
        quit() 
    return data 

# \func init_grid()
# Initialize the 2D polar grid 
# x1: R, x2: \phi 
def init_grid(x1min,x1max,x2min,x2max,nx1,nx2,log=False): 
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

    # Now get the cell-centered grid, no ghost zones 
    if log:
        dlx1 = lx1f[1]-lx1f[0]
        lx1c = np.arange(np.log(x1min)+0.5*dlx1,
                         np.log(x1max)         , dlx1)
        x1c  = np.exp(lx1c)  
    else:
        dx1 = x1f[1] - x1f[0]
        x1c = np.arange(x1min+0.5*dx1,
                        x1max        , dx1)   
    
    dx2 = x2f[1] - x2f[0] 
    x2c = np.arange(x2min+0.5*dx2,x2max, dx2) 

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


#\func initialize()
# Initializes everything to be passed to the solver 
def initialize(prob, zfunc, log):
    # first setup grid data based on prob  
    nx1 = 128
    nx2 = 128
    if prob == 'constant':
        x1min = 0.4
        x1max = 2.5
        x2min = 0.0
        x2max = 2.0*np.pi
    elif prob == 'exp':
        x1min = 1e-2
        x1max = 20.0
        x2min = 0.0
        x2max = 2.0*np.pi
    elif prob == 'mestel':
        x1min = 1e-1
        x1max = 20000.0
        x2min = 0.0
        x2max = 2.0*np.pi
    elif prob == 'kuzmin':
        x1min = 0.1 
        x1max = 20.0
        x2min = 0.0
        x2max = 2.0*np.pi
    elif prob == 'cylinders': # cylinder prob of Chan et al. 2005
        x1min = 1.0
        x1max = 5.0
        x2min = 0.0
        x2max = 2.0*np.pi
    elif prob == 'rotcyl':
        x1min = 0.2
        x1max = 1.8
        x2min = 0.0
        x2max = 2.0*np.pi
    else:
        print('[initialize]: prob %s not understood, exiting...' %(prob)) 
        quit()

    # Initialize the grid
    x1f, x1c, x2f, x2c = init_grid(x1min,x1max,
                                   x2min,x2max, nx1, nx2, log=log) 
    # Choose the Gfunc
    Gfunc = get_Gfunc(zfunc)

    # Initialize the grid data 
    data  = init_data(prob,x1c,x2c) 
    
    # Set \phi - \phi'
    pmpp = x2c 

    # Initialize I(R,R',\phi-\phi') 
    R, PMPP, RP = np.meshgrid(x1c, pmpp, x1c) 
    Idata = Ifunc(Gfunc, R, RP, PMPP) 
    '''
    Idata = np.zeros( (len(x2c), len(x1c), len(x1c)) )
    for j in range(nx2):
        for i in range(nx1):
            for ii in range(nx1):
                Idata[j,i,ii] = Ifunc(Gfunc, x1c[i], x1c[ii], x2c[j])
    '''            
    
    # Package the grid data 
    gdata = (x1f, x1c, x2f, x2c) 

    return gdata, data, Idata  

#\func solve()
# Given the returned stuff from initialize
# solves for the potential \PHI(R,\phi)
def solve(myinit):
    # Parse mystuff
    gdata, data, Idata = myinit 

    # Parse gdata 
    x1f, x1c, x2f, x2c = gdata 

    nx2  = len(x2c) 
    nx1  = len(x1c) 

    # Take fourier transform of data and Idata 
    fdata  = fft(data)
    fIdata = fft(Idata) # dct(Idata,type=2,axis=0)/(2.0*np.pi)  

    # Now compute potential in Fourier space 
    Phim = np.zeros( (nx2, nx1), dtype='complex') 
    for j in range(nx2):
        Phim[j] = trapz(fdata[j,:] * fIdata[j,:,:], axis=-1, x=x1c) 
        '''
        for i in range(nx1):
            # here the last axis is the rp axis (the axis of integration)  
            Phim[j,i] = trapz(fdata[j,:] * fIdata[j,i,:], x=x1c)
        '''

    print(Phim[0,0]) 
              
    # Now transfer the potential back to real space
    Phi = ifft(Phim)/float(nx2)  
    #Phi = np.real(Phi) 

    print(np.mean(Phi)) 

    return Phi 
#\func plot()
# Given Phi and myinit
# plots the potential 
def plot(Phi, myinit, prob, ana):
    # Parse myinit
    gdata, data, Idata = myinit

    # Parse gdata 
    x1f, x1c, x2f, x2c = gdata 
    
    # Plot Phi
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(111) 

    X1F, X2F   = np.meshgrid(x1f, x2f, indexing='xy')
    x1cf, x2cf = X1F*np.cos(X2F), X1F*np.sin(X2F) 
    
    if ana:
        X1C, X2C = np.meshgrid(x1c, x2c, indexing='xy') 
        if prob == 'exp':
            sig  = 1.
            sig0 = 1.
            y   = X1C/(2.*sig)  
            Phiana = -np.pi*G*sig0*X1C*(i0(y)*kn(1,y) - i1(y)*kn(0,y))   
        elif prob == 'constant': 
            sig0  = 2.e-3
            v_max = x1c/np.max(x1c)
            u_min = np.min(x1c)/x1c 
            grana = 4*G*sig0*( (ellipe(v_max)-ellipk(v_max))/v_max + ellipk(u_min) - ellipe(u_min)) 

        elif prob == 'mestel':
            v0 = 1.
            eps    = np.max(x1f) 
            Phiana = v0**2.*(np.log(X1C/eps) + np.log(0.5) )  

        elif prob == 'kuzmin':
            a = 1.
            M = 1.
            Phiana = -G*M/np.sqrt( X1C**2. + a**2. ) 

        elif prob == 'cylinders':
            sig  = 1.0
            Rk   = lambda rk,pk: np.sqrt( X1C**2. + rk**2. - 2.*X1C*rk*np.cos(X2C-pk) )
            yk   = lambda rk,pk: Rk(rk,pk)/(2.*sig)
            Phik = lambda rk,pk: -0.5*(G/sig**2.)*Rk(rk,pk)*( i0(yk(rk,pk))*kn(1,yk(rk,pk)) - i1(yk(rk,pk))*kn(0,yk(rk,pk)) )

            Phiana = 2.*Phik(1.,1e-3) + 0.5*Phik(1.,np.pi+1e-3) + Phik(0.9,0.75*np.pi)
            
            
        print('[plot]: Plotting analytical solution for %s prob' %(prob)) 
        # Compute RMS error 
        rms = np.sqrt( np.sum((Phiana - Phi)**2.)/(float(Phiana.size)) )

        # Plot analytic result 
        plt.figure(facecolor='white')
        plt.plot(x1c, Phi[0],'b-',label='$\Phi_G$ (numerical)')
        plt.plot(x1c, Phiana[0],'r--',label='$\Phi_G$ (analytical)') 
        plt.xlabel('R [code units]')
        plt.ylabel('$\Phi_G$')
        plt.title('RMS error: %3.2e' %(rms) )  
        plt.legend(loc=4)


    
    im  = ax1.pcolorfast(x1cf, x2cf, Phi,cmap='magma') 
    ax1.set_aspect('equal') 
    ax1.set_xlabel('x [code units]')
    ax1.set_ylabel('y [code units]') 
    cbar = fig.colorbar(im,label='$\Phi_G$')
    
    plt.show() 
    return 

def main():
    # Step 0) Read in system arguments 
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--prob',type=str,default='exp',
                        help="Test problem to solve, options are:\n"
                             "      (a) constant (b) exp",required=False)
    parser.add_argument('--zfunc',type=str,default='delta',
                        help="Z(r,z) profile to use, options are:\n"
                             "      (a) delta (b) gauss",required=False)
    parser.add_argument('--log',action='store_true',required=False,
                        help="Switch for log-grid") 
    parser.add_argument('--ana', action='store_true',required=False,
                        help="Switch to compare numerical to analytical solution")
    args  = parser.parse_args()
    prob  = args.prob
    zfunc = args.zfunc 
    log   = args.log 
    ana   = args.ana

    # Step 1) Initialize the system
    pass_to_solve = initialize(prob,zfunc,log) 
    # Step 2) Solve the system
    Phi           = solve(pass_to_solve) 
    # Step 3) Plot the system 
    plot(Phi, pass_to_solve, prob, ana) 

    return 
main() 
