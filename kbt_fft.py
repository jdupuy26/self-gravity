#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
# Scipy imports
from scipy.special import kn, gamma 
from scipy.special import i0, i1, ellipe, ellipk  
from scipy.signal import convolve2d
# Argparse
import argparse
from argparse import RawTextHelpFormatter

#===================================================================
#
#  Code: kbt_fft.py
#
#  Purpose: Test the FFT method of Kalnajs (1971)   
#
#  References: Binney & Tremaine (2008), Section 2.6.3
#              Kalnajs (1971)  
#
#  Keywords: python kbt_fft.py -h   
#
#  Usage: python kbt_fft.py args   
#
#  Author: John Dupuy & Boyu Xue 
#          UNC Chapel Hill
#  Date:    06/28/18 
#  Updated: 06/28/18 
#====================================================================

#---------------- GLOBAL CONSTANTS ----------------------------------
G = 1.0 # gravitational constant 

#---------------- FUNCTION DEFINITIONS ------------------------------
# \func get_Kfunc() 
# Given, Z(r,z) this returns the corresponding Green's function 
def get_Kfunc(zfunc):
    if zfunc == 'delta':
        Kfunc = Krazor
    else:
        print("[get_Gfunc]: zfunc not understood, got %s, exiting..." %(zfunc))
        quit() 
    return Kfunc 

# \func Krazor() 
# Kernel for razor thin disk 
#   Z(r,z) = \delta (z) 
def Krazor(u,p): 
    eps = 0.0 
    R2  = np.cosh(u) - np.cos(p) + eps**2.      
    return -G*R2**(-0.5)/np.sqrt(2.)  

# \func Nfunc(), explicit form for FT of the kernel
def N(a,m):

    mask = (a**2. + m**2.) > 1e20
    N = np.zeros(a.shape, dtype='complex')

    N[ mask] = np.pi*np.sqrt(a[mask]**2. + m[mask]**2.)
    N[~mask] = nfunc(a[~mask],m[~mask])
    
    return N

def nfunc(a,m):
    arg1 = (m + 0.5 + 1j*a)/2.  
    arg2 = (m + 0.5 - 1j*a)/2. 
    arg3 = (m + 1.5 + 1j*a)/2. 
    arg4 = (m + 1.5 - 1j*a)/2. 

    f1 = gamma(arg1)
    f2 = gamma(arg2)
    f3 = gamma(arg3)
    f4 = gamma(arg4)
     
    return np.pi*(f1*f2)/(f3*f4) 


# \func pad_array(), pads array with zeros 
def pad_array(data, n):
    
    ny, nx    = data.shape 
    pad_array = np.zeros((ny, n*nx), dtype='complex') 

    pad_array[:, ((nx*(n-1))/2):((nx*(n+1))/2)] = data.copy()
    
    return pad_array  

# \func pad_kernel(), compute kernel at zero-padded points
def pad_kernel(Kfunc, u, p, n): 
    # get spacing
    du = u[1]-u[0]
    
    nx = len(u) 
    # no. of pad cells on each side 
    npad = (nx*n-nx)/2
    # get new umin, umax
    umin   = u[ 0] - npad*du
    umax   = u[-1] + npad*du
    # get new u array
    ukern  = np.arange(umin,umax+du,du)

    U, P = np.meshgrid(ukern,p,indexing='xy')
    return ukern, Kfunc(U,P) 


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
        sig = 0.1 
        Rk   = lambda rk,pk: np.sqrt( R**2. + rk**2. - 2.*R*rk*np.cos(p-pk) )
        dens = lambda rk,pk: np.exp(-Rk(rk,pk)/sig)/(2.*np.pi*sig**2.)
        data = 2.*dens(3., 1e-3) + 0.5*dens(3.,np.pi+1e-3) + dens(2.,0.75*np.pi) 
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
def init_grid(x1min,x1max,x2min,x2max,nx1,nx2): 
    # First get the face-centered grid  
    lx1f = np.linspace(np.log(x1min),
                       np.log(x1max), nx1+1)
    x1f  = np.exp(lx1f) 
    
    x2f = np.linspace(x2min,
                      x2max, nx2+1) 

    # Now get the cell-centered grid, no ghost zones 
    dlx1 = lx1f[1]-lx1f[0]
    lx1c = np.arange(np.log(x1min)+0.5*dlx1,
                     np.log(x1max)         , dlx1)
    
    x1c  = 0.5*(x1f[:-1] + x1f[1:]) #np.exp(lx1c)  

    #x1c  = np.exp(lx1c) 

    dx2 = x2f[1] - x2f[0] 
    x2c = np.arange(x2min+0.5*dx2,x2max, dx2) 

    return x1f, x1c, x2f, x2c 


#\func initialize()
# Initializes everything to be passed to the solver 
def initialize(prob, zfunc):
    # first setup grid data based on prob  
    nx1 = 256 
    nx2 = 256
    x2min = 0.0
    x2max = 2.*np.pi
    if prob == 'constant':
        x1min = 0.4
        x1max = 2.5
    elif prob == 'exp':
        x1min = 1e-2 
        x1max = 20.0
    elif prob == 'mestel':
        x1min = 1e-1
        x1max = 20.0
    elif prob == 'kuzmin':
        x1min = 0.1 
        x1max = 20.0
    elif prob == 'cylinders': # cylinder prob of Chan et al. 2005
        x1min = 1e-2 
        x1max = 20.0
    elif prob == 'rotcyl':
        x1min = 0.2
        x1max = 1.8
    else:
        print('[initialize]: prob %s not understood, exiting...' %(prob)) 
        quit()

    # Initialize the grid
    x1f, x1c, x2f, x2c = init_grid(x1min,x1max,
                                   x2min,x2max, nx1, nx2) 

    # Initialize the grid data 
    data  = init_data(prob,x1c,x2c) 

    # Compuate the reduced surface density S = exp(3u/2) \Sigma 
    u     = np.log(x1c.copy())

    U, P  = np.meshgrid(u,x2c,indexing='xy') 
    Sdata = np.exp(1.5*U)*(data.copy())  

    # Package the grid data 
    gdata = (x1f, x1c, x2f, x2c) 

    return gdata, Sdata 

#\func solve()
# Given the returned stuff from initialize
# solves for the potential \PHI(R,\phi)
def solve(myinit):
    # Parse mystuff
    gdata, Sdata    = myinit 

    # Parse gdata 
    x1f, x1c, x2f, x2c = gdata 
    u    = np.log(x1c.copy())
    nx2  = len(x2c) 
    nx1  = len(x1c) 

    U, P = np.meshgrid(u,x2c,indexing='xy')

    # Take the 2D FFT of Sdata
    pad           = 4 
    Spad          = np.zeros((nx2, pad*nx1))
    Spad[:,0:nx1] = Sdata.copy() 

    FS         = np.fft.fft2(Spad) 
    # Compute the potential in Fourier space
        # First get m, alpha coeffs 
    du   =   u[1] -   u[0] 
    dp   = x2c[1] - x2c[0]
        # get m, a (kp, ku)
    dku  = 2.0*np.pi/float(pad*nx1)
    dkp  = 2.0*np.pi/float(    nx2)

    ku   = np.arange(0, pad*nx1, 1.)*dku/du
    kp   = np.arange(0,     nx2, 1.)*dkp/dp 

    ku[ (pad*nx1)//2: ] -= pad*nx1*dku/du
    kp[ (    nx2)//2: ] -=     nx2*dkp/dp
    
    A, M = np.meshgrid(ku,kp,indexing='xy')

    Nam  = N(A,M)
    Nam[np.isnan(Nam)] = 0.
    '''
    Nam  = np.zeros( (nx2, pad*nx1),dtype='complex')
    for j in range(nx2):
        for i in range(pad*nx1):
            Nam[j,i] = N(ku[i],kp[j])
            if np.isnan(Nam[j,i]):
                Nam[j,i] = 0.
                print(j,i)
    '''
    FV    = -G*Nam*FS 
    
    # Now take the inverse 2D FFT of FV to get V(u,p)
    Vup = np.fft.ifft2(FV)  
    Vup = Vup[:,:nx1].copy() 

    # Apply u-factor to get Phi 
    Phi  = np.real(Vup)*(np.exp(-0.5*U))

    return Phi 

#\func plot()
# Given Phi and myinit
# plots the potential 
def plot(Phi, myinit, prob, ana):
    # Parse myinit
    gdata, Sdata = myinit

    # Parse gdata 
    x1f, x1c, x2f, x2c = gdata 
    u    = np.log(x1c) 

    # Plot Phi
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(111) 

    X1F, X2F   = np.meshgrid(x1f, x2f, indexing='xy')
    x1cf, x2cf = X1F*np.cos(X2F), X1F*np.sin(X2F) 

    if ana:
        X1C, X2C = np.meshgrid(x1c, x2c, indexing='xy') 
        if prob == 'exp':
            Rd   = 1. 
            sig0 = 1.
            y   = X1C/(2.*Rd)  
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
            sig  = 0.1
            Rk   = lambda rk,pk: np.sqrt( X1C**2. + rk**2. - 2.*X1C*rk*np.cos(X2C-pk) )
            yk   = lambda rk,pk: Rk(rk,pk)/(2.*sig)
            Phik = lambda rk,pk: -0.5*(G/sig**2.)*Rk(rk,pk)*( i0(yk(rk,pk))*kn(1,yk(rk,pk)) - i1(yk(rk,pk))*kn(0,yk(rk,pk)) )

            Phiana = 2.*Phik(3.,1e-3) + 0.5*Phik(3.,np.pi+1e-3) + Phik(2.0,0.75*np.pi)

        else:
            print('[plot]: Analytic solution not provided for prob: %s, exiting...' %(prob))
            quit() 
            
            
        print('[plot]: Plotting analytical solution for %s prob' %(prob)) 
        # Compute RMS error 
        rms = np.sqrt( np.sum((Phiana - Phi)**2.)/(float(Phiana.size)) )

        # Plot analytic result 
        plt.figure(facecolor='white')
        plt.plot(x1c, Phi[0],'b.',label='$\Phi_G$ (numerical)')
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

    # Plot data
    data = Sdata/np.exp(1.5*u)
    '''
    
    fig1 = plt.figure(facecolor='white')
    ax2 = fig1.add_subplot(111) 
    im  = ax2.pcolorfast(x1cf, x2cf, data,cmap='magma') 
    ax2.set_aspect('equal') 
    ax2.set_xlabel('x [code units]')
    ax2.set_ylabel('y [code units]') 
    cbar = fig1.colorbar(im,label='$\Sigma$')
    '''
    plt.show() 
    
    return 

def main():
    # Step 0) Read in system arguments 
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--prob',type=str,default='exp',
                        help="Test problem to solve, options are:\n"
                             "      (a) constant (b) exp",required=False)
    parser.add_argument('--ana', action='store_true',required=False,
                        help="Switch to compare numerical to analytical solution")
    parser.add_argument('--zfunc',type=str,default='delta',
                        help="Z(r,z) profile to use, options are:\n"
                             "      (a) delta ",required=False)
    parser.add_argument('--gana', action='store_true', required=False,
                        help="Switch to compare numerical grav acc to analytic grav acc")

    args  = parser.parse_args()
    prob  = args.prob
    zfunc = args.zfunc
    ana   = args.ana
    gana  = args.gana

    
    # Step 1) Initialize the system
    pass_to_solve = initialize(prob,zfunc) 
    # Step 2) Solve the system
    Phi           = solve(pass_to_solve) 
    # Step 3) Plot the system 
    plot(Phi, pass_to_solve, prob, ana) 

    return 
main() 
