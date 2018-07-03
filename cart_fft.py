#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
# Scipy imports
from scipy.special import kn, gamma 
from scipy.special import i0, i1, ellipe, ellipk  
from scipy.ndimage import filters
# Argparse
import argparse
from argparse import RawTextHelpFormatter

#=====================================================================
#
#  Code: kbt_fft.py
#
#  Purpose: Test the FFT method for self-gravity in Cartesian coords.
#           This method uses the convolution theorem to obtain 
#           FFT(Phi) = FFT(K)*FFT(\Sigma), where K is the Kernel
#           Phi      = IFFT( FFT(K)*FFT(\Sigma) )   
#
#  References: Binney & Tremaine (2008)
#              Kalnajs (1971)  
#
#  Keywords: python cart_fft.py -h   
#
#  Usage: python cart_fft.py args   
#
#  Author: John Dupuy & Boyu Xue 
#          UNC Chapel Hill
#  Date:    06/28/18 
#  Updated: 06/28/18 
#======================================================================

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
def Krazor(x,y): 
    eps = 0.0 
    R  = np.sqrt( x**2. + y**2.)
    return -G/R   

# \func pad_array(), pads array with zeros 
def pad_array(data, n):
    
    ny, nx    = data.shape 
    pad_array = np.zeros((n*ny, n*nx), dtype='complex') 

    pad_array[((ny*(n-1))/2):((ny*(n+1))/2), ((nx*(n-1))/2):((nx*(n+1))/2)] = data.copy()
    
    return pad_array  

# \func pad_kernel(), compute kernel at zero-padded points
def pad_kernel(Kfunc, x, y, n): 
    # get spacing
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    nx = len(x) 
    ny = len(y) 
    # no. of pad cells on each side 
    npadx = (nx*n-nx)/2
    npady = (ny*n-ny)/2
    # get new umin, umax
    xmin   = x[ 0] - npadx*dx
    xmax   = x[-1] + npadx*dx
    ymin   = y[ 0] - npady*dy
    ymax   = y[-1] + npady*dy
    # get new u array
    xkern  = np.arange(xmin,xmax+dx,dx)
    ykern  = np.arange(ymin,ymax+dy,dy)

    X, Y = np.meshgrid(xkern,ykern,indexing='xy')
    return Kfunc(X,Y) 


# \func init_data()
# Initialize the density data based on test problem  
def init_data(prob,x,y): 
    nx1, nx2 = len(x), len(y) 
    X, Y = np.meshgrid(x,y, indexing='xy')
    R, p = np.sqrt(X**2. + Y**2.), np.arctan2(Y, X)
    p[ p < 0] += 2.*np.pi

    if prob == 'constant':
        sig0 = 2.e-3
        data = sig0*np.ones( (nx2, nx1) ) 
    elif prob == 'exp':
        Rd   = 0.5
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
        data = 2.*dens(3.,1e-3) + 0.5*dens(3.,np.pi+1e-3) + dens(2.,0.75*np.pi) 
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
# Initialize the 2D cartesian grid 
# x1: x, x2: y 
def init_grid(x1min,x1max,x2min,x2max,nx1,nx2): 
    # First get the face-centered grid  
    x1f = np.linspace(x1min,
                      x1max, nx1+1)
    x2f = np.linspace(x2min,
                      x2max, nx2+1) 

    # Now get the cell-centered grid, no ghost zones 
    dx1 = x1f[1]-x1f[0]
    x1c = np.arange(x1min+0.5*dx1,
                    x1max   , dx1)
    
    dx2 = x2f[1] - x2f[0] 
    x2c = np.arange(x2min+0.5*dx2,
                    x2max,    dx2) 

    return x1f, x1c, x2f, x2c 


#\func initialize()
# Initializes everything to be passed to the solver 
def initialize(prob, zfunc):
    # first setup grid data based on prob  
    nx1 = 256 
    nx2 = 256 
    x1min = -10.
    x1max =  10.
    x2min = -10.
    x2max =  10.

    # Initialize the grid
    x1f, x1c, x2f, x2c = init_grid(x1min,x1max,
                                   x2min,x2max, nx1, nx2) 

    # Initialize the grid data 
    data  = init_data(prob,x1c,x2c) 
    data *= 2.*np.pi*G
    #data -= np.mean(data) 

    # Initialize Kdata
    X, Y  = np.meshgrid(x1c, x2c, indexing='xy')
    Kfunc = get_Kfunc(zfunc)
    Kdata = Kfunc(X,Y)  

    # Package the grid data 
    gdata = (x1f, x1c, x2f, x2c) 

    return gdata, data, Kdata  

#\func solve()
# Given the returned stuff from initialize
# solves for the potential \PHI(R,\phi)
def solve(myinit):
    # Parse mystuff
    gdata, data, Kdata = myinit 

    # Parse gdata 
    x1f, x1c, x2f, x2c = gdata 
    nx2  = len(x2c) 
    nx1  = len(x1c) 

    X, Y = np.meshgrid(x1c,x2c,indexing='xy')

    # Take the 2D FFT of data & the kernel
    pad  = 2 

    pdata  = np.zeros( (pad*nx2, pad*nx1) )
    #pKdata = np.zeros( (pad*nx2, pad*nx1) )

    pdata[ 0:nx2, 0:nx1] = data.copy() 
    #pKdata[0:nx2, 0:nx1] = Kdata.copy()

    fsize  = (pad*nx2, pad*nx1) 
    if pad != 1:
        p2, p1 = 0, 0 #nx2//2, nx1//2
    else:
        p2, p1 = 0, 0


    FS   = np.fft.fft2(pdata,s=fsize)
    #FK   = np.fft.fft2(pKdata,s=fsize)#/(2.*np.pi)**2.
    # Now explicitly multiply FS by the kernel
    dx1 = x1c[1] - x1c[0]
    dkx = 2.0*np.pi/float(pad*nx1)
    dx2 = x2c[1] - x2c[0]
    dky = 2.0*np.pi/float(pad*nx2)
    for j in range(pad*nx2):
        #ky = j*dky
        if j < (pad*nx2) //2:
            ky = j*dky/dx2
        else:
            ky = (j-pad*nx2)*dky/dx2
        for i in range(pad*nx1):
            #kx = i*dkx
            if i < (pad*nx1)//2:
                kx = i*dkx/dx1
            else:
                kx = (i-pad*nx1)*dkx/dx1
            if (i == 0) and (j == 0):
                FS[j,i] = np.mean(pdata)
            else:
                pcoeff   = -1./np.sqrt(kx**2. + ky**2.)
                #pcoeff   = 1.0/((2.0*np.cos(i*dkx)-2.0)/dx1**2. +
                #                (2.0*np.cos(j*dky)-2.0)/dx2**2.)
                FS[j,i] *= pcoeff 
    FV = FS.copy() 
            
    # Compute potential in Fourier space via the convolution theorem
    #FV   = FK*FS
    #FV[0,0] = 0.
   
    # Now take the inverse 2D FFT of FV to get Phi
    Phi = np.fft.ifft2(FV)#/(2.*np.pi)**2.
    Phi = Phi[p2:p2+nx2, p1:p1+nx1]
    #Phi  = filters.convolve(data, Kdata, mode='constant',cval=0)
    
    #print(np.mean(np.abs(Phi - Pc))) 

    return np.real(Phi) 

#\func plot()
# Given Phi and myinit
# plots the potential 
def plot(Phi, myinit, prob, ana):
    # Parse myinit
    gdata, Sdata, Kdata = myinit

    # Parse gdata 
    x1f, x1c, x2f, x2c = gdata 
    nx2, nx1 = len(x2c), len(x1c) 

    # Plot Phi
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(111) 

    X, Y   = np.meshgrid(x1f, x2f, indexing='xy')

    if ana:
        XC, YC   = np.meshgrid(x1c, x2c, indexing='xy') 
        X1C, X2C = np.sqrt( XC**2. + YC**2.), np.arctan2(YC, XC)
        X2C[ X2C < 0 ] += 2.*np.pi 

        if prob == 'exp':
            Rd   = 0.5
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
        plt.plot(x1c, Phi[nx2/2,:],'b.',label='$\Phi_G$ (numerical)')
        plt.plot(x1c, Phiana[nx2/2,:],'r--',label='$\Phi_G$ (analytical)') 
        plt.xlabel('R [code units]')
        plt.ylabel('$\Phi_G$')
        plt.title('RMS error: %3.2e' %(rms) )  
        plt.legend(loc=4)


    
    im  = ax1.pcolorfast(X, Y, Phi,cmap='magma') 
    ax1.set_aspect('equal') 
    ax1.set_xlabel('x [code units]')
    ax1.set_ylabel('y [code units]') 
    cbar = fig.colorbar(im,label='$\Phi_G$')

    # Plot data
    data = Sdata
    '''
    fig1 = plt.figure(facecolor='white')
    ax2 = fig1.add_subplot(111) 
    im  = ax2.pcolorfast(X, Y, data,cmap='magma') 
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
