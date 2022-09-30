import numpy as np
from numpy import sqrt
import scipy.constants
import scipy.special

m_e=scipy.constants.m_e
m_p=scipy.constants.m_p
e=scipy.constants.e
c=scipy.constants.c
u = scipy.constants.u
pi = np.pi
r0 = scipy.constants.physical_constants['classical electron radius'][0]
epsilon_0=scipy.constants.epsilon_0
exp=np.exp
Il=scipy.special.iv #modified bessel function of first kind


def k_to_l(k):
    return 2.*np.pi/k

def ohm_pe(ne_pcc):
    return 5.64e4*ne_pcc**0.5


def S_k_omega_unscaled(l, l0, theta, A, T_e, T_i, n_e, Z, \
    v_fi=0, v_fe=0):
    '''
    Returns a spectral density function.
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temperatures
    Electron velocity is with respect to ion velocity
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in nm
    Theta is angle between k_in and k_s in degrees
    A i atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    V_fi and V_fe in m/s
    '''
    
    lambda_in = l0*1e-9
    lambda_range = l*1e-9
    
    #physical parameters
    pi=np.pi
    m_i=u*A
    om_pe=5.64e4*n_e**0.5
    
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b=sqrt(2*e*T_i/m_i)
    x_e=(omega/k - (v_fe+v_fi))/a 
    x_i=(omega/k-v_fi)/b
    lambda_De=7.43*(T_e/n_e)**0.5 #Debeye length in m
    #the all important alpha parameter
    alpha=1/(k*lambda_De)
    #set up the Fadeeva function
    w=scipy.special.wofz
    chi_i=alpha**2*Z*T_e/T_i*(1+1j*sqrt(pi)*x_i*w(x_i)) #ion susceptibility
    chi_e=alpha**2*(1+1j*sqrt(pi)*x_e*w(x_e))#electron susceptibility
    epsilon=1+chi_e+chi_i#dielectric function
    fe0=1/(sqrt(pi)*a)*np.exp(-x_e**2)#electron Maxwellian function
    fi0=1/(sqrt(pi)*b)*np.exp(-x_i**2)#ion Maxwellian
    Skw=2*pi/k*(abs(1-chi_e/epsilon)**2*fe0+Z*abs(chi_e/epsilon)**2*fi0)
    return Skw, alpha


def S_k_omega(l, l0, theta, A, T_e, T_i, n_e, Z, \
    v_fi=0, v_fe=0):
    '''
    Returns a normalised spectral density function.
    '''
    Skw, alpha = S_k_omega_unscaled(l, l0, theta, A, T_e, T_i, n_e, Z, \
    v_fi, v_fe)
    S_norm = Skw/Skw.max()
    return S_norm, alpha

def Ps(Pi, dSig, L, l, l0, theta, A, T_e, T_i, n_e, Z, \
    v_fi=0., v_fe=0.):
    '''
    Returns spectrally resolved scattered power in w/m/sr. Args:
    Pi - input power [w]
    dSid - Solid angle subtended [sr]
    l - spectral grid [nm]
    l0 probe wavelength [nm]
    theta - scatteting angle [degs]
    A - atomic mass number
    Z - ionic charge
    T_e/i - elec/ion temp [eV]
    ne - electron density [pcc]
    v_fe/i - elec/ion drift velocity [m/s]

    '''

    ne_perm = n_e*1e6
    pi=np.pi
    m_i=m_p*A
    om_pe=5.64e4*n_e**0.5

    lambda_in = l0*1e-9
    lambda_range = l*1e-9
    
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma
    ls = 2.*pi/ks #Scattering wavelength in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift

    Skw = S_k_omega_unscaled(l, l0, theta, A, T_e, T_i, n_e, Z, v_fi, v_fe)
    d_omega_s = (2.*pi*c)**2/(omega_s*l**3) #(*dls)

    return (Pi*r0**2*dSig*d_omega_s/(2*pi))*(1+2.*omega/omega_i)*L*ne_perm*Skw

def S_k_omega_multi_ion_unscaled(l, l0, theta, A, T_e, T_i, n_e, Z, f_i,\
    v_i=0, j=0):
    '''
    Returns a spectral density function.
    Implements the model of Sheffield (2nd Ed.)
    One ion, one electron species with independent temperatures
    Electron velocity is with respect to ion velocity
    Returns S(k,w) for each wavelength in lambda_range assuming
    input wavelength lambda_in. Both in nm
    Theta is angle between k_in and k_s in degrees
    A i atomic mass, Z is ion charge
    T_e, T_i in eV, n_e in cm^-3
    v_i in m/s, j in A/cm^2
    '''
    
    lambda_in = l0*1e-9
    lambda_range = l*1e-9
    
    #physical parameters
    pi=np.pi
    m_i = []
    for aa in A:
        m_i.append(u*aa)

    fz = 0.
    for zz, ff in zip(Z, f_i): # Rather inelegant conversion  
        fz+=zz*ff              # from fractional to abs ion
    n_it = n_e/fz              # densities
    n_i = []                   # (write it out...) 
    for ff in f_i:
        n_i.append(ff*n_it)

    j_it = 0. # Total ion-current density    
    for nn, zz, vv in zip(n_i, Z, v_i): 
        j_it += nn*zz*vv
    v_e = j_it/n_e - 1e-2*j/(e*n_e)
    om_pe=5.64e4*n_e**0.5
    
    #define omega and k as in Sheffield 113
    omega_i = 2*pi/lambda_in * c #input free space frequency
    ki = ((omega_i**2 - om_pe**2)/c**2)**0.5 #input wave-vector in plasma

    omega_s = 2*pi/lambda_range * c #scattering free space frequency
    ks = ((omega_s**2 - om_pe**2)/c**2)**0.5 #scattering wave-vector in plasma

    th=theta/180.0*np.pi
    k=(ks**2+ki**2-2*ks*ki*np.cos(th))**0.5
    omega=omega_s-omega_i #frequency shift

    #define dimensionless parameters ala Sheffield
    a=sqrt(2*e*T_e/m_e)
    b = [] 
    for tt, mm in zip(T_i, m_i):
        b.append(sqrt(2*e*tt/mm))
 
    x_e=(omega/k - v_e)/a
    x_i = []
    for bb, vv in zip(b, v_i):
        x_i.append((omega/k - vv)/bb)
    lambda_De=7.43*(T_e/n_e)**0.5 #electron Debeye length in m
    lambda_Di = [] # Debeye length for each ion species
    for tt, nn in zip(T_i, n_i):
        lambda_Di.append(7.43*np.sqrt(tt/nn)/zz)

    #the all important *electron* alpha parameter
    alpha_e=1/(k*lambda_De)
    alpha_i = []
    for ll in lambda_Di:
        alpha_i.append(1/(k*ll))

    #set up the Fadeeva function
    w=scipy.special.wofz

    chi_i = []
    for aa, xx in zip(alpha_i, x_i): 
        chi_i.append(aa**2*(1+1j*sqrt(pi)*xx*w(xx))) #ion susceptibility
    chi_i = np.array(chi_i)
    chi_i = np.sum(chi_i, axis=0)
    chi_e=alpha_e**2*(1+1j*sqrt(pi)*x_e*w(x_e))#electron susceptibility
    epsilon=1+chi_e+chi_i#dielectric function
    fe0=1/(sqrt(pi)*a)*np.exp(-x_e**2)#electron Maxwellian function
    Skw=2*pi/k*abs(1-chi_e/epsilon)**2*fe0
    for bb, xx, zz, nn in zip(b, x_i, Z, n_i): 
        fj0=1/(sqrt(pi)*bb)*np.exp(-xx**2)#ion Maxwellian
        Skw+=2*(pi/k)*(zz**2)*(nn/n_e)*abs(chi_e/epsilon)**2*fj0
    return Skw, alpha_e

def S_k_omega_multi_ion(l, l0, theta, A, T_e, T_i, n_e, Z, f_i,\
    v_i=0, j=0):
    s, a = S_k_omega_multi_ion_unscaled(l, l0, theta, A, T_e, T_i, n_e, Z, f_i,\
    v_i, j)
    return s/s.max(), a    