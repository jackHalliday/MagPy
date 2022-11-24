import numpy as np
import scipy.constants as spc

def Planck_energy(E_eV, T_eV):
    A = 2.*(spc.e**4)*(spc.c**-2)*(spc.h**-3)
    B = np.power(E_eV, 3.)
    C = np.exp(E_eV/T_eV) - 1.
    return A*B/C
    