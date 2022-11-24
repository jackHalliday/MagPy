import numpy as np
from scipy.interpolate import interp1d
import re
import pkg_resources

SFF_PATH = pkg_resources.resource_filename('MagPy.Radiation',\
 'scattering_factors/')

class Henke:
    hc=1.240e-4 #eVcm
    r0 = 2.818e-13 #cm
    def __init__(self, element, numberDensity_cm):
        path = SFF_PATH + element + '.nff'
        dataTable = np.genfromtxt(path, skip_header=1)
        En, f1, f2 = dataTable.transpose()
        self.f2 = interp1d(En, f2)
        self.n = numberDensity_cm
    def kappa(self, energy_eV):
        mua = self.f2(energy_eV) * 2. * self.r0 * self.hc / energy_eV
        return mua*self.n
    def tau(self, energy_eV, thickness_um):
        thickness_cm = thickness_um*1e-4
        return self.kappa(energy_eV)*thickness_cm
    def transmission(self, energy_eV, thickness_um):
        tau = self.tau(energy_eV, thickness_um)
        return np.exp(-tau)
    def paCrossSection(self, energy_eV):
        mua = self.f2(energy_eV) * 2. * self.r0 * self.hc / energy_eV
        return mua
    def mu_a(self, energy_eV):
        mua = self.f2(energy_eV) * 2. * self.r0 * self.hc / energy_eV
        return mua
        
class CompositeHenke:
    def __init__(self, chemicalFormula, numberDensity_cm):
        elements = re.findall('[A-Z][^A-Z]*', chemicalFormula)
        els = []
        frs = []
        for element in elements:
            splitted =  re.split('(\d+)', element)
            if(len(splitted)>1):
                el = splitted[0] 
                fr = float(splitted[1])
            else:
                el = splitted[0]
                fr = 1.
            els.append(el)
            frs.append(fr)
        frs = np.array(frs)
        frs /= frs.sum()
        self.components = []
        for el, fr in zip(els, frs):
            component = Henke(el, numberDensity_cm*fr)
            self.components.append(component)   
    def tau(self, energy_eV, thickness_um):
        tau = np.zeros_like(energy_eV)
        for component in self.components:
            tau += component.tau(energy_eV, thickness_um)
        return tau      
    def transmission(self, energy_eV, thickness_um):
        tau = self.tau(energy_eV, thickness_um)
        return np.exp(-tau)
        
'''class Element:
    def __init__(self, symbol):
        data = np.genfromtxt('periodic_table.dat', delimiter=',', \
        skip_header=1, dtype=str)
        row = np.where(data[:,2]==symbol)[0][0]
        self.atomic_data = row
    def atomic_mass(self):
        A = row[3]
        return '''