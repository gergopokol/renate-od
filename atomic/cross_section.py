from atomic.tools import Transition
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from atomic.aladdin import AladdinData
from atomic.hydrogenic_cross_sections import HydrogenicData

"""
class CrossSection(AladdinData):
    def __init__(self, source, projectile=None):
        if not isinstance(source, str): TypeError('A string is expected to decide which data source to pursue.')
        if source is 'aladdin':
            AladdinData.__init__(self)
        elif source is 'internal':
            pass
        else: ValueError('The data source requested is not supported: ' + source)

    def get_cross_section(self, transition, energy_grid):
        pass
"""


class CrossSection:

    CROSSEC_SOURCES = {'aladdin': [], 'internal': ['hydrogenic']}

    def __new__(cls, source, projectile=None):
        if not isinstance(source, str):
            raise TypeError('A string is expected to decide which data source to pursue.\n'+CrossSection.GetAvailableData())
        if source == 'aladdin':
            return AladdinData()
        elif source == 'internal':
            if not isinstance(projectile, str):
                raise TypeError('The data source requires the projectile parameter to be a valid string.\n'+CrossSection.GetAvailableData())
            if projectile == 'hydrogenic':
                return HydrogenicData()
            else:
                raise ValueError('The given projectile is not available in the data source.\n'+CrossSection.GetAvailableData())
        else:
            raise ValueError('The data source requested is not supported.\n'+CrossSection.GetAvailableData())

    @staticmethod
    def GetAvailableData():
        return('Available sources and projectiles:\n'+str(CrossSection.CROSSEC_SOURCES))


class RateCoeff:
    def __init__(self, transition, crossection):
        self.transition = transition
        self.crossection = crossection

    def generate_rates(self, T_range, beamenergy):
        m_t = self.transition.target.mass
        m_b = self.transition.projectile.mass
        v_b = np.sqrt(2*beamenergy*sc.eV/m_b)
        rates = np.zeros(len(T_range))
        for i, T in enumerate(T_range):
            w = np.sqrt(2*T*sc.eV/m_t)
            E_range = self._get_energy_range(T, v_b, w)
            v = np.sqrt(2*E_range*sc.eV/m_t)
            crossfunc = self.crossection.get_cross_section(self.transition, E_range)
            kernel = v**2*crossfunc*(np.exp(-((v-v_b)/w)**2)
                                     - np.exp(-((v+v_b)/w)**2))/(np.sqrt(np.pi)*w*v_b**2)
            rates[i] = np.trapz(kernel, v)
            if self.transition.name == 'de-ex':
                rates[i] = self.crossection.get_deex_rate(rates[i], T, self.transition)
        return rates

    def _get_velocity_and_kernel(self, T, beamenergy, E_range=None):
        m_t = self.transition.target.mass
        m_b = self.transition.projectile.mass
        v_b = np.sqrt(2*beamenergy*sc.eV/m_b)
        w = np.sqrt(2*T*sc.eV/m_t)
        if isinstance(E_range, type(None)):
            E_range = self._get_energy_range(T, v_b, w)
        v = np.sqrt(2*E_range*sc.eV/m_t)
        crossfunc = self.crossection.get_cross_section(self.transition, E_range)
        kernel = v**2*crossfunc*(np.exp(-((v-v_b)/w)**2)
                                 - np.exp(-((v+v_b)/w)**2))/(np.sqrt(np.pi)*w*v_b**2)
        return v, kernel

    def _get_energy_range(self, T, vb, w, minE=13.6, N=1000, k=3):
        v_min = max(vb-k*w, 0)
        v_max = vb+k*w
        roi_start = max(0.5*self.transition.target.mass*v_min**2/sc.eV, minE)
        roi_max = 0.5*self.transition.target.mass*v_max**2/sc.eV
        E_range = np.linspace(roi_start, roi_max, N)
        return E_range
