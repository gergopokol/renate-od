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

    def generate_rate(self, temperature, beamenergy):
        self.temperature = temperature
        self.beamenergy = beamenergy

        m_t = self.transition.target.mass
        w = np.sqrt(2*self.temperature*sc.eV/m_t)

        m_b = self.transition.projectile.mass
        v_b = np.sqrt(2*self.beamenergy*sc.eV/m_b)

        E_range = self.crossection.impact_energy
        v = np.sqrt(2*E_range*sc.eV/m_t)
        self.velocity = v
        kernel = v**2*self.crossection.function*(np.exp(-((v-v_b)/w)**2) -
                                                 np.exp(-((v+v_b)/w)**2))/(np.sqrt(np.pi)*w*v_b**2)
        self.kernel = kernel

        self.rate = np.trapz(self.kernel, self.velocity)
        if self.transition.name == 'de-ex':
            self.rate = self.crossection.atomic_dict['de-ex'](self)

        return self.rate
