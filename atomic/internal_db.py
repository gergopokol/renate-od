from utility import Particle
from utility import Transition
from atomic import CrossSection, RateCoeff
from scipy.interpolate import interp1d
from lxml import etree
from utility import getdata
import utility.convert as uc
import numpy as np


class InternalDB():

    def __init__(self, param, cross_section_source, data_path=None):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.param = getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        self.__projectile_parameters()
        self.temperature_axis = 10**(np.linspace(0, 5, 400))
        self.cross_section = CrossSection(source=cross_section_source, projectile=self.projectile_type)
        self.spontaneous_trans = self.cross_section.spontaneous_trans

    def __projectile_parameters(self):
        self.energy = self.param.getroot().find('body').find('beamlet_energy').text
        self.projectile = self.param.getroot().find('body').find('beamlet_species').text
        if self.projectile in ['H', 'D', 'T']:
            self.projectile_type = 'hydrogenic'
            if self.projectile == 'H':
                self.projectile_particle = Particle(label='H', mass_number=1, atomic_number=1)
            elif self.projectile == 'D':
                self.projectile_particle = Particle(label='D', mass_number=2, atomic_number=1)
            elif self.projectile == 'T':
                self.projectile_particle = Particle(label='T', mass_number=3, atomic_number=1)
            self.atomic_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, None: None}
            self.atomic_levels = 6
            self.inv_atomic_dict = {index: name for name, index in self.atomic_dict.items()}
            self.mass = self.projectile_particle.mass

    def _get_projectile_velocity(self):
        self.velocity = uc.calculate_velocity_from_energy(uc.convert_keV_to_eV(float(self.energy)), self.mass)
        return self.velocity

    def set_default_atomic_levels(self):
        if self.projectile in ['H', 'D', 'T']:
            return '3', '2', '1', '3n-->2n'

    def get_rate_interpolator(self, reaction_type, target, from_level, to_level=None):
        #print(reaction_type, target, from_level, to_level)
        if reaction_type == 'excitation' and isinstance(to_level, int):
            if from_level == to_level:
                return interp1d(self.temperature_axis, np.zeros(self.temperature_axis.shape), fill_value='extrapolate')
            elif from_level < to_level:
                trans_type = 'ex'
            elif from_level > to_level:
                trans_type = 'de-ex'
        elif reaction_type == 'electron_loss':
            trans_type = 'eloss'
        target_particle = Particle(label='Z', mass_number=target['A'], atomic_number=target['Z'], charge=target['q'])
        trans = Transition(projectile=self.projectile_particle, target=target_particle,
                           from_level=self.inv_atomic_dict[from_level],
                           to_level=self.inv_atomic_dict[to_level], trans=trans_type)
        rate_generator = RateCoeff(trans, self.cross_section)
        rates = rate_generator.generate_rates(self.temperature_axis, float(self.energy)*1000)
        return interp1d(self.temperature_axis, uc.convert_from_cm2_to_m2(rates), fill_value='extrapolate')
