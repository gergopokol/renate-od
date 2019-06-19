import utility
from utility.getdata import GetData
from utility.convert import calculate_velocity_from_energy
from utility.constants import Constants
import pandas
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
from crm_solver.atomic_db import AtomicDB


class Beamlet:
    def __init__(self, param=None, profiles=None, components=None, atomic_db=None,
                 solver='numerical', data_path="beamlet/testimp0001.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.read_beamlet_param(data_path)
        self.profiles = profiles
        self.components = components
        self.atomic_db = atomic_db
        if atomic_db is None:
            self.atomic_db = AtomicDB(param=self.param)
        if not (isinstance(self.components, pandas.DataFrame) and isinstance(self.profiles, pandas.DataFrame)):
            self.read_beamlet_profiles()
        if not isinstance(self.param.getroot().find('body').find('beamlet_mass'), etree._Element):
            self.get_mass()
        if not isinstance(self.param.getroot().find('body').find('beamlet_velocity'), etree._Element):
            self.get_velocity()
        self.const = Constants()
        self.coefficient_matrix = None
        self.initial_condition = None
        self.calculate_beamevolution(solver)

    def read_beamlet_param(self, data_path):
        self.param = utility.getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('Beamlet.param read from file: ' + data_path)

    def read_beamlet_profiles(self):
        hdf5_path = self.param.getroot().find('body').find('beamlet_source').text
        self.components = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['components']).data
        assert isinstance(self.components, pandas.DataFrame)
        print('Beamlet.imp_components read from file: ' + hdf5_path)
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['profiles']).data
        assert isinstance(self.profiles, pandas.DataFrame)
        print('Beamlet.imp_profiles read from file: ' + hdf5_path)

    def get_mass(self):
        data_path_name = 'atomic_data/' + self.param.getroot().find('body').find('beamlet_species').text + \
                         '/supplementary_data/default/' + \
                         self.param.getroot().find('body').find('beamlet_species').text + '_m.txt'
        mass_str = GetData(data_path_name=data_path_name, data_format="array").data
        try:
            mass = float(mass_str)
        except ValueError:
            print('Unexpected data in file: ' + data_path_name + '(Expecting single float!)')
            raise ValueError
        new_element = etree.Element('beamlet_mass')
        new_element.text = mass_str
        new_element.set('unit', 'kg')
        self.param.getroot().find('body').append(new_element)
        return

    def get_velocity(self):
        energy = self.param.getroot().find('body').find('beamlet_energy').text
        mass = self.param.getroot().find('body').find('beamlet_mass').text
        velocity = calculate_velocity_from_energy(energy, float(mass))
        new_element = etree.Element('beamlet_velocity')
        new_element.text = str(velocity)
        new_element.set('unit', 'm/s')
        self.param.getroot().find('body').append(new_element)
        return

    def initialize_ode(self):
        self.coefficient_matrix = CoefficientMatrix(self.param, self.profiles, self.components, self.atomic_db)
        self.initial_condition = [self.get_linear_density()] + [0] * (self.atomic_db.atomic_levels - 1)

    def get_linear_density(self):
        current = float(self.param.getroot().find('body').find('beamlet_current').text)
        velocity = float(self.param.getroot().find('body').find('beamlet_velocity').text)
        return current / (velocity * self.const.charge_electron)

    def solve_numerically(self):
        if self.coefficient_matrix is None or self.initial_condition is None:
            self.initialize_ode()

        ode = Ode(coeff_matrix=self.coefficient_matrix.matrix, init_condition=self.initial_condition)
        numerical = ode.calculate_numerical_solution(self.profiles['beamlet grid']['distance']['m'])

        for level in range(self.atomic_db.atomic_levels):
            label = 'level ' + self.atomic_db.inv_atomic_dict[level]
            self.profiles[label] = numerical[:, level]
        return

    def calculate_beamevolution(self, solver):
        assert isinstance(solver, str)
        if solver is 'numerical':
            self.solve_numerically()
        elif solver is 'analytical':
            # TODO: Implement analytical solver
            pass
        elif solver is 'disregard':
            print('Beam evolution not calculated.')
            return
        else:
            raise Exception('The numerical solver: ' + solver + ' is not supported. '
                            'Supported solvers are: numerical, analytical, disregard.')

    def was_beamevolution_performed(self):
        try:
            dummy = self.profiles['level ' + self.atomic_db.set_default_atomic_levels()[2]]
            return True
        except KeyError:
            return False

    def compute_linear_emission_density(self, to_level=None, from_level=None):
        if to_level is None or from_level is None:
            from_level, to_level, ground_level, transition_label = self.atomic_db.set_default_atomic_levels()
        if isinstance(to_level, str) and isinstance(from_level, str):
            if self.atomic_db.atomic_dict[to_level] >= self.atomic_db.atomic_dict[from_level]:
                raise Exception('Dude! Please stop screwing around. '
                                'Electrons spontaneously transit from higher to lower atomic states.')
        else:
            raise Exception('The expected input for atomic transitions are strings. '
                            'Bundled-n for H,D,T beam species ex:[1n, 2n, ... 6n]. '
                            'l-n resolved labels for Li ex: [2s, 2p, ... 4f] and Na ex: [3s, 3p, ... 5s]')
        if self.was_beamevolution_performed():
            transition_label = from_level + '-' + to_level
            self.profiles[transition_label] = \
                self.profiles['level '+from_level] * self.atomic_db.spontaneous_trans[from_level][to_level]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def compute_linear_density_attenuation(self):
        if self.was_beamevolution_performed():
            self.profiles['linear_density_attenuation'] = self.profiles['level ' + self.atomic_db.inv_atomic_dict[0]]
            for level in range(1, self.atomic_db.atomic_levels):
                self.profiles['linear_density_attenuation'] += self.profiles['level ' +
                                                                             self.atomic_db.inv_atomic_dict[level]]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def compute_relative_populations(self, reference_level=None):
        if self.was_beamevolution_performed():
            if reference_level is None:
                reference_level = self.atomic_db.set_default_atomic_levels()[2]
            assert isinstance(reference_level, str)
            for level in range(self.atomic_db.atomic_levels):
                self.profiles['rel.pop ' + self.atomic_db.inv_atomic_dict[level]] = \
                    self.profiles['level ' + self.atomic_db.inv_atomic_dict[level]] / \
                    self.profiles['level ' + reference_level]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')
