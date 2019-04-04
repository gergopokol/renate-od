import utility
from utility.getdata import GetData
from utility.convert import calculate_velocity_from_energy
from utility.constants import Constants
import pandas
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode


class Beamlet:
    def __init__(self, param=None, profiles=None, components=None,
                 solver='numerical', data_path="beamlet/testimp0001.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.read_beamlet_param(data_path)
        self.profiles = profiles
        self.components = components
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
        self.coefficient_matrix = CoefficientMatrix(self.param, self.profiles, self.components)
        self.initial_condition = [self.get_linear_density()] + [0] * (self.coefficient_matrix.number_of_levels - 1)

    def get_linear_density(self):
        current = float(self.param.getroot().find('body').find('beamlet_current').text)
        velocity = float(self.param.getroot().find('body').find('beamlet_velocity').text)
        return current / (velocity * self.const.charge_electron)

    def solve_numerically(self):
        if self.coefficient_matrix is None or self.initial_condition is None:
            self.initialize_ode()

        ode = Ode(coeff_matrix=self.coefficient_matrix.matrix, init_condition=self.initial_condition)
        numerical = ode.calculate_numerical_solution(self.profiles['beamlet grid']['distance']['m'])

        for level in range(self.coefficient_matrix.number_of_levels):
            label = 'level ' + str(level)
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

    def get_beamlet_emission(self):
        atom = self.param.getroot().find('body').find('beamlet_species').text
        if self.beamevolution_performed():
            emission = self.profiles[self.observed_level(atom)]

    def beamevolution_performed(self):
        try:
            dummy = self.profiles['level 0']
            return True
        except KeyError:
            return False

    @staticmethod
    def observed_level(atom):
        if atom is 'H' or atom is 'D' or atom is 'T':
            return 'level 2'
        elif atom is 'Li' or atom is 'Na':
            return 'level 1'
        else:
            raise ValueError('The atomic species: ' + ' is not supported')

    def get_beamlet_attenuation(self):
        # TODO: Calculate the total attenuation of the beamlet
        pass

    def get_relative_electron_populations(self):
        # TODO: Calculate the relative electron distribution on atomic levels
        pass
