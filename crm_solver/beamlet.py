import utility
from utility.getdata import GetData
import pandas
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode


class Beamlet:
    def __init__(self, param=None, profiles=None, data_path="beamlet/test0001.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.read_beamlet_param(data_path)
        self.profiles = profiles
        if not isinstance(self.profiles, pandas.DataFrame):
            self.read_beamlet_profiles()
        self.coefficient_matrix = None
        self.initial_condition = None
        if not isinstance(self.param.getroot().find('body').find('mass'), etree._Element):
            self.get_mass()

    def initialize_ode(self):
        self.coefficient_matrix = CoefficientMatrix(self.param, self.profiles)
        self.initial_condition = [1] + [0] * (self.coefficient_matrix.number_of_levels - 1)

    def read_beamlet_param(self, data_path):
        self.param = utility.getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('Beamlet.param read from file: ' + data_path)

    def read_beamlet_profiles(self):
        hdf5_path = self.param.getroot().find('body').find('beamlet_profiles').text
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['profiles']).data
        assert isinstance(self.profiles, pandas.DataFrame)
        print('Beamlet.profiles read from file: ' + hdf5_path)

    def solve_numerically(self):
        if self.coefficient_matrix is None or self.initial_condition is None:
            self.initialize_ode()
        ode = Ode(coefficient_matrix=self.coefficient_matrix.matrix, initial_condition=self.initial_condition,
                  steps=self.profiles['beamlet_grid'])
        numerical = ode.calculate_solution()
        for level in range(self.coefficient_matrix.number_of_levels):
            label = 'level ' + str(level)
            self.profiles[label] = numerical[:, level]
        return

    def get_mass(self):
        data_path_name = 'atomic_data/' + self.param.getroot().find('body').find('beamlet_species') + '/supplementary_data/default/' + \
                         self.param.getroot().find('body').find('beamlet_species') + '_m.txt'
        mass_str = GetData(data_path_name=data_path_name, data_format="array").data
        try:
            mass = float(mass_str)
        except ValueError:
            print('Unexpected data in file: ' + data_path_name + '(Expecting single float!)')
            raise ValueError
        new_element = etree.Element('mass')
        new_element.text = mass
        new_element.set('unit', 'kg')
        self.param.getroot().find('body').append(new_element)
        return
