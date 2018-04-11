import utility
import pandas
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
import matplotlib.pyplot


class Beamlet:
    def __init__(self, param=None, profiles=None, data_path="beamlet/test.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.read_beamlet_param(data_path)
        self.profiles = profiles
        if not isinstance(self.profiles, pandas.DataFrame):
            self.read_beamlet_profiles()
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
        ode = Ode(coefficient_matrix=self.coefficient_matrix.matrix, initial_condition=self.initial_condition,
                  steps=self.profiles['beamlet_grid'])
        numerical = ode.calculate_solution()
        for level in range(self.coefficient_matrix.number_of_levels):
            label = 'level ' + str(level)
            self.profiles[label] = numerical[:, level]
        return

    def write_beamlet_profiles(self):
        hdf5_path = self.param.getroot().find('body').find('beamlet_profiles').text
        try:
            self.profiles.to_hdf(path_or_buf="data/" + hdf5_path, key="profiles")
            print('Beamlet profile data written to file: ' + hdf5_path)
        except:
            print('Beamlet profile data could NOT be written to file: ' + hdf5_path)
            raise



