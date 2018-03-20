import utility
import pandas
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
import matplotlib.pyplot


class Beamlet:
    def __init__(self, param="", profiles="", data_path="beamlet/test.xml"):
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
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path).data
        assert isinstance(self.profiles, pandas.DataFrame)
        print('Beamlet.profiles read from file: ' + hdf5_path)

    def solve_numerically(self):
        ode_init = Ode(coefficient_matrix=self.coefficient_matrix.matrix, initial_condition=self.initial_condition,
                       steps=self.profiles['beamlet_grid'])
        numerical = ode_init.calculate_solution()
        for level in range(self.coefficient_matrix.number_of_levels):
            label = 'level ' + str(level)
            self.profiles[label] = numerical[:, level]
        return numerical

    def plot_populations(self):
        solutions = self.solve_numerically()
        for level in range(self.coefficient_matrix.number_of_levels):
            matplotlib.pyplot.plot(self.profiles['beamlet_grid'], solutions[:, level], label='level '+str(level))
            matplotlib.pyplot.yscale('log', nonposy='clip')
            matplotlib.pyplot.ylim((1e-5, 1))
        matplotlib.pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5), ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()
