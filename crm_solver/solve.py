"""
#from crm_solver.coefficientmatrix import CoefficientMatrix
#from crm_solver.ode import Ode
#from observation.observation import Obs_1d
#import matplotlib.pyplot
#import h5py
#import os
"""

import utility
import pandas
from lxml import etree


class Solve:
    def __init__(self, param="", profiles="", data_path="beamlet/test.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.read_beamlet_param(data_path)
        self.profiles = profiles
        if not isinstance(self.profiles, pandas.DataFrame):
            self.read_beamlet_profiles()

    def read_beamlet_param(self, data_path):
        self.param = utility.getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('Beamlet.param read from file: ' + data_path)

    def read_beamlet_profiles(self):
        hdf5_path = self.param.getroot().find('body').find('beamlet_profiles').text
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path).data
        assert isinstance(self.profiles, pandas.DataFrame)
        print('Beamlet.profiles read from file: ' + hdf5_path)



"""
    @staticmethod
    def solve_numerically(beamlet_param):
        inp = beamlet_param
        coeffmatrix = CoefficientMatrix()
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        numerical = ode_init.calculate_solution()
        return numerical

    def plot_populations(self):
        #inp = beamlet_param
        solutions = self.solve_numerically(beamlet_param=inp)
        for level in range(inp.number_of_levels):
            matplotlib.pyplot.plot(inp.steps, solutions[:, level], label='level '+str(level))
            matplotlib.pyplot.yscale('log', nonposy='clip')
            matplotlib.pyplot.ylim((0, 1))
        matplotlib.pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5), ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()

        obs = Obs_1d(inp)
        photon_current = Obs_1d.calculate_light_profile(obs,solutions)
        matplotlib.pyplot.scatter(inp.observation_profile,photon_current, label='photon_current')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.xlabel('Distance along beam [m]')
        matplotlib.pyplot.ylabel('Photon current [1/s]')
        matplotlib.pyplot.show()

    def save_populations(self):
        #inp = Constant_Plasma_Inputs()
        solutions = self.solve_numerically(beamlet_param=inp)
        local_dir = os.getcwd()
        h5f = h5py.File(self.locate_h5_dir(local_dir) + 'solutions.h5', 'w')
        h5f.create_dataset('steps', data=inp.steps)
        h5f.create_dataset('solutions', data=solutions)
        h5f.create_dataset('density', data=inp.density)
        h5f.create_dataset('electron_temperature', data=inp.electron_temperature)
        h5f.create_dataset('ion_temperature', data=inp.ion_temperature)
        h5f.close()
        print('hdf5 file created.')

    @staticmethod
    def locate_h5_dir(cwd):
        rod_loc = (str.find(cwd, 'renate-od.git'))
        return cwd[0:rod_loc] + 'renate-od.git/trunk/data/'


Solve.plot_populations(Solve)
"""