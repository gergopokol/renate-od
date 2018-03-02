from crm_solver.inputs import Constant_Plasma_Inputs
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
from observation.observation import Obs_1d
import matplotlib.pyplot
import h5py
import os
from pdb import set_trace as bp


class Solve:
    @staticmethod
    def solve_numerically(inputs):
        inp = inputs
        coeffmatrix = CoefficientMatrix()
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        numerical = ode_init.calculate_solution()
        return numerical

    def plot_populations(self):
        inp = Constant_Plasma_Inputs()
        solutions = self.solve_numerically(inputs=inp)
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
        inp = Constant_Plasma_Inputs()
        solutions = self.solve_numerically(inputs=inp)
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