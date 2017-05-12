from crm_solver import inputs
from crm_solver import coefficientmatrix
from crm_solver.ode import Ode
import matplotlib.pyplot
import h5py
import os
import numpy


class Solve:
    def __init__(self):
        self.time_index = 0
        return

    @staticmethod
    def change_in_time():
        inp = inputs.Inputs()
        solution_4d = numpy.empty([inp.step_number, inp.number_of_levels])
        print(numpy.size(solution_4d))
        for time_index in range(len(inp.time_slices)):
            print('time' + str(time_index))
            inp_new = inputs.Inputs(time_index=time_index)
            solution_slice = Solve.solve_numerically(inputs=inp_new)
            solution_4d = numpy.dstack([solution_4d, solution_slice])
        return solution_4d

    def plot_populations(self):
        inp = inputs.Inputs()
        solutions = self.solve_numerically(inputs=inp)
        print(solutions)
        for level in range(inp.number_of_levels):
            matplotlib.pyplot.plot(inp.steps, solutions[:, level], label='level '+str(level))
            matplotlib.pyplot.yscale('log', nonposx='clip')
            matplotlib.pyplot.ylim((0, 1))
        matplotlib.pyplot.legend(loc='cene',bbox_to_anchor=(1, 0.5), ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()

    @staticmethod
    def save_populations():
        solutions_4d = Solve.change_in_time()
        print('size: ' + str(numpy.size(solutions_4d)))
        inp = inputs.Inputs()
        local_dir=os.getcwd()
        h5f = h5py.File(Solve.locate_h5_dir(local_dir) + 'solutions.h5', 'w')
        h5f.create_dataset('steps', data=inp.steps)
        h5f.create_dataset('solutions', data=solutions_4d)
        h5f.create_dataset('density', data=inp.density)
        h5f.create_dataset('electron_temperature', data=inp.electron_temperature)
        h5f.create_dataset('ion_temperature', data=inp.ion_temperature)
        h5f.close()

    @staticmethod
    def locate_h5_dir(cwd):
        rod_loc = (str.find(cwd, 'renate-od.git'))
        return cwd[0:rod_loc] + 'renate-od.git/trunk/data/'

    @staticmethod
    def solve_numerically(inputs):
        inp = inputs
        coeffmatrix = coefficientmatrix.CoefficientMatrix(inputs=inp)
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        numerical = ode_init.calculate_solution()
        return numerical

Solve.save_populations()