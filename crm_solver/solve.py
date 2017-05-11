from crm_solver.inputs import Inputs
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
import matplotlib.pyplot
import h5py
import os
import numpy


class Solve:
    @staticmethod
    def solve_numerically(inputs):
        inp = inputs
        coeffmatrix = CoefficientMatrix()
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        numerical = ode_init.calculate_solution()
        return numerical

    def change_in_time(self, inp):
        solution_4d = numpy.empty([])
        numpy.size(solution_4d)
        solution_4d=solution_4d[:,numpy.newaxis]
        for time_index in range(len(inp.time_slices)):
            solution_slice=self.solve_numerically(inputs=Inputs(time_index=time_index))
            solution_4d = numpy.append(solution_4d, solution_slice, axis=2)
        return solution_4d

    def plot_populations(self):
        inp = Inputs()
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

    def save_populations(self):
        inp = Inputs()
        solutions_4d = self.change_in_time(Solve, inp=inp)
        print('size: ' + str(numpy.size(solutions_4d)))
        local_dir=os.getcwd()
        h5f = h5py.File(self.locate_h5_dir(local_dir) + 'solutions.h5', 'w')
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


Solve.save_populations(Solve)