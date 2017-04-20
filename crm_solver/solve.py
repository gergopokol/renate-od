from crm_solver.inputs import Inputs, Inputs2
from crm_solver.rates import Rates
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
import matplotlib.pyplot

class Solve:
    @staticmethod
    def solve_numerically(inputs):
        inp = inputs
        r = Rates()
        coeffmatrix = CoefficientMatrix(inputs=inp)
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        numerical = ode_init.calculate_solution()
        return numerical

    def plot_populations(self):
        inp = Inputs()
        coeffmatrix = CoefficientMatrix()
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        solutions=self.solve_numerically()
        for i in range(inp.number_of_levels):
            matplotlib.pyplot.plot(inp.steps, solutions[:, i], label=str(i))
            matplotlib.pyplot.yscale('log', nonposx='clip')
            matplotlib.pyplot.ylim((0, 1))
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()

    def compare(self):
        inp = Inputs()
        inp2 = Inputs2()
        coeffmatrix = CoefficientMatrix(inp)
        ode_init = Ode(coefficient_matrix=coeffmatrix.matrix, initial_condition=inp.initial_condition, steps=inp.steps)
        solutions = self.solve_numerically(inputs=inp)
        solutions2 = self.solve_numerically(inputs=inp2)
        for i in range(inp2.number_of_levels):
            print(str((solutions[inp.step_number-1,i]-solutions2[inp2.step_number-1,i])/(solutions[inp.step_number-1,i])*100)+' %')
        for i in range(inp.number_of_levels):
            matplotlib.pyplot.plot(inp.steps, solutions[:, i], label='lev'+str(i)+' scen1')
            matplotlib.pyplot.yscale('log', nonposx='clip')
            matplotlib.pyplot.ylim((0, 1))
        for i in range(inp2.number_of_levels):
            matplotlib.pyplot.plot(inp2.steps, solutions2[:, i], linestyle='dashed', label='lev'+str(i)+' scen2')
            matplotlib.pyplot.yscale('log', nonposx='clip')
            matplotlib.pyplot.ylim((0, 1))
        matplotlib.pyplot.legend(loc='center', bbox_to_anchor=(1,0.5), ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()


Solve.compare(Solve)
