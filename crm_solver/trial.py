import numpy
from crm_solver.ode import Ode
import math

class Trial:
    initial_condition = numpy.transpose(numpy.zeros(2))
    initial_condition[0] = 1
    initial_condition[1] = 6
    interval = 0.1
    number_of_steps = 100
    steps = numpy.linspace(0, interval, number_of_steps)
    coefficient_matrix = numpy.zeros((2, 2))
    coefficient_matrix[0, 0] = 2
    coefficient_matrix[0, 1] = 0
    coefficient_matrix[1, 0] = 0
    coefficient_matrix[1, 1] = 10
    onedim_matrix = numpy.zeros((1, 1))
    onedim_matrix[0, 0] = 2
    onedim_init = numpy.zeros(1)
    onedim_init[0] = 1
    a_p = (coefficient_matrix[0, 0] + coefficient_matrix[1, 1] + math.sqrt(
        coefficient_matrix[0, 0] ** 2 - 2 * coefficient_matrix[0, 0]
        * coefficient_matrix[1, 1] + coefficient_matrix[1, 1] ** 2 +
        coefficient_matrix[0, 1] * coefficient_matrix[1, 0])) / 2
    a_n=(coefficient_matrix[0, 0] + coefficient_matrix[1, 1] - math.sqrt(
        coefficient_matrix[0, 0] ** 2 - 2 * coefficient_matrix[0, 0]
        * coefficient_matrix[1, 1] + coefficient_matrix[1, 1] ** 2 +
        coefficient_matrix[0, 1] * coefficient_matrix[1, 0])) / 2
    print(coefficient_matrix)

    analytical=Ode.analytical_solution(initial_condition, steps, coefficient_matrix)

    analytical_solution_1d=Ode.analytical_solution(onedim_init, steps, onedim_matrix)
    print(analytical_solution_1d-onedim_init*numpy.exp(onedim_matrix*steps[4]))

    print(Ode.calculate_solution(Ode.set_up_equation, initial_condition, steps, coefficient_matrix))


    numerical=Ode.calculate_solution(Ode.set_up_equation, initial_condition, steps, coefficient_matrix)

    for i in range(steps.size):
        for j in range(initial_condition.size):
            be=initial_condition[j]*numpy.exp(coefficient_matrix[j,j]*steps[i])
            print('num-an='+str(numerical[i, j]-analytical[i, j]))
            print('num-be='+str(numerical[i,j]-be))
            print('an-be='+str(analytical[i,j]-be))
