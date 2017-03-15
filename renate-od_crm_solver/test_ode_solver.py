from unittest import TestCase
from ode_solver import ode_solver
import numpy


class TestOde_solver(TestCase):
    pop = numpy.zeros(2)
    pop[0] = 1
    c = numpy.zeros((2, 2))
    c[0, 1] = 2
    c[0, 0] = 1
    initial_condition = numpy.zeros(4)
    initial_condition[0] = 1
    interval = 100
    number_of_points = 5
    distances = numpy.linspace(0, interval, number_of_points)
    coefficient_matrix = numpy.zeros((4, 4))
    for i in range(4):
        coefficient_matrix[i, i] = 1
        coefficient_matrix[i, i - 1] = 0.7

    def test_set_up_equation(self):
        s = ode_solver.set_up_equation(variable_vector=self.initial_condition, calculation_point=self.distances,coefficient_matrix=self.coefficient_matrix)
        self.assertEqual(4, s.size)

    def test_calculate_solution(self):
        s = ode_solver.calculate_solution(ode_solver.set_up_equation,self.initial_condition,self.distances,self.coefficient_matrix)
        self.assertEqual(s.size,self.number_of_points*self.initial_condition.size)




