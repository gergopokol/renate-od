from unittest import TestCase
from ode_solver import ode_solver
import numpy


class TestOde_solver(TestCase):
    pop = numpy.zeros(2)
    pop[0] = 1
    c = numpy.zeros((2, 2))
    c[0, 1] = 2
    c[0, 0] = 1

    def test_set_up_equation(self):
        s = ode_solver.set_up_equation(variable_vector=self.pop, coefficient_matrix=self.c)
        self.assertEqual(2, s.size)
