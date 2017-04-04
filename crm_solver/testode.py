from unittest import TestCase
from crm_solver.ode import Ode
import numpy


class TestOde(TestCase):
    pop = numpy.array([1., 0.])
    c = numpy.array([[1., 2.],
                     [0., 0.]])
    test_initial_condition = numpy.array([1., 2.])
    test_coefficient_matrix = numpy.array([[-0.2, 0.],
                                           [0., -1.]])
    interval = 0.1
    number_of_steps = 100
    steps = numpy.linspace(0, interval, number_of_steps)
    onedim_matrix = numpy.array([[2.]])
    onedim_init = numpy.array([1.])

    def test_set_up_equation(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix, initial_condition=self.test_initial_condition,
                  steps=self.steps)
        s = ode.set_up_equation(variable_vector=self.test_initial_condition, calculation_point=self.steps,
                                coefficient_matrix=self.test_coefficient_matrix)
        self.assertEqual(2, s.size)

    def test_size_of_solution(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix, initial_condition=self.test_initial_condition,
                  steps=self.steps)
        s = ode.calculate_solution()
        self.assertEqual(s.size, self.number_of_steps * self.test_initial_condition.size)

    def test_1d_analytical(self):
        ode = Ode(coefficient_matrix=self.onedim_matrix, initial_condition=self.onedim_init,
                  steps=self.steps)
        solution = ode.analytical_solution()
        for i in range(self.number_of_steps):
            self.assertEqual(solution[i], self.onedim_init * numpy.exp(self.onedim_matrix * self.steps[i]))

    def test_diagonal_numerical(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix, initial_condition=self.test_initial_condition,
                  steps=self.steps)
        solution = ode.calculate_solution()
        for i in range(self.number_of_steps):
            for j in range(self.test_initial_condition.size):
                self.assertAlmostEqual(solution[i, j], (self.test_initial_condition[j]
                                                        * numpy.exp(self.test_coefficient_matrix[j, j]
                                                                    * self.steps[i])), 2)

    def test_diagonal_analytical(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix, initial_condition=self.test_initial_condition,
                  steps=self.steps)
        solution = ode.analytical_solution()
        for i in range(self.number_of_steps):
            for j in range(self.test_initial_condition.size):
                self.assertAlmostEqual(solution[i, j], (self.test_initial_condition[j]
                                                        * numpy.exp(self.test_coefficient_matrix[j, j]
                                                                    * self.steps[i])), 100)

    def test_numerical_to_analytical(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix, initial_condition=self.test_initial_condition,
                  steps=self.steps)
        numerical = ode.calculate_solution()
        analytical = ode.analytical_solution()
        for i in range(self.number_of_steps):
            for j in range(self.test_initial_condition.size):
                self.assertAlmostEqual(numerical[i, j], analytical[i, j], 3)

    def test_almostequal(self):
        self.assertAlmostEqual(1, 1.004, 2)
