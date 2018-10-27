import unittest

import numpy

from crm_solver.ode import Ode


class OdeTest(unittest.TestCase):

    DECIMALS_2 = 2
    DECIMALS_6 = 6
    START_INTERVAL = 0
    END_INTERVAL = 0.1
    STEP_NUMBER = 100
    STEPS = numpy.linspace(START_INTERVAL, END_INTERVAL, STEP_NUMBER)
    START_POSITION = (STEPS[0] + STEPS[1]) / 2.
    INITIAL_CONDITION = numpy.array([1., 2.])
    COEFFICIENT_MATRIX = numpy.array([[-0.2, 0.],
                                      [0., -1.]])
    COEFFICIENT_MATRIX_CHANGING = numpy.tensordot(COEFFICIENT_MATRIX, STEPS, axes=0)

    INITIAL_CONDITION_1D = [numpy.array([0.]), numpy.array([1.])]
    COEFFICIENT_MATRIX_1D = [numpy.array([[2.]]), numpy.array([[2.]])]
    EXPECTED_SIZE_2 = 2

    def test_size_of_set_up_equation_constant(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.setup(variable_vector=self.INITIAL_CONDITION,
                           actual_position=self.START_POSITION,
                           coefficient_matrix=self.COEFFICIENT_MATRIX,
                           steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)

    def test_size_of_set_up_equation_changing(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX_CHANGING,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.setup(variable_vector=self.INITIAL_CONDITION,
                           actual_position=self.START_POSITION,
                           coefficient_matrix=self.COEFFICIENT_MATRIX_CHANGING,
                           steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)

    def test_size_of_solution(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.calculate_solution()
        self.assertEqual(actual.size, self.STEP_NUMBER * self.INITIAL_CONDITION.size)

    def test_1d_analytical(self):
        for init in self.INITIAL_CONDITION_1D:
            for coefficient in self.COEFFICIENT_MATRIX_1D:
                ode = Ode(coefficient_matrix=coefficient, initial_condition=init, steps=self.STEPS)
                actual = ode.analytical_solution()
                for index, variable in enumerate(self.STEPS):
                    expected = ode.formula_1d(init, coefficient, variable)
                    self.assertEqual(actual[index], expected)

    def test_diagonal_numerical(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.calculate_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                expected = ode.formula_1d(self.INITIAL_CONDITION[j], self.COEFFICIENT_MATRIX[j, j], self.STEPS[i])
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_diagonal_analytical(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.analytical_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                expected = ode.formula_1d(self.INITIAL_CONDITION[j], self.COEFFICIENT_MATRIX[j, j], self.STEPS[i])
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_numerical_to_analytical(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        numerical = ode.calculate_solution()
        analytical = ode.analytical_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                self.assertAlmostEqual(numerical[i, j], analytical[i, j], self.DECIMALS_6)

    def test_almost_equal(self):
        actual = 1.004
        expected = 1
        self.assertAlmostEqual(actual, expected, self.DECIMALS_2)
