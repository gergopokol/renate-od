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
    EXPECTED_SIZE_100 = 100
    EXPECTED_SIZE_200 = 200
    EXPECTED_DERIVATIVE_VECTOR_1 = numpy.array([-0.2, -2.])
    EXPECTED_DERIVATIVE_VECTOR_2 = numpy.array([-0.000101, -0.001010])

    def test_setup_derivative_vector_with_coefficient_matrix(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.setup_derivative_vector(variable_vector=self.INITIAL_CONDITION,
                                             actual_position=self.START_POSITION,
                                             coefficient_matrix=self.COEFFICIENT_MATRIX,
                                             steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)
        self.assertAlmostEqual(actual[0], self.EXPECTED_DERIVATIVE_VECTOR_1[0], self.DECIMALS_6)
        self.assertAlmostEqual(actual[1], self.EXPECTED_DERIVATIVE_VECTOR_1[1], self.DECIMALS_6)

    def test_setup_derivative_vector_with_coefficient_matrix_changing(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX_CHANGING,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.setup_derivative_vector(variable_vector=self.INITIAL_CONDITION,
                                             actual_position=self.START_POSITION,
                                             coefficient_matrix=self.COEFFICIENT_MATRIX_CHANGING,
                                             steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)
        self.assertAlmostEqual(actual[0], self.EXPECTED_DERIVATIVE_VECTOR_2[0], self.DECIMALS_6)
        self.assertAlmostEqual(actual[1], self.EXPECTED_DERIVATIVE_VECTOR_2[1], self.DECIMALS_6)

    def test_calculate_numerical_solution(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.calculate_numerical_solution()
        self.assertEqual(actual.size, self.STEP_NUMBER * self.INITIAL_CONDITION.size)
        # TODO test values in actual

    def test_calculate_analytical_solution_1d(self):
        for init in self.INITIAL_CONDITION_1D:
            for coefficient in self.COEFFICIENT_MATRIX_1D:
                ode = Ode(coefficient_matrix=coefficient, initial_condition=init, steps=self.STEPS)
                actual = ode.calculate_analytical_solution()
                self.assertEqual(actual.size, self.EXPECTED_SIZE_100)
                for index, variable in enumerate(self.STEPS):
                    expected = ode.formula_1d(init, coefficient, variable)
                    self.assertEqual(actual[index], expected)

    def test_calculate_numerical_solution_diagonal(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.calculate_numerical_solution()
        self.assertEqual(actual.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                expected = ode.formula_1d(self.INITIAL_CONDITION[j], self.COEFFICIENT_MATRIX[j, j], self.STEPS[i])
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_calculate_analytical_solution_diagonal(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        actual = ode.calculate_analytical_solution()
        self.assertEqual(actual.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                expected = ode.formula_1d(self.INITIAL_CONDITION[j], self.COEFFICIENT_MATRIX[j, j], self.STEPS[i])
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_two_solutions_with_each_other(self):
        ode = Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                  initial_condition=self.INITIAL_CONDITION,
                  steps=self.STEPS)
        numerical = ode.calculate_numerical_solution()
        analytical = ode.calculate_analytical_solution()
        self.assertEqual(numerical.size, self.EXPECTED_SIZE_100)
        self.assertEqual(analytical.size, self.EXPECTED_SIZE_100)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                self.assertAlmostEqual(numerical[i, j], analytical[i, j], self.DECIMALS_6)

    def test_almost_equal(self):
        actual = 1.004
        expected = 1
        self.assertAlmostEqual(actual, expected, self.DECIMALS_2)
