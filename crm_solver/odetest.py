import unittest
import numpy

from crm_solver.ode import Ode


class OdeTest(unittest.TestCase):

    DECIMALS_2 = 2
    DECIMALS_6 = 6
    STEP_INTERVAL = 0.1
    STEP_NUMBER = 100
    steps = numpy.linspace(0, STEP_INTERVAL, STEP_NUMBER)

    test_initial_condition = numpy.array([1., 2.])
    test_coefficient_matrix_constant = numpy.array([[-0.2, 0.],
                                                    [0., -1.]])
    test_coefficient_matrix_changing = numpy.tensordot(test_coefficient_matrix_constant, steps, axes=0)

    test_initial_condition_1d = [numpy.array([0.]), ]
    test_coefficient_matrix_1d = [numpy.array([[2.]]), ]
    test_initial_condition_1d.append(numpy.array([1.]))
    test_coefficient_matrix_1d.append(numpy.array([[2.]]))
    EXPECTED_SIZE_2 = 2

    def test_size_of_set_up_equation_constant(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix_constant,
                  initial_condition=self.test_initial_condition,
                  steps=self.steps)
        actual = ode.set_up_equation(variable_vector=self.test_initial_condition,
                                     actual_position=(self.steps[0]+self.steps[1])/2.,
                                     coefficient_matrix=self.test_coefficient_matrix_constant,
                                     steps=self.steps)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)

    def test_size_of_set_up_equation_changing(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix_changing,
                  initial_condition=self.test_initial_condition,
                  steps=self.steps)
        actual = ode.set_up_equation(variable_vector=self.test_initial_condition,
                                     actual_position=(self.steps[0]+self.steps[1])/2.,
                                     coefficient_matrix=self.test_coefficient_matrix_changing,
                                     steps=self.steps)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)

    def test_size_of_solution(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix_constant,
                  initial_condition=self.test_initial_condition,
                  steps=self.steps)
        s = ode.calculate_solution()
        self.assertEqual(s.size, self.STEP_NUMBER * self.test_initial_condition.size)

    def test_1d_analytical(self):
        for init in self.test_initial_condition_1d:
            for coefficient in self.test_coefficient_matrix_1d:
                ode = Ode(coefficient_matrix=coefficient, initial_condition=init, steps=self.steps)
                solution = ode.analytical_solution()
                for index, variable in enumerate(self.steps):
                    self.assertEqual(solution[index], self.formula_1d(init, coefficient, variable))

    def test_diagonal_numerical(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix_constant,
                  initial_condition=self.test_initial_condition,
                  steps=self.steps)
        solution = ode.calculate_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.test_initial_condition.size):
                self.assertAlmostEqual(solution[i, j], self.formula_1d(self.test_initial_condition[j],
                                                                       self.test_coefficient_matrix_constant[j, j],
                                                                       self.steps[i]), self.DECIMALS_6)

    def test_diagonal_analytical(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix_constant,
                  initial_condition=self.test_initial_condition,
                  steps=self.steps)
        solution = ode.analytical_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.test_initial_condition.size):
                self.assertAlmostEqual(solution[i, j], self.formula_1d(self.test_initial_condition[j],
                                                                       self.test_coefficient_matrix_constant[j, j],
                                                                       self.steps[i]), self.DECIMALS_6)

    def test_numerical_to_analytical(self):
        ode = Ode(coefficient_matrix=self.test_coefficient_matrix_constant,
                  initial_condition=self.test_initial_condition,
                  steps=self.steps)
        numerical = ode.calculate_solution()
        analytical = ode.analytical_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.test_initial_condition.size):
                self.assertAlmostEqual(numerical[i, j], analytical[i, j], self.DECIMALS_6)

    def test_almost_equal(self):
        self.assertAlmostEqual(1.004, 1, self.DECIMALS_2)

    @staticmethod
    def formula_1d(init, coefficient, variable):
        return init * numpy.exp(coefficient * variable)

