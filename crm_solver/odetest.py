import unittest
import numpy

import crm_solver.ode


class OdeTest(unittest.TestCase):

    DECIMALS_2 = 2
    DECIMALS_6 = 6
    STEP_INTERVAL = 0.1
    STEP_NUMBER = 100
    STEPS = numpy.linspace(0, STEP_INTERVAL, STEP_NUMBER)

    INITIAL_CONDITION = numpy.array([1., 2.])
    COEFFICIENT_MATRIX = numpy.array([[-0.2, 0.],
                                      [0., -1.]])
    COEFFICIENT_MATRIX_CHANGING = numpy.tensordot(COEFFICIENT_MATRIX, STEPS, axes=0)

    INITIAL_CONDITION_1D = [numpy.array([0.]), ]
    COEFFICIENT_MATRIX_1D = [numpy.array([[2.]]), ]
    INITIAL_CONDITION_1D.append(numpy.array([1.]))
    COEFFICIENT_MATRIX_1D.append(numpy.array([[2.]]))
    EXPECTED_SIZE_2 = 2

    def test_size_of_set_up_equation_constant(self):
        ode = crm_solver.ode.Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                                 initial_condition=self.INITIAL_CONDITION,
                                 steps=self.STEPS)
        actual = ode.set_up_equation(variable_vector=self.INITIAL_CONDITION,
                                     actual_position=(self.STEPS[0]+self.STEPS[1])/2.,
                                     coefficient_matrix=self.COEFFICIENT_MATRIX,
                                     steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)

    def test_size_of_set_up_equation_changing(self):
        ode = crm_solver.ode.Ode(coefficient_matrix=self.COEFFICIENT_MATRIX_CHANGING,
                                 initial_condition=self.INITIAL_CONDITION,
                                 steps=self.STEPS)
        actual = ode.set_up_equation(variable_vector=self.INITIAL_CONDITION,
                                     actual_position=(self.STEPS[0]+self.STEPS[1])/2.,
                                     coefficient_matrix=self.COEFFICIENT_MATRIX_CHANGING,
                                     steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)

    def test_size_of_solution(self):
        ode = crm_solver.ode.Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                                 initial_condition=self.INITIAL_CONDITION,
                                 steps=self.STEPS)
        s = ode.calculate_solution()
        self.assertEqual(s.size, self.STEP_NUMBER * self.INITIAL_CONDITION.size)

    def test_1d_analytical(self):
        for init in self.INITIAL_CONDITION_1D:
            for coefficient in self.COEFFICIENT_MATRIX_1D:
                ode = crm_solver.ode.Ode(coefficient_matrix=coefficient, initial_condition=init, steps=self.STEPS)
                solution = ode.analytical_solution()
                for index, variable in enumerate(self.STEPS):
                    self.assertEqual(solution[index], self.formula_1d(init, coefficient, variable))

    def test_diagonal_numerical(self):
        ode = crm_solver.ode.Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                                 initial_condition=self.INITIAL_CONDITION,
                                 steps=self.STEPS)
        solution = ode.calculate_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                self.assertAlmostEqual(solution[i, j], self.formula_1d(self.INITIAL_CONDITION[j],
                                                                       self.COEFFICIENT_MATRIX[j, j],
                                                                       self.STEPS[i]), self.DECIMALS_6)

    def test_diagonal_analytical(self):
        ode = crm_solver.ode.Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                                 initial_condition=self.INITIAL_CONDITION,
                                 steps=self.STEPS)
        solution = ode.analytical_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                self.assertAlmostEqual(solution[i, j], self.formula_1d(self.INITIAL_CONDITION[j],
                                                                       self.COEFFICIENT_MATRIX[j, j],
                                                                       self.STEPS[i]), self.DECIMALS_6)

    def test_numerical_to_analytical(self):
        ode = crm_solver.ode.Ode(coefficient_matrix=self.COEFFICIENT_MATRIX,
                                 initial_condition=self.INITIAL_CONDITION,
                                 steps=self.STEPS)
        numerical = ode.calculate_solution()
        analytical = ode.analytical_solution()
        for i in range(self.STEP_NUMBER):
            for j in range(self.INITIAL_CONDITION.size):
                self.assertAlmostEqual(numerical[i, j], analytical[i, j], self.DECIMALS_6)

    def test_almost_equal(self):
        self.assertAlmostEqual(1.004, 1, self.DECIMALS_2)

    @staticmethod
    def formula_1d(init, coefficient, variable):
        return init * numpy.exp(coefficient * variable)

