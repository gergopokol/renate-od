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
    STEPS_VARYING_NONDIAGONAL = numpy.array([0, 0.1, 0.2, 0.3])
    START_POSITION = (STEPS[0] + STEPS[1]) / 2.

    INIT_CONDITION = numpy.array([1., 2.])
    INIT_CONDITION_CONSTANT_NONDIAGONAL = numpy.array([3, -11, 11])
    INIT_CONDITION_VARYING_NONDIAGONAL = numpy.array([2, -1])
    INIT_CONDITION_1D = [numpy.array([0.]), numpy.array([1.])]

    COEFF_MATRIX_DIM_ERROR = numpy.array([15, 12])
    COEFF_MATRIX_CONSTANT_DIAGONAL = numpy.array([[-0.2, 0.],
                                                  [0., -1.]])
    COEFF_MATRIX_CONSTANT_NONDIAGONAL = numpy.array([[1, 3, -4],
                                                     [-1, 1, -2],
                                                     [-1, -3, 1]])
    COEFF_MATRIX_CHANGING = numpy.tensordot(COEFF_MATRIX_CONSTANT_DIAGONAL, STEPS, axes=0)
    COEFF_MATRIX_1D = [numpy.array([[2.]]), numpy.array([[2.]])]
    COEFF_MATRIX_VARYING_NONDIAGONAL = numpy.zeros((2, 2, STEPS_VARYING_NONDIAGONAL.size))
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 0] = numpy.array([[2, 3], [-1, -2]])
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 1] = numpy.array([[2, 1], [2, -2]])
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 2] = numpy.array([[2, -1], [5, -2]])
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 3] = numpy.array([[2, -3], [8, -2]])

    EXPECTED_RESULT_CONSTANT_NONDIAGONAL = numpy.array([3.38999231881, -14.9942067896, 13.5292786755])
    EXPECTED_RESULT_VARYING_NONDIAGONAL = numpy.array([[2, -1],
                                                       [2.5108421, -0.20367017],
                                                       [3.0526073, 0.0849155],
                                                       [3.3920525, -1.1205913]])
    EXPECTED_SIZE_2 = 2
    EXPECTED_SIZE_100 = 100
    EXPECTED_SIZE_200 = 200
    EXPECTED_DERIVATIVE_VECTOR_1 = numpy.array([-0.2, -2.])
    EXPECTED_DERIVATIVE_VECTOR_2 = numpy.array([-0.000101, -0.001010])
    ACCEPTED_TYPES = [numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                      numpy.float, numpy.float16, numpy.float32, numpy.float64]

    @classmethod
    def setUpClass(cls):
        # TODO add class level variables or constants
        pass

    def setUp(self):
        # TODO add local variables or constants that should initialize before all functions
        pass

    def tearDown(self):
        # TODO if need to destruct a class level variable
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO if need to destruct a local variable after its run
        pass

    def test_setup_derivative_vector_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION)
        actual = ode.set_derivative_vector(variable_vector=self.INIT_CONDITION,
                                           actual_position=self.START_POSITION,
                                           coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL,
                                           steps=self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)
        for index in range(actual.size):
            self.assertIn(type(actual[index]), self.ACCEPTED_TYPES)
            self.assertAlmostEqual(actual[index], self.EXPECTED_DERIVATIVE_VECTOR_1[index], self.DECIMALS_6)

    def test_setup_derivative_vector_for_constant_nondiagonal_case(self):
        pass

    def test_setup_derivative_vector_with_coefficient_matrix_changing(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CHANGING, init_condition=self.INIT_CONDITION)
        actual = ode.set_derivative_vector(variable_vector=self.INIT_CONDITION,
                                           actual_position=self.START_POSITION,
                                           coeff_matrix=self.COEFF_MATRIX_CHANGING,
                                           steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)
        self.assertAlmostEqual(actual[0], self.EXPECTED_DERIVATIVE_VECTOR_2[0], self.DECIMALS_6)
        self.assertAlmostEqual(actual[1], self.EXPECTED_DERIVATIVE_VECTOR_2[1], self.DECIMALS_6)

    def test_setup_derivative_vector_for_dimension_error(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_DIM_ERROR, init_condition=self.INIT_CONDITION)
        with self.assertRaises(ValueError):
            ode.set_derivative_vector(variable_vector=self.INIT_CONDITION,
                                      actual_position=self.START_POSITION,
                                      coeff_matrix=self.COEFF_MATRIX_DIM_ERROR,
                                      steps=self.STEPS)

    def test_calculate_numerical_solution_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION)
        actual = ode.calculate_numerical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INIT_CONDITION.size):
                expected = ode.calculate_1d_solution(self.INIT_CONDITION[j],
                                                     self.COEFF_MATRIX_CONSTANT_DIAGONAL[j, j],
                                                     self.STEPS[i])
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_calculate_numerical_solution_for_constant_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_CONSTANT_NONDIAGONAL)
        actual = ode.calculate_numerical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.STEP_NUMBER * self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size)
        self.assertEqual(actual.shape, (self.STEP_NUMBER, self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size))
        for index in range(self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size):
            self.assertIn(type(actual[-1, index]), self.ACCEPTED_TYPES)
            self.assertAlmostEqual(actual[-1, index], self.EXPECTED_RESULT_CONSTANT_NONDIAGONAL[index], self.DECIMALS_6)

    def test_calculate_numerical_solution_for_varying_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_VARYING_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_VARYING_NONDIAGONAL)
        actual = ode.calculate_numerical_solution(self.STEPS_VARYING_NONDIAGONAL)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size,
                         self.STEPS_VARYING_NONDIAGONAL.size * self.INIT_CONDITION_VARYING_NONDIAGONAL.size)
        self.assertEqual(actual.shape,
                         (self.STEPS_VARYING_NONDIAGONAL.size, self.INIT_CONDITION_VARYING_NONDIAGONAL.size))
        for i in range(self.STEPS_VARYING_NONDIAGONAL.size):
            for j in range(self.INIT_CONDITION_VARYING_NONDIAGONAL.size):
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)
                # TODO missing assert. Test currently fails. Should write it

    def test_calculate_analytical_solution_1d(self):
        for init in self.INIT_CONDITION_1D:
            for coefficient in self.COEFF_MATRIX_1D:
                ode = Ode(coeff_matrix=coefficient, init_condition=init)
                actual = ode.calculate_analytical_solution(self.STEPS)
                self.assertEqual(type(actual), numpy.ndarray)
                self.assertEqual(actual.size, self.EXPECTED_SIZE_100)
                for index, variable in enumerate(self.STEPS):
                    expected = ode.calculate_1d_solution(init, coefficient, variable)
                    self.assertIn(type(actual[index]), self.ACCEPTED_TYPES)
                    self.assertEqual(actual[index], expected)

    def test_calculate_analytical_solution_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION)
        actual = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INIT_CONDITION.size):
                expected = ode.calculate_1d_solution(self.INIT_CONDITION[j],
                                                     self.COEFF_MATRIX_CONSTANT_DIAGONAL[j, j],
                                                     self.STEPS[i])
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_calculate_analytical_solution_for_constant_nondiagnoal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_CONSTANT_NONDIAGONAL)
        actual = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.STEP_NUMBER * self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size)
        self.assertEqual(actual.shape, (self.STEP_NUMBER, self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size))
        for index in range(self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size):
            self.assertIn(type(actual[-1, index]), self.ACCEPTED_TYPES)
            # TODO missing assert. Test currently fails. Should write it

    def test_calculate_analytical_solution_for_varying_nondiagonal_case(self):
        # TODO relevant function is missing, should write it
        pass

    def test_benchmark_solvers_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION)
        integrate = ode.calculate_numerical_solution(self.STEPS)
        analytical = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(integrate.size, self.EXPECTED_SIZE_200)
        self.assertEqual(analytical.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INIT_CONDITION.size):
                self.assertAlmostEqual(integrate[i, j], analytical[i, j], self.DECIMALS_6)

    def test_benchmark_solvers_for_constant_nondiagonal_case(self):
        # TODO relevant function is missing, should write it
        pass

    def test_benchmark_solvers_for_varying_nondiagonal_case(self):
        # TODO relevant function is missing, should write it
        pass

    def test_almost_equal(self):
        actual = 1.004
        expected = 1
        self.assertAlmostEqual(actual, expected, self.DECIMALS_2)
