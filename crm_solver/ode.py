import numpy
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class Ode:
    def __init__(self,
                 coefficient_matrix=numpy.array([0.]),
                 initial_condition=numpy.array([0.]),
                 steps=numpy.array([0.])):
        self.coefficient_matrix = coefficient_matrix
        self.initial_condition = initial_condition
        self.steps = steps

    def calculate_solution(self):
        return odeint(func=self.setup, y0=self.initial_condition, t=self.steps,
                          args=(self.coefficient_matrix, self.steps))

    def analytical_solution(self):
        eigenvalues, eigenvectors = numpy.linalg.eig(self.coefficient_matrix)
        if self.initial_condition.size == 1:
            analytical_solution = numpy.zeros(self.steps.size)
            for step in range(self.steps.size):
                analytical_solution[step] = 1 / eigenvectors * self.initial_condition * eigenvectors \
                                         * numpy.exp(eigenvalues * self.steps[step])
        else:
            analytical_solution = numpy.zeros((self.steps.size, self.initial_condition.size))
            for step in range(self.steps.size):
                analytical_solution[step, :] = numpy.dot(numpy.dot(numpy.linalg.inv(eigenvectors),
                                                                   self.initial_condition), eigenvectors)\
                                               * numpy.exp(eigenvalues * self.steps[step])
        return analytical_solution

    @staticmethod
    def setup(variable_vector, actual_position, coefficient_matrix, steps):
        if coefficient_matrix.ndim == 3:
            interp_coefficient_matrix = interp1d(steps, coefficient_matrix, axis=2, fill_value='extrapolate')
            derivative_vector = numpy.dot(variable_vector, interp_coefficient_matrix(actual_position))
        elif coefficient_matrix.ndim == 2:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix)
        return derivative_vector

    @staticmethod
    def formula_1d(init, coefficient, variable):
        return init * numpy.exp(coefficient * variable)
