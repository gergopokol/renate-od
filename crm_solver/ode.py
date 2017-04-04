import numpy
from scipy.integrate import odeint


class Ode:
    def __init__(self,
                 coefficient_matrix=numpy.array([0.]),
                 initial_condition=numpy.array([0.]),
                 steps=numpy.array([0.])):
        self.coefficient_matrix = coefficient_matrix
        self.initial_condition = initial_condition
        self.steps = steps

    def calculate_solution(self):
        solution = odeint(self.set_up_equation, self.initial_condition, t=self.steps, args=(self.coefficient_matrix,))
        return solution

    def analytical_solution(self):
        eigenvalues, eigenvectors = numpy.linalg.eig(self.coefficient_matrix)
        if self.initial_condition.size == 1:
            analytical_solution = numpy.zeros(self.steps.size)
            for i in range(self.steps.size):
                analytical_solution[i] = 1 / eigenvectors * self.initial_condition * eigenvectors \
                                         * numpy.exp(eigenvalues * self.steps[i])
        else:
            analytical_solution = numpy.zeros((self.steps.size, self.initial_condition.size))
            for i in range(self.steps.size):
                analytical_solution[i, :] = numpy.dot(numpy.dot(numpy.linalg.inv(eigenvectors), self.initial_condition),
                                                      eigenvectors) * numpy.exp(eigenvalues * self.steps[i])
        return analytical_solution

    @staticmethod
    def set_up_equation(variable_vector, calculation_point, coefficient_matrix):
        if coefficient_matrix.ndim == 3:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix[:, :, calculation_point])
        elif coefficient_matrix.ndim == 2:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix)
        return derivative_vector
