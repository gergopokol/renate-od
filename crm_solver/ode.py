import numpy
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class Ode:
    def __init__(self, coeff_matrix, init_condition):
        self.coeff_matrix = coeff_matrix
        self.init_condition = init_condition

    def calculate_numerical_solution(self, steps):
        return odeint(func=self.setup_derivative_vector, y0=self.init_condition, t=steps,
                      args=(self.coeff_matrix, steps))

    def calculate_analytical_solution(self, steps):
        eigenvalues, eigenvectors = numpy.linalg.eig(self.coeff_matrix)
        if self.init_condition.size == 1:
            return self.__calculate_analytical_solution_1d(steps, eigenvalues)
        else:
            return self.__calculate_analytical_solution_else(steps, eigenvalues, eigenvectors)

    def __calculate_analytical_solution_else(self, steps, eigenvalues, eigenvectors):
        analytical_solution = numpy.zeros((steps.size, self.init_condition.size))
        temporal_constant = numpy.dot(numpy.dot(numpy.linalg.inv(eigenvectors), self.init_condition), eigenvectors)
        for step in range(steps.size):
            analytical_solution[step, :] = temporal_constant * numpy.exp(eigenvalues * steps[step])
        return analytical_solution

    def __calculate_analytical_solution_1d(self, steps, eigenvalues):
        analytical_solution = numpy.zeros(steps.size)
        for step in range(steps.size):
            analytical_solution[step] = self.init_condition * numpy.exp(eigenvalues * steps[step])
        return analytical_solution

    @staticmethod
    def setup_derivative_vector(variable_vector, actual_position, coefficient_matrix, steps):
        if coefficient_matrix.ndim == 3:
            interp_coefficient_matrix = interp1d(steps, coefficient_matrix, axis=2, fill_value='extrapolate')
            return numpy.dot(variable_vector, interp_coefficient_matrix(actual_position))
        elif coefficient_matrix.ndim == 2:
            return numpy.dot(variable_vector, coefficient_matrix)
        else:
            raise ValueError(
                'Rate Coefficient Matrix of dimensions: ' + str(coefficient_matrix.ndim) + ' is not supported')

    @staticmethod
    def calculate_1d_solution(init, coefficient, variable):
        return init * numpy.exp(coefficient * variable)
