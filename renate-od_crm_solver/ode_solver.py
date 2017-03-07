import numpy


class ode_solver:
    def set_up_equation(variable_vector, coefficient_matrix):
        derivative_vector = numpy.dot(variable_vector, coefficient_matrix)
        return derivative_vector


