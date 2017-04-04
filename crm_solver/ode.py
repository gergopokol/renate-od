import numpy
from scipy.integrate import odeint
#from set_up_coefficient_matrix import set_up_coefficient_matrix


class Ode:

    def set_up_equation(self, variable_vector, calculation_point, coefficient_matrix):
        if coefficient_matrix.ndim == 3:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix[:, :, calculation_point])
        elif coefficient_matrix.ndim == 2:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix)
        return derivative_vector


    def calculate_solution(self, equation, initial_condition, steps,\
                           coefficient_matrix):
        solution=odeint(equation,initial_condition,t=steps,args=(coefficient_matrix,))
        return solution

    def analytical_solution(self, initial_condition, steps, coefficient_matrix,):
        eigenvalues, eigenvectors = numpy.linalg.eig(coefficient_matrix)
        if initial_condition.size == 1:
            analytical_solution=numpy.zeros(steps.size)
            for i in range(steps.size):
                analytical_solution[i] = 1/eigenvectors*initial_condition*eigenvectors\
                                         * numpy.exp(eigenvalues * steps[i])
        else:
            analytical_solution=numpy.zeros((steps.size,initial_condition.size))
            for i in range(steps.size):
                analytical_solution[i,:] = numpy.dot(numpy.dot(numpy.linalg.inv(eigenvectors),initial_condition),eigenvectors)\
                                           * numpy.exp(eigenvalues * steps[i])
 #                analytical_solution[i, :] = numpy.dot(numpy.linalg.inv(eigenvectors),initial_condition,eigenvectors)#*numpy.exp(eigenvalues * steps[i])
        return analytical_solution







