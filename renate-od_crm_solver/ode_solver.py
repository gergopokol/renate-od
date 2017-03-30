import numpy
from scipy.integrate import odeint
#from set_up_coefficient_matrix import set_up_coefficient_matrix



class ode_solver:
    def set_up_equation(variable_vector, calculation_point, coefficient_matrix):
        if coefficient_matrix.ndim==3:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix[:, :, calculation_point])
        elif coefficient_matrix.ndim==2:
            derivative_vector = numpy.dot(variable_vector, coefficient_matrix)
        return derivative_vector


    def calculate_solution(set_up_equation,initial_condition,steps,\
                           coefficient_matrix):
        solution=odeint(set_up_equation,initial_condition,t=steps,args=(coefficient_matrix,))
        return solution

    def analytical_solution(initial_condition,steps,coefficient_matrix,):
        w, v = numpy.linalg.eig(coefficient_matrix)
        eigenvectors = v
        eigenvalues = w
        if initial_condition.size == 1:
            analytical_solution=numpy.zeros(steps.size)
            for i in range(steps.size):
              analytical_solution[i] = 1/eigenvectors*initial_condition*eigenvectors * numpy.exp(eigenvalues * steps[i])
        else:
            analytical_solution=numpy.zeros((steps.size,initial_condition.size))
            for i in range(steps.size):
                analytical_solution[i,:] = numpy.dot(numpy.dot(numpy.linalg.inv(eigenvectors),initial_condition),eigenvectors)*numpy.exp(
            eigenvalues*steps[i])
 #                analytical_solution[i, :] = numpy.dot(numpy.linalg.inv(eigenvectors),initial_condition,eigenvectors)#*numpy.exp(eigenvalues * steps[i])
        return analytical_solution









