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
        analytical_solution = numpy.linalg.inv(eigenvectors) * initial_condition * eigenvectors * numpy.exp(
            eigenvalues * steps[0])
        return analytical_solution









