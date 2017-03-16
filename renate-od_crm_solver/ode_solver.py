import numpy
from scipy.integrate import odeint
#from set_up_coefficient_matrix import set_up_coefficient_matrix



class ode_solver:
    def set_up_equation(variable_vector, calculation_point, coefficient_matrix):
        derivative_vector = numpy.dot(variable_vector, coefficient_matrix)
        return derivative_vector

    def calculate_solution(set_up_equation,initial_condition,steps,\
                           coefficient_matrix):
        solution=odeint(set_up_equation,initial_condition,t=steps,args=(coefficient_matrix,))
        return solution







