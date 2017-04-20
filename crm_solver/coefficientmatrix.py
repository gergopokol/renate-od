import numpy

class CoefficientMatrix:
    @staticmethod
    def compile(initial_condition, steps, density, rate_coefficients, velocity):
        number_of_levels=len(initial_condition)
        coefficient_matrix = numpy.zeros((number_of_levels, number_of_levels, len(steps)))
        for i in range(number_of_levels):
            for j in range(number_of_levels):
                for k in range(len(steps)):
                    if j == i:
                        coefficient_matrix[i, i, k] = -density[k] * (sum(rate_coefficients[electron_neutral_collisions][i, :i, k])
                                                                     + sum(rate_coefficients[electron_neutral_collisions][i, i + 1:number_of_levels, k])
                                                                     + rate_coefficients[electron_loss_collisions][0, i, k])\
                        - density[k] * (sum(rate_coefficients[proton_neutral_collisions][i, :i, k])
                                        + sum(rate_coefficients[proton_neutral_collisions][i, i + 1:number_of_levels, k])
                                        + rate_coefficients[electron_loss_collisions][1, i, k])
                                        - sum(rate_coefficients[einstein_coeffs][:, i]) / velocity
                    else:
                        coefficient_matrix[i, j, k] = density[k] * rate_coefficients[electron_neutral_collisions][j, i, k] \
                                     + density[k] * rate_coefficients[proton_neutral_collisions][j, i, k] \
                                     + rate_coefficients[einstein_coeffs][i, j] / velocity