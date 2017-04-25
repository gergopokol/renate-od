import numpy
from crm_solver.inputs import Inputs
from crm_solver.rates import Rates


class CoefficientMatrix:
    def __init__(self):
        self.inputs = Inputs()
        self.rates = Rates(self.inputs)
        self.matrix = self.compile()

    def compile(self):
        coefficient_matrix = numpy.zeros((self.inputs.number_of_levels, self.inputs.number_of_levels, len(self.inputs.steps)))
        for i in range(self.inputs.number_of_levels):
            for j in range(self.inputs.number_of_levels):
                for k in range(len(self.inputs.steps)):
    def __init__(self, inputs=Inputs()):
        self.inputs = inputs
        self.rates = Rates(self.inputs)
        self.matrix = self.compile()

    def compile(self):
        coefficient_matrix = numpy.zeros((self.inputs.number_of_levels, self.inputs.number_of_levels, len(self.inputs.steps)))
        for i in range(self.inputs.number_of_levels):
            for j in range(self.inputs.number_of_levels):
                for k in range(len(self.inputs.steps)):
                    if j == i:
                        coefficient_matrix[i, i, k] = -self.inputs.density * (sum(self.rates.electron_neutral_collisions[i, :i, k])
                                                                         + sum(self.rates.electron_neutral_collisions
                                                                               [i, i + 1:self.inputs.number_of_levels, k])
                                                                         + self.rates.electron_loss_collisions[0, i, k]) \
                                                      - self.inputs.density * (sum(self.rates.proton_neutral_collisions[i, :i, k])
                                                                          + sum(self.rates.proton_neutral_collisions
                                                                                [i, i + 1:self.inputs.number_of_levels, k])
                                                                          + self.rates.electron_loss_collisions[1, i, k])\
                                                      - sum(self.rates.einstein_coeffs[:, i]) / self.rates.velocity
                    else:
                        coefficient_matrix[i, j, k] = self.inputs.density * self.rates.electron_neutral_collisions[j, i, k] \
                                                      + self.inputs.density * self.rates.proton_neutral_collisions[j, i, k] \
                                                      + self.rates.einstein_coeffs[i, j] / self.rates.velocity
        return coefficient_matrix

