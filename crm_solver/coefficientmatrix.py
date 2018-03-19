import numpy
from crm_solver.rates import Rates


class CoefficientMatrix:
    def __init__(self, beamlet_param, beamlet_profiles):
        self.beamlet_profiles = beamlet_profiles
        self.rates = Rates(beamlet_param, beamlet_profiles)
        self.number_of_steps = self.rates.number_of_steps
        self.number_of_levels = self.rates.number_of_levels
        self.matrix = self.assemble()

    def assemble(self):
        coefficient_matrix = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                for step in range(self.number_of_steps):
                    if to_level == from_level:
                        electron_terms = sum(self.rates.electron_neutral_collisions[from_level, :from_level, step]) + \
                                         sum(self.rates.electron_neutral_collisions[from_level, from_level + \
                                            1:self.number_of_levels, step]) + \
                                            self.rates.electron_loss_collisions[0, from_level, step]
                        ion_terms = sum(self.rates.proton_neutral_collisions[from_level, :from_level, step]) + \
                                    sum(self.rates.proton_neutral_collisions[from_level, from_level + \
                                        1:self.number_of_levels, step]) + \
                                        self.rates.electron_loss_collisions[1, from_level, step]
                        photon_terms = sum(self.rates.einstein_coeffs[:, from_level]) / self.rates.velocity
                        coefficient_matrix[from_level, from_level, step] = \
                            -self.beamlet_profiles['beamlet_density'][step] * electron_terms - \
                            self.beamlet_profiles['beamlet_density'][step] * ion_terms - \
                            photon_terms
                    else:
                        coefficient_matrix[from_level, to_level, step] = self.beamlet_profiles['beamlet_density'][step] \
                                                      * self.rates.electron_neutral_collisions[to_level, from_level, step] \
                                                      + self.beamlet_profiles['beamlet_density'][step] \
                                                      * self.rates.proton_neutral_collisions[to_level, from_level, step] \
                                                      + self.rates.einstein_coeffs[from_level, to_level] / self.rates.velocity
        return coefficient_matrix