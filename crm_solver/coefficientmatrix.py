import numpy
from crm_solver.rates import Rates


class CoefficientMatrix:
    def __init__(self, beamlet_param, beamlet_profiles):
        self.beamlet_profiles = beamlet_profiles
        self.rates = Rates(beamlet_param, beamlet_profiles)
        self.number_of_steps = self.rates.number_of_steps
        self.number_of_levels = self.rates.number_of_levels
        # Initialize matrixes
        self.matrix = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        self.electron_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        self.ion_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        self.photon_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        # Fill matrixes
        self.assemble_terms()
        self.apply_density()

    def assemble_terms(self):
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                for step in range(self.number_of_steps):
                    if to_level == from_level:
                        self.electron_terms[from_level, to_level, step] = \
                            - sum(self.rates.electron_neutral_collisions[from_level, :from_level, step]) \
                            - sum(self.rates.electron_neutral_collisions[from_level, from_level + 1:self.number_of_levels, step]) \
                            - self.rates.electron_loss_collisions[0, from_level, step]
                        self.ion_terms[from_level, to_level, step] = \
                            - sum(self.rates.proton_neutral_collisions[from_level, :from_level, step]) \
                            - sum(self.rates.proton_neutral_collisions[from_level, from_level + 1:self.number_of_levels, step]) \
                            - self.rates.electron_loss_collisions[1, from_level, step]
                        self.photon_terms[from_level, to_level, step] = \
                            - sum(self.rates.einstein_coeffs[:, from_level]) / self.rates.velocity
                    else:
                        self.electron_terms[from_level, to_level, step] = \
                            self.rates.electron_neutral_collisions[to_level, from_level, step]
                        self.ion_terms[from_level, to_level, step] = \
                            self.rates.proton_neutral_collisions[to_level, from_level, step]
                        self.photon_terms[from_level, to_level, step] = \
                            self.rates.einstein_coeffs[from_level, to_level] / self.rates.velocity

    def apply_density(self):
        for step in range(self.number_of_steps):
            self.matrix[:, :, step] = self.beamlet_profiles['beamlet_density'][step] * self.electron_terms[:, :, step] \
                                      + self.beamlet_profiles['beamlet_density'][step] * self.ion_terms[:, :, step] \
                                      + self.photon_terms[:, :, step]
