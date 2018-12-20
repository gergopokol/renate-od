import numpy
from crm_solver.rates import Rates


class CoefficientMatrix:
    def __init__(self, beamlet_param, beamlet_profiles, plasma_components):
        self.beamlet_profiles = beamlet_profiles
        self.rates = Rates(beamlet_param, beamlet_profiles, plasma_components)
        self.number_of_steps = self.rates.number_of_steps
        self.number_of_levels = self.rates.number_of_levels
        # Initialize matrices
        self.matrix = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        self.electron_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        self.ion_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        imp_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        self.imp_terms = numpy.concatenate([[imp_terms]*self.rates.number_of_impurities])
        self.photon_terms = numpy.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_steps))
        # Fill matrices
        self.assemble_terms()
        self.apply_density()

    def assemble_terms(self):
        for from_level in range(self.number_of_levels):
            for to_level in range(self.number_of_levels):
                for step in range(self.number_of_steps):
                    if to_level == from_level:
                        self.electron_terms[from_level, to_level, step] = \
                            - sum(self.rates.electron_neutral_collisions[from_level, :to_level, step]) \
                            - sum(self.rates.electron_neutral_collisions[from_level, (to_level+1):self.number_of_levels,
                                  step]) \
                            - self.rates.electron_loss_collisions[from_level, step]
                        self.ion_terms[from_level, to_level, step] = \
                            - sum(self.rates.ion_neutral_collisions[0][from_level, :to_level, step]) \
                            - sum(self.rates.ion_neutral_collisions[0][from_level, (to_level+1):self.number_of_levels,
                                  step]) \
                            - self.rates.electron_loss_ion_collisions[0][from_level, step]
                        for imp in range(self.rates.number_of_impurities):
                            self.imp_terms[imp][from_level, to_level, step] = \
                                - sum(self.rates.imp_neutral_collisions[imp][from_level, :to_level, step]) \
                                - sum(self.rates.imp_neutral_collisions[imp][from_level,
                                      (to_level+1):self.number_of_levels, step]) \
                                - self.rates.electron_loss_imp_collisions[imp][from_level, step]
                        self.photon_terms[from_level, to_level, step] = \
                            - sum(self.rates.einstein_coeffs[:, from_level]) / self.rates.velocity
                    else:
                        self.electron_terms[from_level, to_level, step] = \
                            self.rates.electron_neutral_collisions[from_level, to_level, step]
                        self.ion_terms[from_level, to_level, step] = \
                            self.rates.ion_neutral_collisions[0][from_level, to_level, step]
                        self.photon_terms[from_level, to_level, step] = \
                            self.rates.einstein_coeffs[to_level, from_level] / self.rates.velocity

    def apply_density(self):
        for step in range(self.number_of_steps):
            self.matrix[:, :, step] = self.beamlet_profiles['electron']['density'][step] * self.electron_terms[:, :, step] \
                                      + self.beamlet_profiles['ion1']['density'][step] * self.ion_terms[:, :, step] \
                                      + self.beamlet_profiles['imp1']['density'][step] * self.imp_terms[0][:, :, step] \
                                      + self.beamlet_profiles['imp2']['density'][step] * self.imp_terms[1][:, :, step] \
                                      + self.photon_terms[:, :, step]

