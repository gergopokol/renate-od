import numpy
from lxml import etree
import pandas
from utility import convert


class CoefficientMatrix:
    def __init__(self, beamlet_param, beamlet_profiles, plasma_components, atomic_db):
        assert isinstance(beamlet_param, etree._ElementTree)
        assert isinstance(beamlet_profiles, pandas.DataFrame)
        self.beamlet_profiles = beamlet_profiles

        # Initialize interpolation matrices
        self.electron_impact_trans_np = numpy.zeros((atomic_db.atomic_levels, atomic_db.atomic_levels,
                                                     self.beamlet_profiles['beamlet grid'].size))
        self.electron_impact_loss_np = numpy.zeros((atomic_db.atomic_levels,
                                                    self.beamlet_profiles['beamlet grid'].size))
        self.ion_impact_trans_np = numpy.concatenate([[self.electron_impact_trans_np] * len(
            [comp for comp in plasma_components['Z'] if comp > 0])])
        self.ion_impact_loss_np = numpy.concatenate([[self.electron_impact_loss_np] * len(
            [comp for comp in plasma_components['Z'] if comp > 0])])

        # Initialize assembly matrices
        self.matrix = numpy.zeros(
            (atomic_db.atomic_levels, atomic_db.atomic_levels, self.beamlet_profiles['beamlet grid'].size))
        self.electron_terms = numpy.zeros(
            (atomic_db.atomic_levels, atomic_db.atomic_levels, self.beamlet_profiles['beamlet grid'].size))
        ion_terms = numpy.zeros(
            (atomic_db.atomic_levels, atomic_db.atomic_levels, self.beamlet_profiles['beamlet grid'].size))
        self.ion_terms = numpy.concatenate([[ion_terms] * len([comp for comp in plasma_components['Z'] if comp > 0])])
        self.photon_terms = numpy.zeros(
            (atomic_db.atomic_levels, atomic_db.atomic_levels, self.beamlet_profiles['beamlet grid'].size))
        self.interpolate_rates(atomic_db, plasma_components)
        self.assemble_matrix(atomic_db, plasma_components)

    def interpolate_rates(self, atomic_db, plasma_components):
        for from_level in range(atomic_db.atomic_levels):
            self.interpolate_electron_impact_loss(from_level, atomic_db)
            for to_level in range(atomic_db.atomic_levels):
                if to_level != from_level:
                    self.interpolate_electron_impact_trans(from_level, to_level, atomic_db)
        for ion in range(len([comp for comp in plasma_components['Z'] if comp > 0])):
            for from_level in range(atomic_db.atomic_levels):
                self.interpolate_ion_impact_loss(ion, from_level, atomic_db)
                for to_level in range(atomic_db.atomic_levels):
                    if to_level != from_level:
                        self.interpolate_ion_impact_trans(ion, from_level, to_level, atomic_db)

    def assemble_matrix(self, atomic_db, plasma_components):
        for from_level in range(atomic_db.atomic_levels):
            for to_level in range(atomic_db.atomic_levels):
                if to_level == from_level:
                    self.assemble_electron_impact_population_loss_terms(from_level, to_level, atomic_db)
                else:
                    self.assemble_electron_impact_population_gain_terms(from_level, to_level)
        for step in range(self.beamlet_profiles['beamlet grid'].size):
            self.apply_electron_density(step)
        for ion in range(len([comp for comp in plasma_components['Z'] if comp > 0])):
            for from_level in range(atomic_db.atomic_levels):
                for to_level in range(atomic_db.atomic_levels):
                    if to_level == from_level:
                        self.assemble_ion_impact_population_loss_terms(ion, from_level, to_level, atomic_db)
                    else:
                        self.assemble_ion_impact_population_gain_terms(ion, from_level, to_level)
            for step in range(self.beamlet_profiles['beamlet grid'].size):
                self.apply_ion_density(ion, step)
        for from_level in range(atomic_db.atomic_levels):
            for to_level in range(atomic_db.atomic_levels):
                if to_level == from_level:
                    self.assemble_spontaneous_population_loss_terms(from_level, to_level, atomic_db)
                else:
                    self.assemble_spontaneous_population_gain_terms(from_level, to_level, atomic_db)
        for step in range(self.beamlet_profiles['beamlet grid'].size):
            self.apply_photons(step)

    def interpolate_electron_impact_trans(self, from_level, to_level, atomic_db):
        self.electron_impact_trans_np[from_level, to_level, :] \
            = atomic_db.electron_impact_trans[from_level][to_level](
                self.beamlet_profiles['electron']['temperature']['eV'][:])

    def interpolate_ion_impact_trans(self, ion, from_level, to_level, atomic_db):
        self.ion_impact_trans_np[ion, from_level, to_level, :] = \
            atomic_db.ion_impact_trans[from_level][to_level][ion](
                self.beamlet_profiles['ion'+str(ion+1)]['temperature']['eV'][:])

    def interpolate_electron_impact_loss(self, from_level, atomic_db):
        self.electron_impact_loss_np[from_level, :] = \
            atomic_db.electron_impact_loss[from_level](self.beamlet_profiles['electron']['temperature']['eV'][:])

    def interpolate_ion_impact_loss(self, ion, from_level, atomic_db):
        self.ion_impact_loss_np[ion, from_level, :] = \
            atomic_db.ion_impact_loss[from_level][ion](
                self.beamlet_profiles['ion' + str(ion + 1)]['temperature']['eV'][:])

    def assemble_electron_impact_population_loss_terms(self, from_level, to_level, atomic_db):
        self.electron_terms[from_level, to_level, :] = \
            - numpy.sum(self.electron_impact_trans_np[from_level, :to_level, :], axis=0) \
            - numpy.sum(self.electron_impact_trans_np[from_level, (to_level + 1):atomic_db.atomic_levels, :], axis=0) \
            - self.electron_impact_loss_np[from_level, :]

    def assemble_ion_impact_population_loss_terms(self, ion, from_level, to_level, atomic_db):
        self.ion_terms[ion, from_level, to_level, :] = \
            - numpy.sum(self.ion_impact_trans_np[ion, from_level, :to_level, :], axis=0) \
            - numpy.sum(self.ion_impact_trans_np[ion, from_level, (to_level + 1):atomic_db.atomic_levels, :], axis=0) \
            - self.ion_impact_loss_np[ion, from_level, :]

    def assemble_electron_impact_population_gain_terms(self, from_level, to_level):
        self.electron_terms[from_level, to_level, :] = \
            self.electron_impact_trans_np[from_level, to_level, :]

    def assemble_ion_impact_population_gain_terms(self, ion, from_level, to_level):
        self.ion_terms[ion, from_level, to_level, :] = \
            self.ion_impact_trans_np[ion, from_level, to_level, :]

    def assemble_spontaneous_population_loss_terms(self, from_level, to_level, atomic_db):
        self.photon_terms[from_level, to_level, :] = \
            - numpy.sum(atomic_db.spontaneous_trans[:, from_level]) / atomic_db.velocity
        
    def assemble_spontaneous_population_gain_terms(self, from_level, to_level, atomic_db):
        self.photon_terms[from_level, to_level, :] = \
            atomic_db.spontaneous_trans[to_level, from_level] / atomic_db.velocity
        
    def apply_electron_density(self, step):
        self.matrix[:, :, step] = self.beamlet_profiles['electron']['density']['m-3'][step] \
                                  * convert.convert_from_cm2_to_m2(self.electron_terms[:, :, step])
        
    def apply_ion_density(self, ion, step):
        self.matrix[:, :, step] = self.matrix[:, :, step] + \
                                  self.beamlet_profiles['ion' + str(ion + 1)]['density']['m-3'][step] \
                                  * convert.convert_from_cm2_to_m2(self.ion_terms[ion, :, :, step])
        
    def apply_photons(self, step):
        self.matrix[:, :, step] = self.matrix[:, :, step] + self.photon_terms[:, :, step]
