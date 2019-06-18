import numpy
from lxml import etree
import pandas
from utility import convert
#from scipy.interpolate import interp1d


class CoefficientMatrix:
    def __init__(self, beamlet_param, beamlet_profiles, plasma_components, atomic_db):
        assert isinstance(beamlet_param, etree._ElementTree)
        self.mass = float(beamlet_param.getroot().find('body').find('beamlet_mass').text)
        self.velocity = float(beamlet_param.getroot().find('body').find('beamlet_velocity').text)
        assert isinstance(beamlet_profiles, pandas.DataFrame)
        self.beamlet_profiles = beamlet_profiles
        # Initialize interpolation matrices
        self.electron_neutral_collisions = numpy.zeros((atomic_db.atomic_levels, atomic_db.atomic_levels,
                                                        self.beamlet_profiles['beamlet grid'].size))
        self.electron_loss_collisions = numpy.zeros((atomic_db.atomic_levels,
                                                     self.beamlet_profiles['beamlet grid'].size))
        self.ion_neutral_collisions = numpy.concatenate([[self.electron_neutral_collisions] * len(
            [comp for comp in plasma_components['Z'] if comp > 0])])
        self.electron_loss_ion_collisions = numpy.concatenate([[self.electron_loss_collisions] * len(
            [comp for comp in plasma_components['Z'] if comp > 0])])
        self.einstein_coeffs = atomic_db.spontaneous_trans
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
        self.assemble_matrix(atomic_db, plasma_components)

    def assemble_matrix(self, atomic_db, plasma_components):
        for from_level in range(atomic_db.atomic_levels):
            self.interpolate_electron_impact_loss(from_level, atomic_db)
            for to_level in range(atomic_db.atomic_levels):
                if to_level == from_level:
                    self.assemble_electron_impact_population_loss_terms(from_level, to_level, atomic_db)
                else:
                    self.interpolate_electron_impact_trans(from_level, to_level, atomic_db)
                    self.assemble_electron_impact_population_gain_terms(from_level, to_level)
                for step in range(self.beamlet_profiles['beamlet grid'].size):
                    self.apply_electron_density(step)
        for ion in range(len([comp for comp in plasma_components['Z'] if comp > 0])):
            for from_level in range(atomic_db.atomic_levels):
                self.interpolate_ion_impact_loss(ion, from_level, atomic_db, plasma_components)
                for to_level in range(atomic_db.atomic_levels):
                    if to_level == from_level:
                        self.assemble_ion_impact_population_loss_terms(ion, from_level, to_level, atomic_db)
                    else:
                        self.interpolate_ion_impact_trans(ion, from_level, to_level, atomic_db, plasma_components)
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
        self.electron_neutral_collisions[from_level, to_level, :] \
            = atomic_db.electron_impact_trans[atomic_db.inv_atomic_dict[from_level]][atomic_db.inv_atomic_dict[to_level]](
            self.beamlet_profiles['electron']['temperature']['eV'][:])
        self.electron_neutral_collisions = convert.convert_from_cm2_to_m2(self.electron_neutral_collisions)

    def interpolate_ion_impact_trans(self, ion, from_level, to_level, atomic_db, plasma_components):
        self.ion_neutral_collisions[ion][from_level, to_level, :] = \
            atomic_db.ion_impact_trans[atomic_db.inv_atomic_dict[from_level]][atomic_db.inv_atomic_dict[to_level]][
                atomic_db.charged_states[plasma_components['q']['ion'+str(ion+1)]-1]](
                self.beamlet_profiles['ion'+str(ion+1)]['temperature']['eV'][:])
        self.ion_neutral_collisions = convert.convert_from_cm2_to_m2(self.ion_neutral_collisions)

    def interpolate_electron_impact_loss(self, from_level, atomic_db):
        self.electron_loss_collisions[from_level, :] = \
            atomic_db.electron_impact_loss[atomic_db.inv_atomic_dict[from_level]]['electron'](self.beamlet_profiles['electron']['temperature']['eV'][:])
        self.electron_loss_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_collisions)

    def interpolate_ion_impact_loss(self, ion, from_level, atomic_db, plasma_components):
        self.electron_loss_ion_collisions[ion][from_level, :] = \
            atomic_db.ion_impact_loss[atomic_db.inv_atomic_dict[from_level]][atomic_db.charged_states[
                plasma_components['q']['ion'+str(ion+1)]-1]](self.beamlet_profiles['ion' + str(ion + 1)]['temperature']['eV'][:])
        self.electron_loss_ion_collisions = convert.convert_from_cm2_to_m2(self.electron_loss_ion_collisions)

    def assemble_electron_impact_population_loss_terms(self, from_level, to_level, atomic_db):
        self.electron_terms[from_level, to_level, :] = \
            - sum(self.electron_neutral_collisions[from_level, :to_level, :]) \
            - sum(self.electron_neutral_collisions[from_level, (to_level + 1):atomic_db.atomic_levels, :]) \
            - self.electron_loss_collisions[from_level, :]

    def assemble_ion_impact_population_loss_terms(self, ion, from_level, to_level, atomic_db):
        self.ion_terms[ion][from_level, to_level, :] = \
            - sum(self.ion_neutral_collisions[ion][from_level, :to_level, :]) \
            - sum(self.ion_neutral_collisions[ion][from_level,
                  (to_level + 1):atomic_db.atomic_levels, :]) \
            - self.electron_loss_ion_collisions[ion][from_level, :]

    def assemble_electron_impact_population_gain_terms(self, from_level, to_level):
        self.electron_terms[from_level, to_level, :] = \
            self.electron_neutral_collisions[from_level, to_level, :]

    def assemble_ion_impact_population_gain_terms(self, ion, from_level, to_level):
        self.ion_terms[ion][from_level, to_level, :] = \
            self.ion_neutral_collisions[ion][from_level, to_level, :]

    def assemble_spontaneous_population_loss_terms(self, from_level, to_level, atomic_db):
        self.photon_terms[from_level, to_level, :] = \
            - sum(self.einstein_coeffs[:][atomic_db.inv_atomic_dict[to_level]]) / self.velocity
        
    def assemble_spontaneous_population_gain_terms(self, from_level, to_level, atomic_db):
        self.photon_terms[from_level, to_level, :] = \
            self.einstein_coeffs[atomic_db.inv_atomic_dict[from_level]][atomic_db.inv_atomic_dict[to_level]] / self.velocity
        
    def apply_electron_density(self, step):
        self.matrix[:, :, step] = self.beamlet_profiles['electron']['density']['m-3'][step] \
                                  * self.electron_terms[:, :, step]
        
    def apply_ion_density(self, ion, step):
        self.matrix[:, :, step] = self.matrix[:, :, step] + \
                                  self.beamlet_profiles['ion' + str(ion + 1)]['density']['m-3'][step] \
                                  * self.ion_terms[ion][:, :, step]
        
    def apply_photons(self, step):
        self.matrix[:, :, step] = self.matrix[:, :, step] + self.photon_terms[:, :, step]
