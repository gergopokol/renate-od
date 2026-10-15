import pandas
import numpy


class Obs1d:
    def __init__(self, beamlet, from_level=None, to_level=None):
        self.beamlet = beamlet
        if (from_level is None) or (to_level is None):
            self.observed_level = beamlet.atomic_db.set_default_atomic_levels()[3]
        else:
            self.observed_level = str(from_level) + '-' + str(to_level)
        beamlet.compute_linear_emission_density(from_level, to_level)
        self.photon_emission_profile = numpy.zeros(self.obs_profile.size)
        self.emission_profile = pandas.DataFrame()

    def calculate_photon_emission_profile(self, interpolate=False):
        if interpolate:
            new_resolution = int((max(self.beamlet.profiles['beamlet grid']['distance']['m']) -
                                  min(self.beamlet.profiles['beamlet grid']['distance']['m'])) /
                                 self.detector_size * 10)
            self.beamlet_grid_int = numpy.linspace(min(self.beamlet.profiles['beamlet grid']['distance']['m']),
                                                   max(self.beamlet.profiles['beamlet grid']['distance']['m']),
                                                   new_resolution)
            self.observed_profile_int = numpy.interp(self.beamlet_grid_int, self.beamlet.profiles['beamlet grid'][
                'distance']['m'], self.beamlet.profiles[self.observed_level])
        else:
            self.beamlet_grid_int = self.beamlet.profiles['beamlet grid']['distance']['m']
            self.observed_profile_int = self.beamlet.profiles[self.observed_level]
        step_size = (max(self.beamlet_grid_int)-min(self.beamlet_grid_int))/len(self.beamlet_grid_int)
        for detector_index in range(len(self.obs_profile)):
            if self.obs_profile[0][detector_index] + self.detector_size / 2. < max(self.beamlet_grid_int):
                detector_steps = [i for i in range(self.beamlet_grid_int.size) if abs(
                    self.beamlet_grid_int[i] - self.obs_profile[0][detector_index])
                                       < (self.detector_size/2)]
                self.photon_emission_profile[detector_index] = sum([self.observed_profile_int[step]*step_size
                                                                    for step in detector_steps])
        self.emission_profile = pandas.concat([self.obs_profile, pandas.DataFrame(self.photon_emission_profile)],
                                              axis=1, keys=['Location', 'Emission'])
        return
