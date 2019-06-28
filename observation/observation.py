from lxml import etree
import pandas
import numpy
from utility.constants import Constants
from utility import getdata
import warnings



class Obs1d:
    def __init__(self, beamlet, data_path, from_level=None, to_level=None):
        self.beamlet = beamlet
        self.constants = Constants()
        self.obs_param = self.read_observation_param(data_path)
        self.obs_profile = self.read_observation_profile(data_path=self.obs_param.getroot().find('body').find(
            'observation_profile_path').text)
        self.detector_size = float(self.obs_param.getroot().find('body').find('detector_size').text)
        if (max([self.obs_profile[0][i+1]-self.obs_profile[0][i] for i in range(self.obs_profile[0].size-1)]) >
                min([self.beamlet.profiles['beamlet grid']['distance']['m'][j + 1] -
                 self.beamlet.profiles['beamlet grid']['distance']['m'][j] for j in range(
                self.beamlet.profiles['beamlet grid']['distance']['m'].size - 1)])):
            warnings.warn('The calculated resolution is poorer than the detector resolution. It is recommended to rerun the '
                  'photon emission calculations on a finer grid. If you still want to continue, '
                  'call calculate_photon_emission_profile() function.')

        if (from_level is None) or (to_level is None):
            self.observed_level = beamlet.atomic_db.set_default_atomic_levels()[3]
        else:
            self.observed_level = str(from_level) + '-' + str(to_level)
        beamlet.compute_linear_emission_density(from_level, to_level)
        self.photon_emission_profile = numpy.zeros(self.obs_profile.size)
        self.emission_profile = pandas.DataFrame()

    def calculate_photon_emission_profile(self):
        for detector_index in range(len(self.obs_profile)):
            if self.obs_profile[0][detector_index] + self.detector_size / 2. < max(self.beamlet.profiles['beamlet grid']['distance']['m']):
                detector_steps = [i for i in range(self.beamlet.profiles['beamlet grid']['distance']['m'].size) if abs(
                    self.beamlet.profiles['beamlet grid']['distance']['m'][i] - self.obs_profile[0][detector_index])
                                       < (self.detector_size/2)]
                self.photon_emission_profile[detector_index] = sum([self.beamlet.profiles[self.observed_level][step]
                                                                    for step in detector_steps])
        observing_detectors = numpy.nonzero(self.photon_emission_profile)
        self.emission_profile = pandas.concat([self.obs_profile, pandas.DataFrame(self.photon_emission_profile)],
                                              axis=1, keys=['Location', 'Emission'])
        print('Only detectors completely inside the calculated region could observe the virtual beamlet, these detector'
              ' numbers are: ' + str(observing_detectors) + '.')
        return

    @staticmethod
    def read_observation_param(data_path):
        obs_param = getdata.GetData(data_path_name=data_path).data
        assert isinstance(obs_param, etree._ElementTree)
        print('obs_param read from file: ' + data_path)
        return obs_param

    @staticmethod
    def read_observation_profile(data_path):
        obs_profile = getdata.GetData(data_path_name=data_path).data
        assert isinstance(obs_profile, pandas.DataFrame)
        print('obs_profile read from file: ' + data_path)
        return obs_profile
