from lxml import etree
import pandas
import numpy
import math
from utility.constants import Constants
from utility import convert
from utility import getdata
import re
from pdb import set_trace as bp


class Obs1d:
    def __init__(self, beamlet, data_path, beam_current):
        self.beamlet = beamlet
        self.constants = Constants()
        self.obs_param = self.read_observation_param(data_path)
        self.obs_profile = self.read_observation_profile(data_path=self.obs_param.getroot().find('body').find('observation_profile_path').text)
        self.observed_level = self.obs_param.getroot().find('body').find('observed_level').text
        self.observed_level_index = int(re.findall('\\d+', self.observed_level)[0])
        self.photon_fraction = beam_current * self.beamlet.coefficient_matrix.spontaneous_trans_np[0, self.observed_level_index]\
                               / self.constants.charge_electron / float(beamlet.param.getroot().find('body').find('beamlet_velocity').text)
        self.photon_emission_profile = numpy.zeros(self.obs_profile.size)
        self.emission_profile = pandas.DataFrame()

    def calculate_photon_emission_profile(self):
        for detector_index in range(len(self.obs_profile)):
            if self.obs_profile[0][detector_index] + float(self.obs_param.getroot().find('body').find(
                    'detector_size').text) / 2 < max(self.beamlet.profiles['beamlet grid']):
                detector_steps = numpy.where(abs(self.beamlet.profiles['beamlet grid']-self.obs_profile[0][detector_index])
                                             < float(self.obs_param.getroot().find('body').find('detector_size').text) / 2)
                for step_index in detector_steps[0]:
                    self.photon_emission_profile[detector_index] = self.photon_emission_profile[detector_index] + \
                                                    self.beamlet.profiles[self.observed_level][step_index] * \
                                                    self.photon_fraction
        observing_detectors = numpy.where(self.photon_emission_profile != 0)[0]
        self.emission_profile = pandas.concat([self.obs_profile, pandas.DataFrame(self.photon_emission_profile)], axis=1,
                                              keys=['Location', 'Emission'])
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
