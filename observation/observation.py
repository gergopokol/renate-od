import os
import numpy
import math
from utility.constants import Constants
from utility import convert
from utility import getdata
from pdb import set_trace as bp


class Obs1d:
    def __init__(self, input_obj):
        self.inputs = input_obj
        self.constants = Constants()
        self.velocity = convert.calculate_velocity_from_energy(self.inputs.beam_energy,
                                                               getdata.get_mass(self.inputs.beam_species))
        self.photon_fraction = self.inputs.beam_current/self.constants.charge_electron/self.velocity

    def calculate_light_profile(self, population_evolution):
        light_profile = numpy.full(self.inputs.observation_profile.size, 0)
        for detector_index in range(len(self.inputs.observation_profile)):
            loc_up = self.inputs.observation_profile[detector_index] + self.inputs.detector_size / 2
            loc_down = self.inputs.observation_profile[detector_index] - self.inputs.detector_size / 2
            position_index_1 = numpy.where((self.inputs.steps < loc_up))
            position_index_2 = numpy.where((self.inputs.steps > loc_down))
            low = list(position_index_1)
            up = list(position_index_2)
            for step_index in range(up[0][0], low[0][-1]):
                light_profile[detector_index] = light_profile[detector_index] + population_evolution[step_index, 1]\
                                                                                * self.photon_fraction
        return light_profile
