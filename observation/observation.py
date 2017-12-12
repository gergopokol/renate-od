import os
import numpy
import math
from pdb import set_trace as bp

class Obs_1d:
    def __init__(self, input_obj):
        self.inpu = input_obj
        self.charge = 1.602*1e-19
        self.einstein = 3.72*1e7
        self.mass = 1.1525*1e-26
        self.velocity = math.sqrt(2 * self.inpu.beam_energy * 1000 * self.charge/ self.mass)
        self.photon_fraction = (1/self.charge)/(self.velocity)*self.inpu.beam_current

    def calculate_light_profile(self,population_evolution):
        light_profile = numpy.full(self.inpu.observation_profile.size,0)
        for detector_index in range(len(self.inpu.observation_profile)):
            loc_up = self.inpu.observation_profile[detector_index]+self.inpu.detector_size/2
            loc_down = self.inpu.observation_profile[detector_index]-self.inpu.detector_size/2
            position_index_1 = numpy.where((self.inpu.steps < loc_up))
            position_index_2 = numpy.where((self.inpu.steps > loc_down))
            low = list(position_index_1)
            up = list(position_index_2)

            for step_index in range(up[0][0],low[0][-1]):
                light_profile[detector_index] = light_profile[detector_index]+ population_evolution[step_index,1]*self.photon_fraction
        return light_profile