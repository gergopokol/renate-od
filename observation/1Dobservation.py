import h5py
import os
import numpy

class 1Dobservation:

        def __init__(self):
            detectors = 6
            observed_range = [0.01,0.08]
            detector_pos = numpy.linspace(observed_range[0],observed_range[1],detectors)
            observation_range = 0.0025
            current = 0.002


