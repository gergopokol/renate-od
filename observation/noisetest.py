import unittest
import numpy
from observation.noise import Noise, APD, PMT, PP, Detector


class NoiseTest(unittest.TestCase):

    INPUT_SIGNAL_CONSTANT = numpy.full(100, 10000)
    INSTANCE_NR = 100000

    def setUp(self):
        self.noise_gen = Noise()

    def tearDown(self):
        del self.noise_gen

    def test_poisson_generator(self):
        pass

    def test_gaussian_generator(self):
        pass

    def test_seeded_generator(self):
        pass
