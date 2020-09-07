import unittest
import numpy
from observation.noise import Noise, APD, PMT, PP, Detector


class NoiseBasicTestCase(unittest.TestCase):
    pass


class NoiseGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 100
    INPUT_INSTANCE = 100000

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
