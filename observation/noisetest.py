import unittest
import numpy
import pandas


class NoiseTest(unittest.TestCase):

    INPUT_SIGNAL = pandas.Series(data=numpy.full(100, 1E9))

    def setUp(self):
        pass

    def tearDown(self):
        pass

