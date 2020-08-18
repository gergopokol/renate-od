import unittest
import numpy
import pandas
from observation.noise import Generator


class GeneratorTest(unittest.TestCase):

    INPUT_SIGNAL = pandas.Series(data=numpy.full(100, 1E9))

    def setUp(self):
        self.generator = Generator(Data=self.INPUT_SIGNAL)

    def tearDown(self):
        del self.generator

