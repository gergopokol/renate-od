import unittest
import numpy
import scipy.stats as st
from unittest.util import safe_repr
from observation.noise import Noise, APD, PMT, PP, Detector


class NoiseBasicTestCase(unittest.TestCase):
    @staticmethod
    def _WithinPrecision(actual, reference, precision):
        if abs(actual - reference)/reference <= precision:
            return True, 'Actual precision: \t %s, Reference precision: \t %s. \n' % (safe_repr(abs(actual - reference)
                                                                                            / reference), precision)
        else:
            return False, 'Actual precision: \t %s, Reference precision: \t %s. \n' % (safe_repr(abs(actual - reference)
                                                                                             / reference), precision)

    def assertDistributionVariance(self, series, reference_variance, precision=1E-2, msg=''):
        actual = series.var()
        status, statement = self._WithinPrecision(actual, reference_variance, precision)
        if not status:
            standardMsg = '\n Actual distribution function variance: \t %s and \n reference variance: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_variance)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionMean(self, series, reference_mean, precision=1E-2, msg=''):
        actual = series.mean()
        status, statement = self._WithinPrecision(actual, reference_mean, precision)
        if not status:
            standardMsg = '\n Actual distribution function mean: \t %s and \n reference mean: \t %s are not within ' \
                          'precision margin. \n' % (safe_repr(actual), safe_repr(reference_mean)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionStandardDeviation(self, series, reference_std, precision=1E-2, msg=''):
        actual = series.std()
        status, statement = self._WithinPrecision(actual, reference_std, precision)
        if not status:
            standardMsg = '\n Actual distribution function std: \t %s and \n reference std: \t %s are not within ' \
                          'precision margin. \n' % (safe_repr(actual), safe_repr(reference_std)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionSkewness(self, series, reference_skewness, precision=1E-2, msg=''):
        actual = st.skew(series)
        status, statement = self._WithinPrecision(actual, reference_skewness, precision)
        if not status:
            standardMsg = '\n Actual distribution function skewness: \t %s and \n reference skewness: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_skewness)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionKurtosis(self, series, reference_kurtosis, precision=1E-2, msg=''):
        actual = st.kurtosis(series)
        status, statement = self._WithinPrecision(actual, reference_kurtosis, precision)
        if not status:
            standardMsg = '\n Actual distribution function kurtosis: \t %s and \n reference kurtosis: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_kurtosis)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)


class NoiseGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000

    def setUp(self):
        self.noise_gen = Noise()

    def tearDown(self):
        del self.noise_gen

    def test_poisson_generator(self):
        test_data = self.noise_gen.poisson(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE))
        self.assertDistributionVariance(test_data, self.INPUT_VALUE, msg='Poisson generator variance test FAIL.')
        self.assertDistributionMean(test_data, self.INPUT_VALUE, msg='Poisson generator mean test FAIL.')
        self.assertDistributionStandardDeviation(test_data, numpy.sqrt(self.INPUT_VALUE),
                                                 msg='Poisson generator std test FAIL.')

    def test_gaussian_generator(self):
        pass

    def test_seeded_generator(self):
        pass
