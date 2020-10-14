import unittest
import numpy
import scipy.stats as st
from unittest.util import safe_repr
from utility.constants import Constants
from utility.getdata import GetData
from observation.noise import Noise, APD, PMT, PPD, Detector


class NoiseBasicTestCase(unittest.TestCase):
    @staticmethod
    def _WithinRelativePrecision(actual, reference, precision):
        if abs(actual - reference)/reference <= precision:
            return True, 'Actual precision: \t %s, Reference precision: \t %s. \n' % (safe_repr(abs(actual - reference)
                                                                                      / reference), precision)
        else:
            return False, 'Actual precision: \t %s, Reference precision: \t %s. \n' % (safe_repr(abs(actual - reference)
                                                                                       / reference), precision)

    @staticmethod
    def _WithinAbsolutePrecision(actual, reference, precision):
        if abs(actual - reference) <= precision:
            return True, 'Actual precision: \t %s, Reference precision: \t %s. \n' % \
                   (safe_repr(abs(actual - reference)), precision)
        else:
            return False, 'Actual precision: \t %s, Reference precision: \t %s. \n' % \
                   (safe_repr(abs(actual - reference)), precision)

    def assertDistributionVariance(self, series, reference_variance, precision=1E-2, msg=''):
        actual = series.var()
        status, statement = self._WithinRelativePrecision(actual, reference_variance, precision)
        if not status:
            standardMsg = '\n Actual distribution function variance: \t %s and \n reference variance: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_variance)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionMean(self, series, reference_mean, precision=1E-2, msg=''):
        actual = series.mean()
        status, statement = self._WithinRelativePrecision(actual, reference_mean, precision)
        if not status:
            standardMsg = '\n Actual distribution function mean: \t %s and \n reference mean: \t %s are not within ' \
                          'precision margin. \n' % (safe_repr(actual), safe_repr(reference_mean)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionStandardDeviation(self, series, reference_std, precision=1E-2, msg=''):
        actual = series.std()
        status, statement = self._WithinRelativePrecision(actual, reference_std, precision)
        if not status:
            standardMsg = '\n Actual distribution function std: \t %s and \n reference std: \t %s are not within ' \
                          'precision margin. \n' % (safe_repr(actual), safe_repr(reference_std)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionSkewness(self, series, reference_skewness, precision=1E-2, msg=''):
        actual = st.skew(series)
        status, statement = self._WithinAbsolutePrecision(actual, reference_skewness, precision)
        if not status:
            standardMsg = '\n Actual distribution function skewness: \t %s and \n reference skewness: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_skewness)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertDistributionKurtosis(self, series, reference_kurtosis, precision=1E-2, msg=''):
        actual = st.kurtosis(series)
        status, statement = self._WithinAbsolutePrecision(actual, reference_kurtosis, precision)
        if not status:
            standardMsg = '\n Actual distribution function kurtosis: \t %s and \n reference kurtosis: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_kurtosis)) + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertHasAttributes(self, actual_class, reference_attributes, msg=''):
        status = True
        statement = 'Expected attributes: \t Presence of attributes: \n'
        for attribute in reference_attributes:
            if not hasattr(actual_class, attribute):
                status = False
                statement += '%s, \t %s, \n' % (safe_repr(attribute), safe_repr(False))
            else:
                statement += '%s, \t %s, \n' % (safe_repr(attribute), safe_repr(True))
        if not status:
            standardMsg = '\n Input class does not have all expected attributes. \n' + statement
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)


class NoiseGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_INSTANCE_2 = 20
    INPUT_SEED = 20
    INPUT_STD = 100
    INPUT_FREQUENCY = 1E6
    INPUT_PHOTON_FLUX = 1E9
    INPUT_SBR = 5
    INPUT_QE = 2
    INPUT_LOAD_RESIST = 2
    INPUT_BANDWIDTH = 2
    INPUT_GAIN = 2
    INPUT_NOISE_INDEX = 1
    INPUT_CONST = Constants()
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_PHOTON_FLUX)
    INPUT_SIGNAL_2 = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    EXPECTED_PRECISION_4 = 4

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
        self.assertDistributionSkewness(test_data, 1/numpy.sqrt(self.INPUT_VALUE),
                                        msg='Poisson generator skewness test FAIL')
        self.assertDistributionKurtosis(test_data, 1/self.INPUT_VALUE, precision=2E-2,
                                        msg='Poisson generator kurtosis test FAIL.')

    def test_gaussian_generator(self):
        test_data = self.noise_gen.normal(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE), self.INPUT_STD)
        self.assertDistributionMean(test_data, self.INPUT_VALUE, msg='Normal generator mean test FAIL.')
        self.assertDistributionVariance(test_data, self.INPUT_STD**2, msg='Normal generator variance FAIL.')
        self.assertDistributionStandardDeviation(test_data, self.INPUT_STD, msg='Normal generator std test FAIL.')
        self.assertDistributionSkewness(test_data, 0, msg='Normal generator skewness test FAIL.')
        self.assertDistributionKurtosis(test_data, 0, precision=2E-2, msg='Normal generator kurtosis test FAIL.')

    def test_seeded_poisson_generator(self):
        reference_gen = Noise(seed=self.INPUT_SEED)
        self.noise_gen.seed(self.INPUT_SEED)
        actual_data = self.noise_gen.poisson(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE))
        reference_data = reference_gen.poisson(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE))
        self.assertListEqual(list(actual_data), list(reference_data),
                             msg='Generator seed test fail for Poisson distribution.')

    def test_seeded_normal_generator(self):
        reference_gen = Noise(seed=self.INPUT_SEED)
        self.noise_gen.seed(self.INPUT_SEED)
        actual_data = self.noise_gen.normal(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE), self.INPUT_STD)
        reference_data = reference_gen.normal(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE), self.INPUT_STD)
        self.assertListEqual(list(actual_data), list(reference_data),
                             msg='Generator seed test fail for Normal distribution.')

    def test_signal_length(self):
        self.assertEqual(self.noise_gen.signal_length(self.INPUT_SIGNAL), self.INPUT_INSTANCE, msg='<signal_length> '
                         'routine is expected to return the length of the input signal.')

    def test_photon_flux_to_count(self):
        actual_signal = self.noise_gen._photon_flux_to_photon_number(self.INPUT_SIGNAL, self.INPUT_FREQUENCY)
        self.assertTupleEqual(actual_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The expected output signal shape is expected to be equal to input signal shape.')
        self.assertEqual(actual_signal.mean(), self.INPUT_PHOTON_FLUX / self.INPUT_FREQUENCY,
                         msg='The actual signal values are expected to be normalized with the signal frequency.')

    def test_background_addition(self):
        actual_signal = self.noise_gen.background_addition(self.INPUT_SIGNAL, self.INPUT_SBR)
        self.assertTupleEqual(actual_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The background signal addition is not expected to change the signal length.')
        self.assertEqual(actual_signal.mean(), self.INPUT_PHOTON_FLUX * (1 + self.INPUT_SBR**-1),
                         msg='The background light addition should be an SBR-th portion of the modelled mean signal.')

    def test_photon_flux_to_detector_voltage(self):
        actual_signal = self.noise_gen.photon_flux_to_detector_voltage(self.INPUT_SIGNAL, self.INPUT_GAIN,
                                                                       self.INPUT_QE, self.INPUT_LOAD_RESIST)
        reference_detector_voltage = self.INPUT_PHOTON_FLUX * self.INPUT_LOAD_RESIST * self.INPUT_QE *\
                                     self.INPUT_GAIN * self.INPUT_CONST.charge_electron
        self.assertTupleEqual(actual_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The detector voltage converter is not expected to change the output signal shape.')
        self.assertAlmostEqual(actual_signal.mean(), reference_detector_voltage, places=self.EXPECTED_PRECISION_4,
                               msg='The detector voltage converter is expected to be <signal*e*G*QE*R_L>')

    def test_photon_noise_generator(self):
        actual_signal = self.noise_gen.generate_photon_noise(self.INPUT_SIGNAL)
        self.assertTupleEqual(actual_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The photon noise generator routine is not expected to change the output signal.')
        self.assertDistributionMean(actual_signal, self.INPUT_PHOTON_FLUX,
                                    msg='The photon noise generator does not return expected <mean> value.')
        self.assertDistributionStandardDeviation(actual_signal, numpy.sqrt(self.INPUT_PHOTON_FLUX),
                                                 msg='Photon Noise Generator is expected to create Poisson '
                                                     'distributions. The actual STD does not match.')

    def test_shot_noise_setup(self):
        mean, variance = self.noise_gen._shot_noise_setup(self.INPUT_SIGNAL_2, self.INPUT_GAIN, self.INPUT_LOAD_RESIST,
                                                          self.INPUT_NOISE_INDEX, self.INPUT_BANDWIDTH)
        self.assertTupleEqual(mean.shape, self.INPUT_SIGNAL_2.shape, msg='The mean values for shot signal noise '
                              'generation is expected to have the same shape.')
        self.assertTupleEqual(variance.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The variance values for shot noise generation is expected to have the same shape.')
        for index in range(self.INPUT_INSTANCE):
            self.assertEqual(mean[index], self.INPUT_SIGNAL_2[index], msg='Mean value for noise generator is expected '
                                                                          'to be equal to input signal values.')
            self.assertEqual(variance[index], self.INPUT_LOAD_RESIST * numpy.sqrt(2 * self.INPUT_CONST.charge_electron *
                             self.INPUT_BANDWIDTH * numpy.power(self.INPUT_GAIN, self.INPUT_NOISE_INDEX+1) *
                             self.INPUT_SIGNAL_2[index]), msg='Variance values for noise generator is '
                                                              'expected to be equal to R*sqrt(2*q*B*G^(x+1)*signal)')

    def test_shot_noise_generator(self):
        noisy_signal = self.noise_gen.shot_noise_generator(self.INPUT_SIGNAL_2, detector_gain=self.INPUT_GAIN,
                                                           load_resistance=self.INPUT_LOAD_RESIST,
                                                           noise_index=self.INPUT_NOISE_INDEX,
                                                           bandwidth=self.INPUT_BANDWIDTH)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The shot noise generator routine is not expected to change the signal shape.')
        mean, variance = self.noise_gen._shot_noise_setup(self.INPUT_SIGNAL_2, detector_gain=self.INPUT_GAIN,
                                                          load_resistance=self.INPUT_LOAD_RESIST,
                                                          noise_index=self.INPUT_NOISE_INDEX,
                                                          bandwidth=self.INPUT_BANDWIDTH)
        self.assertDistributionMean(noisy_signal, mean.mean(),
                                    msg='The shot noise generator does not return expected <mean> value.')
        self.assertDistributionStandardDeviation(noisy_signal, variance.mean(),
                                                 msg='Shot Noise Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')


class APDGeneratorTest(NoiseBasicTestCase):

    DEFAULT_APD_PATH = 'detector/apd_default.xml'
    EXPECTED_ATTRIBUTES = ['detector_temperature', 'detector_gain', 'quantum_efficiency', 'noise_index', 'bandwidth',
                           'dark_current', 'load_resistance', 'load_capacity', 'voltage_noise', 'internal_capacity',
                           'sampling_frequency', 'signal_to_background']

    def setUp(self):
        self.APD = APD(GetData(data_path_name=self.DEFAULT_APD_PATH).data)

    def tearDown(self):
        del self.APD

    def test_class_inheritance(self):
        self.assertIsInstance(self.APD, Noise, msg='<APD> class is expected to be a child of <Noise>.')

    def test_parameter_attributes(self):
        self.assertHasAttributes(self.APD, self.EXPECTED_ATTRIBUTES, msg='<APD> class does not initiate will all '
                                                                         'expected attributes.')


class PMTGeneratorTest(NoiseBasicTestCase):

    DEFAULT_PMT_PATH = 'detector/pmt_default.xml'

    def setUp(self):
        self.PMT = PMT(GetData(data_path_name=self.DEFAULT_PMT_PATH).data)

    def tearDown(self):
        del self.PMT

    def test_class_inheritance(self):
        self.assertIsInstance(self.PMT, Noise, msg='<PMT> class is expected to be a child of <Noise>.')


class PPGeneratorTest(NoiseBasicTestCase):

    DEFAULT_PPD_PATH = 'detector/ppd_default.xml'

    def setUp(self):
        self.PPD = PPD(GetData(data_path_name=self.DEFAULT_PPD_PATH).data)

    def tearDown(self):
        del self.PPD

    def test_class_inheritance(self):
        self.assertIsInstance(self.PPD, Noise, msg='<PPD> class is expected to be a child of <Noise>.')


class DetectorGeneratorTest(NoiseBasicTestCase):

    INPUT_APD_TYPE = 'apd'
    INPUT_PMT_TYPE = 'pmt'
    INPUT_PPD_TYPE = 'ppd'
    INPUT_VAL_ERROR_1 = 'bbb'
    INPUT_VAL_ERROR_2 = 123

    def test_APD_instantiation(self):
        actual = Detector(detector_type=self.INPUT_APD_TYPE)
        self.assertIsInstance(actual, APD, msg='<Detector> class is expected to return a <APD> class.')

    def test_PMT_instantiation(self):
        actual = Detector(detector_type=self.INPUT_PMT_TYPE)
        self.assertIsInstance(actual, PMT, msg='<Detector> class is expected to return a <PMT> class.')

    def test_PPD_instantiation(self):
        actual = Detector(detector_type=self.INPUT_PPD_TYPE)
        self.assertIsInstance(actual, PPD, msg='<Detector> class is expected to return a <PPD> class.')

    def test_exception_handling(self):
        with self.assertRaises(ValueError, msg='ValueError raising is expected in case of wrong str input.'):
            actual = Detector(detector_type=self.INPUT_VAL_ERROR_1)

        with self.assertRaises(AssertionError, msg='AssertionError raising is expected in case '
                                                   'of wrong detector type input.'):
            actual = Detector(detector_type=self.INPUT_VAL_ERROR_2)
