import unittest
import numpy
import scipy.stats as st
import os
from shutil import rmtree
from unittest.util import safe_repr
from utility.accessdata import AccessData
from utility.constants import Constants
from utility.getdata import GetData
from observation.noise import Noise, APD, PMT, PPD, MPPC, Detector


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

    def _WithinPrecision(self, actual, reference, precision, comparison):
        if comparison == 'relative':
            return self._WithinRelativePrecision(actual, reference, precision)
        elif comparison == 'absolute':
            return self._WithinAbsolutePrecision(actual, reference, precision)
        else:
            raise ValueError('The comparison type: ' + comparison + ' is not supported.')

    def assertDistributionVariance(self, series, reference_variance, precision=1E-2, comparison='relative', msg=''):
        actual = series.var()
        status, statement = self._WithinPrecision(actual, reference_variance, precision, comparison)
        if not status:
            standard_msg = '\n Actual distribution function variance: \t %s and \n reference variance: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_variance)) + statement
            msg = self._formatMessage(msg, standard_msg)
            self.fail(msg)

    def assertDistributionMean(self, series, reference_mean, precision=1E-2, comparison='relative', msg=''):
        actual = series.mean()
        status, statement = self._WithinPrecision(actual, reference_mean, precision, comparison)
        if not status:
            standard_msg = '\n Actual distribution function mean: \t %s and \n reference mean: \t %s are not within ' \
                          'precision margin. \n' % (safe_repr(actual), safe_repr(reference_mean)) + statement
            msg = self._formatMessage(msg, standard_msg)
            self.fail(msg)

    def assertDistributionStandardDeviation(self, series, reference_std, comparison='relative', precision=1E-2, msg=''):
        actual = series.std()
        status, statement = self._WithinPrecision(actual, reference_std, precision, comparison)
        if not status:
            standard_msg = '\n Actual distribution function std: \t %s and \n reference std: \t %s are not within ' \
                          'precision margin. \n' % (safe_repr(actual), safe_repr(reference_std)) + statement
            msg = self._formatMessage(msg, standard_msg)
            self.fail(msg)

    def assertDistributionSkewness(self, series, reference_skewness, comparison='relative', precision=1E-2, msg=''):
        actual = st.skew(series)
        status, statement = self._WithinPrecision(actual, reference_skewness, precision, comparison)
        if not status:
            standard_msg = '\n Actual distribution function skewness: \t %s and \n reference skewness: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_skewness)) + statement
            msg = self._formatMessage(msg, standard_msg)
            self.fail(msg)

    def assertDistributionKurtosis(self, series, reference_kurtosis, comparison='relative', precision=1E-2, msg=''):
        actual = st.kurtosis(series)
        status, statement = self._WithinPrecision(actual, reference_kurtosis, precision, comparison)
        if not status:
            standard_msg = '\n Actual distribution function kurtosis: \t %s and \n reference kurtosis: \t %s are not ' \
                          'within precision margin. \n' % (safe_repr(actual), safe_repr(reference_kurtosis)) + statement
            msg = self._formatMessage(msg, standard_msg)
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
            standard_msg = '\n Input class does not have all expected attributes. \n' + statement
            msg = self._formatMessage(msg, standard_msg)
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
    INPUT_AMPLIFICATION = 2
    INPUT_NOISE_AMPLIFICATION = 2
    INPUT_DET_TEMP = 300
    INPUT_DARK_CURRENT = 10
    INPUT_VOLTAGE_NOISE = 10
    INPUT_LOAD_CAPACITY = 5
    INPUT_INTERNAL_CAPACITY = 8
    INPUT_NET_GAIN = 2
    INPUT_CONST = Constants()
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_PHOTON_FLUX)
    INPUT_SIGNAL_2 = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    EXPECTED_PRECISION_4 = 4
    EXPECTED_JOHNSON_MEAN = 0

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
        self.assertDistributionSkewness(test_data, 1/numpy.sqrt(self.INPUT_VALUE), comparison='absolute',
                                        msg='Poisson generator skewness test FAIL')
        self.assertDistributionKurtosis(test_data, 1/self.INPUT_VALUE, precision=2E-2, comparison='absolute',
                                        msg='Poisson generator kurtosis test FAIL.')

    def test_gaussian_generator(self):
        test_data = self.noise_gen.normal(numpy.full(self.INPUT_INSTANCE, self.INPUT_VALUE), self.INPUT_STD)
        self.assertDistributionMean(test_data, self.INPUT_VALUE, msg='Normal generator mean test FAIL.')
        self.assertDistributionVariance(test_data, self.INPUT_STD**2, msg='Normal generator variance FAIL.')
        self.assertDistributionStandardDeviation(test_data, self.INPUT_STD, msg='Normal generator std test FAIL.')
        self.assertDistributionSkewness(test_data, 0, comparison='absolute',
                                        msg='Normal generator skewness test FAIL.')
        self.assertDistributionKurtosis(test_data, 0, precision=2E-2, comparison='absolute',
                                        msg='Normal generator kurtosis test FAIL.')

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
        mean, std = self.noise_gen._shot_noise_setup(signal=self.INPUT_SIGNAL_2,
                                                     load_resistance=self.INPUT_LOAD_RESIST,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     amplification=self.INPUT_AMPLIFICATION,
                                                     noise_amplification=self.INPUT_NOISE_AMPLIFICATION,
                                                     quantum_efficiency=self.INPUT_QE,
                                                     sampling_frequency=self.INPUT_FREQUENCY)
        self.assertTupleEqual(mean.shape, self.INPUT_SIGNAL_2.shape, msg='The mean values for shot signal noise '
                              'generation is expected to have the same shape.')
        self.assertTupleEqual(std.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The std values for shot noise generation is expected to have the same shape.')
        self.assertEqual(mean.all(), (self.INPUT_SIGNAL_2 * self.INPUT_CONST.charge_electron * self.INPUT_AMPLIFICATION
                                      * self.INPUT_QE * self.INPUT_LOAD_RESIST * self.INPUT_FREQUENCY).all(),
                         msg='Mean value for shot noise setup is expected to be equal to theoretical indicated values.')
        self.assertEqual(std.all(), numpy.sqrt(2 * self.INPUT_CONST.charge_electron * mean * self.INPUT_AMPLIFICATION *
                                               self.INPUT_NOISE_AMPLIFICATION * self.INPUT_BANDWIDTH *
                                               self.INPUT_LOAD_RESIST).all(),
                         msg='Std values for noise generator is expected to be equal to theoretical indicated value')

    def test_shot_noise_generator(self):
        noisy_signal = self.noise_gen.shot_noise_generator(signal=self.INPUT_SIGNAL_2,
                                                           load_resistance=self.INPUT_LOAD_RESIST,
                                                           bandwidth=self.INPUT_BANDWIDTH,
                                                           amplification=self.INPUT_AMPLIFICATION,
                                                           noise_amplification=self.INPUT_NOISE_AMPLIFICATION,
                                                           quantum_efficiency=self.INPUT_QE,
                                                           sampling_frequency=self.INPUT_FREQUENCY)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The shot noise generator routine is not expected to change the signal shape.')
        mean, std = self.noise_gen._shot_noise_setup(signal=self.INPUT_SIGNAL_2,
                                                     load_resistance=self.INPUT_LOAD_RESIST,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     amplification=self.INPUT_AMPLIFICATION,
                                                     noise_amplification=self.INPUT_NOISE_AMPLIFICATION,
                                                     quantum_efficiency=self.INPUT_QE,
                                                     sampling_frequency=self.INPUT_FREQUENCY)
        self.assertDistributionMean(noisy_signal, mean.mean(),
                                    msg='The shot noise generator does not return expected <mean> value.')
        self.assertDistributionStandardDeviation(noisy_signal, std.mean(),
                                                 msg='Shot Noise Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')

    def test_johnson_noise_setup(self):
        mean, std = self.noise_gen._johnson_noise_setup(detector_temperature=self.INPUT_DET_TEMP,
                                                        bandwidth=self.INPUT_BANDWIDTH,
                                                        load_resistance=self.INPUT_LOAD_RESIST)
        self.assertEqual(mean, self.EXPECTED_JOHNSON_MEAN,
                         msg='The expected mean value for the Johnson noise contribution is <0>')
        self.assertEqual(std, numpy.sqrt(4 * self.INPUT_BANDWIDTH * self.INPUT_LOAD_RESIST * self.INPUT_DET_TEMP *
                         self.INPUT_CONST.Boltzmann), msg='The expected std for the Johnson noise '
                                                          'generator is expected to be <sqrt(4kBTR)>')

    def test_johnson_noise_generator(self):
        noisy_signal = self.noise_gen.johnson_noise_generator(detector_temperature=self.INPUT_DET_TEMP,
                                                              load_resistance=self.INPUT_LOAD_RESIST,
                                                              bandwidth=self.INPUT_BANDWIDTH,
                                                              signal_size=self.INPUT_SIGNAL_2.shape[0])
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The Johnson noise generator is expected to create a similar sized signal.')
        mean, std = self.noise_gen._johnson_noise_setup(detector_temperature=self.INPUT_DET_TEMP,
                                                        load_resistance=self.INPUT_LOAD_RESIST,
                                                        bandwidth=self.INPUT_BANDWIDTH)
        self.assertDistributionMean(noisy_signal, mean, comparison='absolute',
                                    msg='Johnson Noise Generator does not return expected value: 0.')
        self.assertDistributionStandardDeviation(noisy_signal, std,
                                                 msg='Johnson Noise Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')

    def test_dark_noise_setup(self):
        mean, std = self.noise_gen._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     load_resistance=self.INPUT_LOAD_RESIST,
                                                     net_gain=self.INPUT_NET_GAIN)
        self.assertEqual(mean, self.INPUT_LOAD_RESIST * self.INPUT_DARK_CURRENT,
                         msg='Mean value for dark current distribution function is expected to be I_dark * R_load.')
        self.assertEqual(std, self.INPUT_LOAD_RESIST * numpy.sqrt(2 * self.INPUT_CONST.charge_electron *
                         self.INPUT_BANDWIDTH * self.INPUT_DARK_CURRENT * self.INPUT_NET_GAIN),
                         msg='The expected STD for dark current generator function is R_load*sqrt(2*I_dark*B*q).')

    def test_dark_noise_generator(self):
        noisy_signal = self.noise_gen.dark_noise_generator(dark_current=self.INPUT_DARK_CURRENT,
                                                           bandwidth=self.INPUT_BANDWIDTH,
                                                           load_resistance=self.INPUT_LOAD_RESIST,
                                                           signal_size=self.INPUT_SIGNAL_2.size,
                                                           net_gain=self.INPUT_NET_GAIN)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The Dark Current noise generator is expected to create a similar sized signal.')
        mean, std = self.noise_gen._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     load_resistance=self.INPUT_LOAD_RESIST,
                                                     net_gain=self.INPUT_NET_GAIN)
        self.assertDistributionMean(noisy_signal, mean,
                                    msg='The Dark Current Generator does not return expected mean value')
        self.assertDistributionStandardDeviation(noisy_signal, std,
                                                 msg='Dark Current Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')

    def test_voltage_noise_setup(self):
        mean, std = self.noise_gen._voltage_noise_setup(voltage_noise=self.INPUT_VOLTAGE_NOISE,
                                                        load_resistance=self.INPUT_LOAD_RESIST,
                                                        load_capacity=self.INPUT_LOAD_CAPACITY,
                                                        internal_capacity=self.INPUT_INTERNAL_CAPACITY,
                                                        expected_value=0)
        self.assertEqual(mean, 0,
                         msg='Mean value for voltage noise distribution function is expected to be 0.')
        self.assertEqual(std, self.INPUT_VOLTAGE_NOISE * numpy.sqrt(1 / (2 * numpy.pi * self.INPUT_LOAD_RESIST *
                                                                         (self.INPUT_LOAD_CAPACITY +
                                                                          self.INPUT_INTERNAL_CAPACITY))) +
                         self.INPUT_VOLTAGE_NOISE * (1 + self.INPUT_INTERNAL_CAPACITY / self.INPUT_LOAD_CAPACITY) *
                         numpy.sqrt(1.57 * 1 / (2 * numpy.pi * self.INPUT_LOAD_RESIST * self.INPUT_LOAD_CAPACITY)),
                         msg='The expected STD for voltage noise generator function is N_V*'
                             'sqrt(1/(2pi*R_load*(C_load+C_in)))+N_V*(1+C_in/C_load)*sqrt(1.57*1/(2pi*R_load*C_load))')

    def test_voltage_noise_generator(self):
        noisy_signal = self.noise_gen.voltage_noise_generator(voltage_noise=self.INPUT_VOLTAGE_NOISE,
                                                              load_resistance=self.INPUT_LOAD_RESIST,
                                                              load_capacity=self.INPUT_LOAD_CAPACITY,
                                                              internal_capacity=self.INPUT_INTERNAL_CAPACITY,
                                                              signal_size=self.INPUT_SIGNAL_2.shape[0])
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The Voltage noise generator is expected to create a similar sized signal.')
        mean, std = self.noise_gen._voltage_noise_setup(voltage_noise=self.INPUT_VOLTAGE_NOISE,
                                                        load_resistance=self.INPUT_LOAD_RESIST,
                                                        load_capacity=self.INPUT_LOAD_CAPACITY,
                                                        internal_capacity=self.INPUT_INTERNAL_CAPACITY,
                                                        expected_value=0)
        self.assertDistributionMean(noisy_signal, mean, comparison='absolute',
                                    msg='The Voltage Noise Generator does not return expected mean value')
        self.assertDistributionStandardDeviation(noisy_signal, std,
                                                 msg='The Voltage Noise Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')

    def test_derive_background_emission_in_photon_count(self):
        background = self.noise_gen.derive_background_emission_in_photon_count(self.INPUT_SIGNAL_2, self.INPUT_SBR)
        mean = self.INPUT_SIGNAL_2.mean() / self.INPUT_SBR
        std = numpy.sqrt(self.INPUT_SIGNAL_2.mean() / self.INPUT_SBR)
        self.assertDistributionMean(series=background, reference_mean=mean,
                                    msg='The derive background emission into photon count function needs to create a '
                                        'theoretically indicated mean')
        self.assertDistributionStandardDeviation(series=background, reference_std=std,
                                                 msg='The derive background emission into photon count function needs'
                                                     ' to create Poisson distribution')


class APDGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    INPUT_VALUE_2 = 1E9
    INPUT_INSTANCE_2 = 1000000
    INPUT_SIGNAL_2 = numpy.full(INPUT_INSTANCE_2, INPUT_VALUE_2)
    INPUT_SEED = 20
    INPUT_CONST = Constants()
    INPUT_LOAD_RES = 1E5
    INPUT_FREQUENCY = 1E6
    INPUT_QE = 0.85
    INPUT_BANDWIDTH = 2E6
    INPUT_DETECTOR_GAIN = 50
    INPUT_NOISE_INDEX = 1
    INPUT_DARK_CURRENT = 1E-3

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

    def test_apd_noiseless_transfer(self):
        detector_voltage = self.APD._apd_noiseless_transfer(signal=self.INPUT_SIGNAL)
        mean = (self.INPUT_SIGNAL * (1 + 1 / self.APD.signal_to_background) * self.APD.quantum_efficiency *
                self.INPUT_CONST.charge_electron * self.APD.detector_gain + self.APD.dark_current) * self.\
            APD.load_resistance
        self.assertEqual(detector_voltage.all(), mean.all(),
                         msg='The APD noiseless transfer function is expected to create a theoretical '
                         'indicated value.')

    def test_apd_amplification(self):
        amplification = self.APD._apd_amplification(detector_gain=self.INPUT_DETECTOR_GAIN)
        self.assertEqual(amplification, self.INPUT_DETECTOR_GAIN,
                         msg='The APD amplification function needs to get back the given value')

    def test_apd_noise_amplification(self):
        noise_amplification = self.APD._apd_noise_amplification(detector_gain=self.INPUT_DETECTOR_GAIN,
                                                                noise_index=self.INPUT_NOISE_INDEX)
        self.assertEqual(noise_amplification, self.INPUT_DETECTOR_GAIN ** self.INPUT_NOISE_INDEX,
                         msg='The APD noise amplification function needs to get back the gain ^ noise_index value')

    def test_apd_shot_noise_setup(self):
        mean, std = self.APD._shot_noise_setup(signal=self.INPUT_SIGNAL,
                                               load_resistance=self.INPUT_LOAD_RES,
                                               bandwidth=self.INPUT_BANDWIDTH,
                                               amplification=self.APD._apd_amplification(self.INPUT_DETECTOR_GAIN),
                                               noise_amplification=self.APD._apd_noise_amplification
                                               (self.INPUT_DETECTOR_GAIN, self.INPUT_NOISE_INDEX),
                                               quantum_efficiency=self.INPUT_QE,
                                               sampling_frequency=self.INPUT_FREQUENCY)
        self.assertEqual(mean.all(), (self.INPUT_SIGNAL * self.INPUT_FREQUENCY * self.INPUT_DETECTOR_GAIN *
                                      self.INPUT_QE * self.INPUT_CONST.charge_electron * self.INPUT_LOAD_RES).all(),
                         msg='Mean value for noise generator is expected to be equal to input signal values.')
        self.assertEqual(std.all(), (numpy.sqrt(2*self.INPUT_CONST.charge_electron * mean * self.INPUT_DETECTOR_GAIN **
                                                (self.INPUT_NOISE_INDEX + 1) * self.INPUT_BANDWIDTH *
                                                self.INPUT_LOAD_RES)).all(),
                         msg='Std value for noise generator is expected to be equal to input signal values.')

    def test_apd_shot_noise_generator(self):
        noisy_signal = self.APD.shot_noise_generator(signal=self.INPUT_SIGNAL,
                                                     load_resistance=self.INPUT_LOAD_RES,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     amplification=self.APD._apd_amplification(self.INPUT_DETECTOR_GAIN
                                                                                               ),
                                                     noise_amplification=self.APD._apd_noise_amplification
                                                     (self.INPUT_DETECTOR_GAIN, self.INPUT_NOISE_INDEX),
                                                     quantum_efficiency=self.INPUT_QE,
                                                     sampling_frequency=self.INPUT_FREQUENCY)
        mean = (self.INPUT_SIGNAL * self.INPUT_FREQUENCY * self.INPUT_DETECTOR_GAIN *
                self.INPUT_QE * self.INPUT_CONST.charge_electron * self.INPUT_LOAD_RES).mean()
        std = (numpy.sqrt(2*self.INPUT_CONST.charge_electron * mean * self.INPUT_DETECTOR_GAIN **
                          (self.INPUT_NOISE_INDEX + 1) * self.INPUT_BANDWIDTH * self.INPUT_LOAD_RES)).mean()
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The function need to keep the shape of the array')
        self.assertDistributionMean(series=noisy_signal, reference_mean=mean,
                                    msg='The APD shot noise generation function needs to create an array with a well '
                                        'defined mean')
        self.assertDistributionStandardDeviation(series=noisy_signal, reference_std=std,
                                                 msg='The APD shot noise generation function needs to create an array '
                                                     'with a well defined std')

    def test_apd_dark_noise_setup(self):
        mean, std = self.APD._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                               bandwidth=self.INPUT_BANDWIDTH,
                                               load_resistance=self.INPUT_LOAD_RES)
        self.assertEqual(mean, self.INPUT_LOAD_RES * self.INPUT_DARK_CURRENT,
                         msg='Mean value for pmt dark current distribution function is expected to be I_dark * R_load.')
        self.assertEqual(std, numpy.sqrt(2 * self.INPUT_CONST.charge_electron * mean * self.INPUT_BANDWIDTH *
                                         self.INPUT_LOAD_RES),
                         msg='The expected STD for dark current generator function is a theoretical indicated value.')

    def test_apd_dark_noise_generator(self):
        noisy_signal = self.APD.dark_noise_generator(dark_current=self.INPUT_DARK_CURRENT,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     load_resistance=self.INPUT_LOAD_RES,
                                                     signal_size=self.INPUT_SIGNAL_2.size)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The Dark Current noise generator is expected to create a similar sized signal.')
        mean, std = self.APD._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                               bandwidth=self.INPUT_BANDWIDTH,
                                               load_resistance=self.INPUT_LOAD_RES)
        self.assertDistributionMean(noisy_signal, mean,
                                    msg='The PMT Dark Current Generator does not return expected mean value')
        self.assertDistributionStandardDeviation(noisy_signal, std,
                                                 msg='The PMT Dark Current Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')


class PMTGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    INPUT_VALUE_2 = 1E9
    INPUT_INSTANCE_2 = 100000
    INPUT_SIGNAL_2 = numpy.full(INPUT_INSTANCE_2, INPUT_VALUE_2)
    INPUT_SEED = 20
    INPUT_CONST = Constants()
    INPUT_DYNODE_NUMBER = 9
    INPUT_DYNODE_GAIN = 6
    INPUT_DARK_CURRENT = 1E-3
    INPUT_DARK_CURRENT_2 = 1E-6
    INPUT_FREQUENCY = 1E6
    INPUT_QE = 0.35
    INPUT_LOAD_RES = 1E5
    INPUT_BANDWIDTH = 2E6

    DEFAULT_PMT_PATH = 'detector/pmt_default.xml'
    EXPECTED_ATTRIBUTES = ['detector_temperature', 'dynode_number', 'dynode_gain', 'quantum_efficiency',
                           'signal_to_background', 'dark_current', 'bandwidth', 'sampling_frequency', 'load_resistance']

    def setUp(self):
        self.PMT = PMT(GetData(data_path_name=self.DEFAULT_PMT_PATH).data)

    def tearDown(self):
        del self.PMT

    def test_class_inheritance(self):
        self.assertIsInstance(self.PMT, Noise, msg='<PMT> class is expected to be a child of <Noise>.')

    def test_parameter_attributes(self):
        self.assertHasAttributes(self.PMT, self.EXPECTED_ATTRIBUTES, msg='<PMT> class does not initiate will all '
                                                                         'expected attributes.')

    def test_pmt_noiseless_transfer(self):
        detector_voltage = self.PMT._pmt_noiseless_transfer(signal=self.INPUT_SIGNAL)
        mean = self.INPUT_SIGNAL * (1 + 1 / self.PMT.signal_to_background) * self.PMT.quantum_efficiency * self.\
            INPUT_CONST.charge_electron * (self.PMT.dynode_gain ** self.PMT.dynode_number) + self.PMT.dark_current
        self.assertEqual(detector_voltage.all(), mean.all(),
                         msg='The PMT noiseless transfer function is expected to create a theoretical '
                         'indicated value.')

    def test_pmt_amplification(self):
        amplification = self.PMT._pmt_amplification(dynode_gain=self.INPUT_DYNODE_GAIN,
                                                    dynode_number=self.INPUT_DYNODE_NUMBER)
        self.assertEqual(amplification, self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER,
                         msg='The PMT amplification function needs to get back the dynode_gain ^ dynode_number value')

    def test_pmt_noise_amplification(self):
        noise_amplification = self.PMT._pmt_noise_amplification(dynode_gain=self.INPUT_DYNODE_GAIN)
        self.assertEqual(noise_amplification, self.INPUT_DYNODE_GAIN / (self.INPUT_DYNODE_GAIN - 1),
                         msg='The PMT noise amplification function needs to get back the dynode_gain / '
                             '(dynode_gain -1) value')

    def test_pmt_gaussian_net_gain(self):
        net_gain = self.PMT._pmt_gaussian_net_gain(dynode_gain=self.INPUT_DYNODE_GAIN,
                                                   dynode_number=self.INPUT_DYNODE_NUMBER)
        self.assertEqual(net_gain, 2 * self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER * self.INPUT_DYNODE_GAIN /
                         (self.INPUT_DYNODE_GAIN - 1),
                         msg='The PMT gaussian net gain function needs to get back 2*amplification*noise_amplification')

    def test_pmt_shot_noise_setup(self):
        mean, std = self.PMT._shot_noise_setup(signal=self.INPUT_SIGNAL,
                                               load_resistance=self.INPUT_LOAD_RES,
                                               bandwidth=self.INPUT_BANDWIDTH,
                                               amplification=self.PMT._pmt_amplification(self.INPUT_DYNODE_GAIN,
                                                                                         self.INPUT_DYNODE_NUMBER),
                                               noise_amplification=self.PMT._pmt_noise_amplification(self.
                                                                                                     INPUT_DYNODE_GAIN),
                                               quantum_efficiency=self.INPUT_QE,
                                               sampling_frequency=self.INPUT_FREQUENCY)
        self.assertEqual(mean.all(), (self.INPUT_SIGNAL * self.INPUT_FREQUENCY * self.INPUT_DYNODE_GAIN **
                                      self.INPUT_DYNODE_NUMBER * self.INPUT_QE *
                                      self.INPUT_CONST.charge_electron * self.INPUT_LOAD_RES).all(),
                         msg='Mean value for PMT shot noise generator is expected to be equal to theoretical indicated '
                             'values.')
        self.assertEqual(std.all(), (numpy.sqrt(2*self.INPUT_CONST.charge_electron * mean *
                                                (self.INPUT_DYNODE_GAIN/(self.INPUT_DYNODE_GAIN-1)) *
                                                self.INPUT_BANDWIDTH * self.INPUT_LOAD_RES *
                                                self.INPUT_DYNODE_GAIN**self.INPUT_DYNODE_NUMBER)).all(),
                         msg='Std value for PMT shot noise generator is expected to be equal to theoretical indicated '
                             'values.')

    def test_pmt_shot_noise_generator(self):
        noisy_signal = self.PMT.shot_noise_generator(signal=self.INPUT_SIGNAL,
                                                     load_resistance=self.INPUT_LOAD_RES,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     amplification=self.PMT._pmt_amplification(self.INPUT_DYNODE_GAIN,
                                                                                               self.INPUT_DYNODE_NUMBER
                                                                                               ),
                                                     noise_amplification=self.PMT._pmt_noise_amplification
                                                     (self.INPUT_DYNODE_GAIN),
                                                     quantum_efficiency=self.INPUT_QE,
                                                     sampling_frequency=self.INPUT_FREQUENCY)
        mean = (self.INPUT_SIGNAL * self.INPUT_FREQUENCY * self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER *
                self.INPUT_QE * self.INPUT_CONST.charge_electron * self.INPUT_LOAD_RES).mean()
        std = (numpy.sqrt(2 * self.INPUT_CONST.charge_electron * mean *
                          (self.INPUT_DYNODE_GAIN / (self.INPUT_DYNODE_GAIN-1)) * self.INPUT_BANDWIDTH *
                          self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER * self.INPUT_LOAD_RES)).mean()
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The function need to keep the shape of the array')
        self.assertDistributionMean(series=noisy_signal, reference_mean=mean,
                                    msg='The PMT shot noise generation function needs to create an array with a well '
                                        'defined mean')
        self.assertDistributionStandardDeviation(series=noisy_signal, reference_std=std,
                                                 msg='The PMT shot noise generation function needs to create an array '
                                                     'with a well defined std')

    def test_pmt_single_dynode_noise_generator(self):
        noisy_signal = self.PMT._pmt_dynode_noise_generator(signal=self.INPUT_SIGNAL,
                                                            dynode_number=1,
                                                            dynode_gain=self.INPUT_DYNODE_GAIN)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The PMT Dynode Noise Generator is expected to create a similar sized signal.')
        mean, std = (self.INPUT_SIGNAL * self.INPUT_DYNODE_GAIN).mean(), numpy.sqrt((self.INPUT_SIGNAL * self.
                                                                                     INPUT_DYNODE_GAIN).mean())
        self.assertDistributionMean(noisy_signal, mean,
                                    msg='The PMT Dynode Noise Generator does not return expected mean value')
        self.assertDistributionStandardDeviation(noisy_signal, std,
                                                 msg='The PMT Dynode Noise Generator is expected to create Poisson '
                                                     'distributions. The actual STD does not match.')

    def test_pmt_photo_cathode_electron_generation(self):
        emitted_electrons = self.PMT._pmt_photo_cathode_electron_generation(signal=self.INPUT_SIGNAL)
        mean = (self.INPUT_SIGNAL * self.PMT.quantum_efficiency).mean()
        std = (numpy.sqrt(self.INPUT_SIGNAL * self.PMT.quantum_efficiency)).mean()
        self.assertTupleEqual(self.INPUT_SIGNAL.shape, emitted_electrons.shape,
                              msg='PMT photo cathode electron generation function needs to create a similar sized '
                                  'array')
        self.assertDistributionMean(series=emitted_electrons, reference_mean=mean,
                                    msg='PMT photo cathode electron generation function needs to create a '
                                        'theoretically indicated mean')
        self.assertDistributionStandardDeviation(series=emitted_electrons, reference_std=std,
                                                 msg='PMT photo cathode electron generation function needs to create '
                                                     'Poisson distribution')

    def test_pmt_high_thermionic_dark_electron_generator(self):
        electron_generation = self.PMT._pmt_thermionic_dark_electron_generator(signal_length=self.INPUT_SIGNAL.shape,
                                                                               dark_current=self.INPUT_DARK_CURRENT,
                                                                               dynode_gain=self.INPUT_DYNODE_GAIN,
                                                                               dynode_number=self.INPUT_DYNODE_NUMBER,
                                                                               sampling_frequency=self.INPUT_FREQUENCY)
        mean = self.INPUT_DARK_CURRENT / (self.INPUT_FREQUENCY * self.INPUT_CONST.charge_electron *
                                          self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER)
        std = numpy.sqrt(self.INPUT_DARK_CURRENT / (self.INPUT_FREQUENCY * self.INPUT_CONST.charge_electron *
                                                    self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER))
        self.assertTupleEqual(self.INPUT_SIGNAL.shape, electron_generation.shape,
                              msg='The pmt thermionic dark electron generation function needs to keep the size of an '
                                  'array')
        self.assertDistributionMean(series=electron_generation, reference_mean=mean,
                                    msg='The pmt high thermionic dark electron generator function needs to create a '
                                        'theoretically indicated mean')
        self.assertDistributionStandardDeviation(series=electron_generation, reference_std=std,
                                                 msg='The pmt high thermionic dark electron generator function needs'
                                                     ' to create Poisson distribution')

    def test_pmt_low_thermionic_dark_electron_generator(self):
        electron_generation_2 = self.PMT._pmt_thermionic_dark_electron_generator(signal_length=self.INPUT_SIGNAL.size,
                                                                                 dark_current=self.INPUT_DARK_CURRENT_2,
                                                                                 dynode_gain=self.INPUT_DYNODE_GAIN,
                                                                                 dynode_number=self.INPUT_DYNODE_NUMBER,
                                                                                 sampling_frequency=self.INPUT_FREQUENCY
                                                                                 )
        for index in range(self.INPUT_SIGNAL.size):
            if electron_generation_2[index] != 0 and electron_generation_2[index] != 1:
                return False, 'The pmt low thermionic dark electron generator function needs ' \
                              'to create zeros or ones'

        mean = self.INPUT_DARK_CURRENT_2 / (self.INPUT_FREQUENCY * self.INPUT_CONST.charge_electron *
                                            self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER)
        self.assertDistributionMean(series=electron_generation_2, reference_mean=mean,
                                    msg='The pmt low thermionic dark electron generator function needs to create the '
                                        'theoretically indicated mean')

    def test_pmt_dark_noise_setup(self):
        mean, std = self.PMT._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                               bandwidth=self.INPUT_BANDWIDTH,
                                               load_resistance=self.INPUT_LOAD_RES,
                                               net_gain=self.PMT._pmt_gaussian_net_gain(self.INPUT_DYNODE_GAIN,
                                                                                        self.INPUT_DYNODE_NUMBER))
        self.assertEqual(mean, self.INPUT_LOAD_RES * self.INPUT_DARK_CURRENT,
                         msg='Mean value for pmt dark current distribution function is expected to be I_dark * R_load.')
        self.assertAlmostEqual(std, numpy.sqrt(4 * self.INPUT_CONST.charge_electron * mean * self.INPUT_DYNODE_GAIN **
                                               self.INPUT_DYNODE_NUMBER * self.INPUT_DYNODE_GAIN /
                                               (self.INPUT_DYNODE_GAIN - 1) * self.INPUT_BANDWIDTH *
                                               self.INPUT_LOAD_RES), places=10,
                               msg='The expected STD for dark current generator function is a theoretical '
                                   'indicated value.')

    def test_pmt_dark_noise_generator(self):
        noisy_signal = self.PMT.dark_noise_generator(dark_current=self.INPUT_DARK_CURRENT,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     load_resistance=self.INPUT_LOAD_RES,
                                                     signal_size=self.INPUT_SIGNAL_2.size,
                                                     net_gain=self.PMT._pmt_gaussian_net_gain(self.INPUT_DYNODE_GAIN,
                                                                                              self.INPUT_DYNODE_NUMBER))
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The Dark Current noise generator is expected to create a similar sized signal.')
        mean, std = self.PMT._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                               bandwidth=self.INPUT_BANDWIDTH,
                                               load_resistance=self.INPUT_LOAD_RES,
                                               net_gain=self.PMT._pmt_gaussian_net_gain(self.INPUT_DYNODE_GAIN,
                                                                                        self.INPUT_DYNODE_NUMBER))
        self.assertDistributionMean(noisy_signal, mean,
                                    msg='The PMT Dark Current Generator does not return expected mean value')
        self.assertDistributionStandardDeviation(noisy_signal, std,
                                                 msg='The PMT Dark Current Generator is expected to create Normal '
                                                     'distributions. The actual STD does not match.')


class PPDGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    INPUT_CONST = Constants()

    DEFAULT_PPD_PATH = 'detector/ppd_default.xml'
    EXPECTED_ATTRIBUTES = ['detector_temperature', 'quantum_efficiency', 'bandwidth', 'dark_current',
                           'load_resistance', 'load_capacity', 'voltage_noise', 'internal_capacity',
                           'sampling_frequency', 'signal_to_background']

    def setUp(self):
        self.PPD = PPD(GetData(data_path_name=self.DEFAULT_PPD_PATH).data)

    def tearDown(self):
        del self.PPD

    def test_class_inheritance(self):
        self.assertIsInstance(self.PPD, Noise, msg='<PPD> class is expected to be a child of <Noise>.')

    def test_parameter_attributes(self):
        self.assertHasAttributes(self.PPD, self.EXPECTED_ATTRIBUTES, msg='<PPD> class does not initiate will all '
                                                                         'expected attributes.')

    def test_ppd_noiseless_transfer(self):
        detector_voltage = self.PPD._ppd_noiseless_transfer(signal=self.INPUT_SIGNAL)
        mean = (self.INPUT_SIGNAL * (1 + 1 / self.PPD.signal_to_background) * self.PPD.quantum_efficiency *
                self.INPUT_CONST.charge_electron + self.PPD.dark_current) * self.PPD.load_resistance
        self.assertEqual(detector_voltage.all(), mean.all(),
                         msg='The PPD noiseless transfer function is expected to create a theoretical '
                         'indicated value.')


class MPPCGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    INPUT_CONST = Constants()

    DEFAULT_MPPC_PATH = 'detector/mppc_default.xml'
    EXPECTED_ATTRIBUTES = ['detector_gain', 'quantum_efficiency', 'bandwidth', 'dark_count_rate',
                           'sampling_frequency', 'signal_to_background', 'photon_detection_efficiency',
                           'terminal_capacitance', 'total_pixel_count', 'quenching_resistance']

    def setUp(self):
        self.MPPC = MPPC(GetData(data_path_name=self.DEFAULT_MPPC_PATH).data)

    def tearDown(self):
        del self.MPPC

    def test_class_inheritance(self):
        self.assertIsInstance(self.MPPC, Noise, msg='<MPPC> class is expected to be a child of <Noise>.')

    def test_parameter_attributes(self):
        self.assertHasAttributes(self.MPPC, self.EXPECTED_ATTRIBUTES, msg='<MPPC> class does not initiate will all '
                                                                          'expected attributes.')


class DetectorGeneratorTest(NoiseBasicTestCase):

    INPUT_APD_TYPE = 'apd'
    INPUT_PMT_TYPE = 'pmt'
    INPUT_PPD_TYPE = 'ppd'
    INPUT_MPPC_TYPE = 'mppc'
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

    def test_MPPC_instantiation(self):
        actual = Detector(detector_type=self.INPUT_MPPC_TYPE)
        self.assertIsInstance(actual, MPPC, msg='<Detector> class is expected to return a <MPPC> class.')

    def test_exception_handling(self):
        with self.assertRaises(ValueError, msg='ValueError raising is expected in case of wrong str input.'):
            actual = Detector(detector_type=self.INPUT_VAL_ERROR_1)

        with self.assertRaises(AssertionError, msg='AssertionError raising is expected in case '
                                                   'of wrong detector type input.'):
            actual = Detector(detector_type=self.INPUT_VAL_ERROR_2)


class NoiseRegressionTest(NoiseBasicTestCase):

    INPUT_APD = 'apd'
    INPUT_PMT = 'pmt'
    INPUT_PPD = 'ppd'

    INPUT_GAUSSIAN_NOISE = 'gaussian'
    INPUT_DETALIED_NOISE = 'detailed'

    def setUp(self):
        self.path = 'test_dataset/noise_regressiontests/actual/'

    def tearDown(self):
        public_folder = os.path.join(os.getcwd(), 'data', 'dummy', 'test_dataset')
        private_folder = os.path.join(os.getcwd(), 'data', 'test_dataset')
        access = AccessData(None)
        if access.private_key is None:
            rmtree(public_folder)
        else:
            rmtree(private_folder)

    def test_APD_Gaussian_regression(self):
        detector = Detector(detector_type=self.INPUT_APD,
                            data_path=self.path + 'test_' + self.INPUT_APD + '_detector.xml')
        signal_data = GetData(data_path_name=self.path + 'test_' + self.INPUT_APD + '_' +
                              self.INPUT_GAUSSIAN_NOISE + '_detector.h5', data_key=['signals']).data
        detector.seed(signal_data['seed'][0])
        actual = detector.add_noise_to_signal(signal_data['expected_signal'], noise_type=self.INPUT_GAUSSIAN_NOISE)
        self.assertAlmostEqual(actual[0], signal_data['noisy_signal'][0], msg='The values are not equal.',places=8)

    def test_PMT_Gaussian_regression(self):
        detector = Detector(detector_type=self.INPUT_PMT,
                            data_path=self.path + 'test_' + self.INPUT_PMT + '_detector.xml')
        signal_data = GetData(data_path_name=self.path + 'test_' + self.INPUT_PMT + '_' +
                              self.INPUT_GAUSSIAN_NOISE + '_detector.h5', data_key=['signals']).data
        detector.seed(signal_data['seed'][0])
        actual = detector.add_noise_to_signal(signal_data['expected_signal'], noise_type=self.INPUT_GAUSSIAN_NOISE)
        self.assertAlmostEqual(actual[0], signal_data['noisy_signal'][0], msg='The values are not equal.', places=8)

    def test_PMT_Detailed_regression(self):
        detector = Detector(detector_type=self.INPUT_PMT,
                            data_path=self.path + 'test_' + self.INPUT_PMT + '_detector.xml')
        signal_data = GetData(data_path_name=self.path + 'test_' + self.INPUT_PMT + '_' +
                              self.INPUT_DETALIED_NOISE + '_detector.h5', data_key=['signals']).data
        detector.seed(signal_data['seed'][0])
        actual = detector.add_noise_to_signal(signal_data['expected_signal'], noise_type=self.INPUT_DETALIED_NOISE)
        self.assertAlmostEqual(actual[0], signal_data['noisy_signal'][0], msg='The values are not equal.', places=8)

    def test_PPD_Gaussian_regression(self):
        detector = Detector(detector_type=self.INPUT_PPD,
                            data_path=self.path + 'test_' + self.INPUT_PPD + '_detector.xml')
        signal_data = GetData(data_path_name=self.path + 'test_' + self.INPUT_PPD + '_' +
                              self.INPUT_GAUSSIAN_NOISE + '_detector.h5', data_key=['signals']).data
        detector.seed(signal_data['seed'][0])
        actual = detector.add_noise_to_signal(signal_data['expected_signal'], noise_type=self.INPUT_GAUSSIAN_NOISE)
        self.assertAlmostEqual(actual[0], signal_data['noisy_signal'][0], msg='The values are not equal.', places=8)
