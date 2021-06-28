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
    INPUT_GAIN = 2
    INPUT_NOISE_INDEX = 1
    INPUT_DET_TEMP = 300
    INPUT_DARK_CURRENT = 10
    INPUT_VOLTAGE_NOISE = 10
    INPUT_LOAD_CAPACITY = 5
    INPUT_INTERNAL_CAPACITY = 8
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

    def test_photon_flux_to_detector_voltage(self):
        actual_signal = self.noise_gen.photon_flux_to_detector_voltage(self.INPUT_SIGNAL, self.INPUT_GAIN,
                                                                       self.INPUT_QE, self.INPUT_LOAD_RESIST,
                                                                       self.INPUT_FREQUENCY)
        reference_detector_voltage = self.INPUT_PHOTON_FLUX * self.INPUT_LOAD_RESIST * self.INPUT_QE * self.\
            INPUT_GAIN * self.INPUT_CONST.charge_electron * self.INPUT_FREQUENCY
        self.assertTupleEqual(actual_signal.shape, self.INPUT_SIGNAL.shape,
                              msg='The detector voltage converter is not expected to change the output signal shape.')
        self.assertAlmostEqual(actual_signal.mean(), reference_detector_voltage, places=self.EXPECTED_PRECISION_4,
                               msg='The detector voltage converter is expected to be <signal*e*M*QE*R_L>')

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
        mean, std = self.noise_gen._shot_noise_setup(self.INPUT_SIGNAL_2, self.INPUT_GAIN, self.INPUT_LOAD_RESIST,
                                                     self.INPUT_NOISE_INDEX, self.INPUT_BANDWIDTH)
        self.assertTupleEqual(mean.shape, self.INPUT_SIGNAL_2.shape, msg='The mean values for shot signal noise '
                              'generation is expected to have the same shape.')
        self.assertTupleEqual(std.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The std values for shot noise generation is expected to have the same shape.')
        for index in range(self.INPUT_INSTANCE):
            self.assertEqual(mean[index], self.INPUT_SIGNAL_2[index], msg='Mean value for noise generator is expected '
                                                                          'to be equal to input signal values.')
            self.assertEqual(std[index], numpy.sqrt(2 * self.INPUT_CONST.charge_electron * self.INPUT_LOAD_RESIST *
                             self.INPUT_BANDWIDTH * numpy.power(self.INPUT_GAIN, self.INPUT_NOISE_INDEX+1) *
                             self.INPUT_SIGNAL_2[index]), msg='Std values for noise generator is '
                                                              'expected to be equal to sqrt(2*q*B*G^(x+1)*signal*R)')

    def test_shot_noise_generator(self):
        noisy_signal = self.noise_gen.shot_noise_generator(self.INPUT_SIGNAL_2, detector_gain=self.INPUT_GAIN,
                                                           load_resistance=self.INPUT_LOAD_RESIST,
                                                           noise_index=self.INPUT_NOISE_INDEX,
                                                           bandwidth=self.INPUT_BANDWIDTH)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The shot noise generator routine is not expected to change the signal shape.')
        mean, std = self.noise_gen._shot_noise_setup(self.INPUT_SIGNAL_2, detector_gain=self.INPUT_GAIN,
                                                     load_resistance=self.INPUT_LOAD_RESIST,
                                                     noise_index=self.INPUT_NOISE_INDEX,
                                                     bandwidth=self.INPUT_BANDWIDTH)
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
                                                     load_resistance=self.INPUT_LOAD_RESIST)
        self.assertEqual(mean, self.INPUT_LOAD_RESIST * self.INPUT_DARK_CURRENT,
                         msg='Mean value for dark current distribution function is expected to be I_dark * R_load.')
        self.assertEqual(std, self.INPUT_LOAD_RESIST * numpy.sqrt(2 * self.INPUT_CONST.charge_electron *
                         self.INPUT_BANDWIDTH * self.INPUT_DARK_CURRENT),
                         msg='The expected STD for dark current generator function is R_load*sqrt(2*I_dark*B*q).')

    def test_dark_noise_generator(self):
        noisy_signal = self.noise_gen.dark_noise_generator(dark_current=self.INPUT_DARK_CURRENT,
                                                           bandwidth=self.INPUT_BANDWIDTH,
                                                           load_resistance=self.INPUT_LOAD_RESIST,
                                                           signal_size=self.INPUT_SIGNAL_2.shape[0])
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The Dark Current noise generator is expected to create a similar sized signal.')
        mean, std = self.noise_gen._dark_noise_setup(dark_current=self.INPUT_DARK_CURRENT,
                                                     bandwidth=self.INPUT_BANDWIDTH,
                                                     load_resistance=self.INPUT_LOAD_RESIST)
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

    def test_apd_add_noise_to_signal(self):
        self.APD.seed(self.INPUT_SEED)
        noisy_signal = self.APD.add_noise_to_signal(self.INPUT_SIGNAL_2)
        default = numpy.loadtxt('detector/test/APD_test_Gaussian.txt')
        for index in range(self.INPUT_INSTANCE_2):
            if noisy_signal[index] != default[index]:
                return False, 'The add noise to signal function cannot create the seeded values'


class PMTGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    INPUT_VALUE_2 = 1E9
    INPUT_INSTANCE_2 = 1000000
    INPUT_SIGNAL_2 = numpy.full(INPUT_INSTANCE_2, INPUT_VALUE_2)
    INPUT_VALUE_3 = 1E9
    INPUT_INSTANCE_3 = 100000
    INPUT_SIGNAL_3 = numpy.full(INPUT_INSTANCE_3, INPUT_VALUE_3)
    INPUT_SEED = 20
    INPUT_CONST = Constants()
    INPUT_DYNODE_NUMBER = 9
    INPUT_DYNODE_GAIN = 6
    INPUT_DARK_CURRENT = 10
    INPUT_FREQUENCY = 1E6

    DEFAULT_PMT_PATH = 'detector/pmt_default.xml'

    def setUp(self):
        self.PMT = PMT(GetData(data_path_name=self.DEFAULT_PMT_PATH).data)

    def tearDown(self):
        del self.PMT

    def test_class_inheritance(self):
        self.assertIsInstance(self.PMT, Noise, msg='<PMT> class is expected to be a child of <Noise>.')

    def test_pmt_noiseless_transfer(self):
        detector_voltage = self.PMT._pmt_noiseless_transfer(signal=self.INPUT_SIGNAL)
        mean = self.INPUT_SIGNAL * (1 + 1 / self.PMT.signal_to_background) * self.PMT.quantum_efficiency * self.\
            INPUT_CONST.charge_electron * (self.PMT.dynode_gain ** self.PMT.dynode_number) + self.PMT.dark_current
        self.assertEqual(detector_voltage.all(), mean.all(),
                         msg='The PMT noiseless transfer function is expected to create a theoretical '
                         'indicated value.')

    def test_pmt_shot_noise_setup(self):
        mean, std = self.PMT._pmt_shot_noise_setup(signal=self.INPUT_SIGNAL)
        self.assertEqual(mean.all(), (self.INPUT_SIGNAL * self.PMT.sampling_frequency * self.PMT.dynode_gain **
                                      self.PMT.dynode_number * self.PMT.quantum_efficiency *
                                      self.INPUT_CONST.charge_electron).all(),
                         msg='Mean value for noise generator is expected to be equal to input signal values.')
        self.assertEqual(std.all(), (numpy.sqrt(2*self.INPUT_CONST.charge_electron*mean *
                                                (self.PMT.dynode_gain/(self.PMT.dynode_gain-1)) * self.PMT.bandwidth *
                                                self.PMT.dynode_gain**self.PMT.dynode_number)).all(),
                         msg='Std value for noise generator is expected to be equal to input signal values.')

    def test_pmt_shot_noise_generation(self):
        noisy_signal = self.PMT._pmt_shot_noise_generation(signal=self.INPUT_SIGNAL)
        mean = (self.INPUT_SIGNAL * self.PMT.sampling_frequency * self.PMT.dynode_gain ** self.PMT.dynode_number *
                self.PMT.quantum_efficiency * self.INPUT_CONST.charge_electron).mean()
        std = (numpy.sqrt(2*self.INPUT_CONST.charge_electron*mean*(self.PMT.dynode_gain/(self.PMT.dynode_gain-1)) *
                          self.PMT.bandwidth*self.PMT.dynode_gain**self.PMT.dynode_number)).mean()
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

    def test_pmt_dark_noise_generation(self):
        noisy_signal = self.PMT._pmt_dark_noise_generation(signal=self.INPUT_SIGNAL_2)
        self.assertTupleEqual(noisy_signal.shape, self.INPUT_SIGNAL_2.shape,
                              msg='The PMT Dark Noise Generation is expected to create a similar sized signal.')
        self.assertDistributionMean(noisy_signal, self.PMT.dark_current, precision=1.5E-01,
                                    msg='The PMT Dark Noise Generation does not return expected mean value')
        self.assertDistributionStandardDeviation(noisy_signal, numpy.sqrt(4*self.INPUT_CONST.charge_electron *
                                                 self.PMT.dark_current*self.PMT.dynode_gain ** self.PMT.dynode_number
                                                 * (self.PMT.dynode_gain/(self.PMT.dynode_gain-1)) *
                                                 self.PMT.bandwidth),
                                                 msg='The PMT Dark Noise Generation does not return '
                                                     'expected std value')

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
        dark_current_1 = 1E-3
        electron_generation = self.PMT._pmt_thermionic_dark_electron_generator(signal_length=self.INPUT_SIGNAL.shape,
                                                                               dark_current=dark_current_1,
                                                                               dynode_gain=self.INPUT_DYNODE_GAIN,
                                                                               dynode_number=self.INPUT_DYNODE_NUMBER,
                                                                               sampling_frequency=self.INPUT_FREQUENCY)
        mean = dark_current_1 / (self.INPUT_FREQUENCY * self.INPUT_CONST.charge_electron *
                                 self.INPUT_DYNODE_GAIN ** self.INPUT_DYNODE_NUMBER)
        std = numpy.sqrt(dark_current_1 / (self.INPUT_FREQUENCY * self.INPUT_CONST.charge_electron *
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
        dark_current_2 = 1E-6
        electron_generation_2 = self.PMT._pmt_thermionic_dark_electron_generator(signal_length=self.INPUT_SIGNAL.shape,
                                                                                 dark_current=dark_current_2,
                                                                                 dynode_gain=self.INPUT_DYNODE_GAIN,
                                                                                 dynode_number=self.INPUT_DYNODE_NUMBER,
                                                                                 sampling_frequency=self.INPUT_FREQUENCY
                                                                                 )
        for index in range(self.INPUT_SIGNAL.size):
            if electron_generation_2[index] != 0:
                return False, 'The pmt low thermionic dark electron generator function needs ' \
                              'to create zeros or ones'

    def test_pmt_detailed_noise_generator(self):
        self.PMT.seed(self.INPUT_SEED)
        noisy_signal = self.PMT.add_noise_to_signal(self.INPUT_SIGNAL_3, noise_type='detailed')
        default = numpy.loadtxt('detector/test/PMT_test_Detailed.txt')
        for index in range(self.INPUT_INSTANCE_2):
            if noisy_signal[index] != default[index]:
                return False, 'The PMT detailed noise generator function cannot create the seeded values'

    def test_pmt_gaussian_noise_generator(self):
        self.PMT.seed(self.INPUT_SEED)
        noisy_signal = self.PMT.add_noise_to_signal(self.INPUT_SIGNAL_2, noise_type='gaussian')
        default = numpy.loadtxt('detector/test/PMT_test_Gaussian.txt')
        for index in range(self.INPUT_INSTANCE_2):
            if noisy_signal[index] != default[index]:
                return False, 'The PMT gaussian noise generator function cannot create the seeded values'


class PPDGeneratorTest(NoiseBasicTestCase):

    INPUT_VALUE = 1000
    INPUT_INSTANCE = 1000000
    INPUT_SIGNAL = numpy.full(INPUT_INSTANCE, INPUT_VALUE)
    INPUT_CONST = Constants()

    DEFAULT_PPD_PATH = 'detector/ppd_default.xml'

    def setUp(self):
        self.PPD = PPD(GetData(data_path_name=self.DEFAULT_PPD_PATH).data)

    def tearDown(self):
        del self.PPD

    def test_class_inheritance(self):
        self.assertIsInstance(self.PPD, Noise, msg='<PPD> class is expected to be a child of <Noise>.')

    def test_ppd_noiseless_transfer(self):
        detector_voltage = self.PPD._ppd_noiseless_transfer(signal=self.INPUT_SIGNAL)
        mean = (self.INPUT_SIGNAL * (1 + 1 / self.PPD.signal_to_background) * self.PPD.quantum_efficiency *
                self.INPUT_CONST.charge_electron + self.PPD.dark_current) * self.PPD.load_resistance
        self.assertEqual(detector_voltage.all(), mean.all(),
                         msg='The PPD noiseless transfer function is expected to create a theoretical '
                         'indicated value.')


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
