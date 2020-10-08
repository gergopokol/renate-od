import numpy
import pandas
import utility.getdata as ut
from utility.constants import Constants
from numpy.random import RandomState
from lxml import etree


class SynthSignals:
    def __init__(self, frequency=1e6, synthetic_signals=None, data_path='signals/test.txt'):
        if isinstance(frequency, float):
            self.frequency = frequency
        else:
            raise TypeError('Expected data type for <frequency> is <int>.')
        self._check_synthetic_signals(synthetic_signals, data_path)

    def _check_synthetic_signals(self, synthetic_signals, path):
        if synthetic_signals is None:
            self._load_synthetic_signals(path)
        elif isinstance(synthetic_signals, pandas.DataFrame) or isinstance(synthetic_signals, pandas.Series):
            self.signals = synthetic_signals
        else:
            raise TypeError('The expected data type for <synthetic signals> is <pandas.DataFrame or pandas.Series>')

    def _load_synthetic_signals(self, path):
        self.signals = ut.GetData(data_path_name=path, data_format="array").data


class Noise(RandomState):
    def __init__(self, seed=None):
        RandomState.__init__(self, seed)
        self.constants = Constants()

    def signal_size(self, signal):
        size = signal.size
        return size

    def _photonflux_to_photnnumber(self, signal, sampling_frequency):
        photonnumber_signal = signal / sampling_frequency
        return photonnumber_signal

    def background_noise_generator(self, signal, signal_to_background):
        background = signal / signal_to_background
        return background

    def signal_preparation(self, signal, sampling_frequency):
        prepared_signal = self._photonflux_to_photnnumber(signal, sampling_frequency)
        return prepared_signal

    def detector_transfer(self, signal, detector_gain, quantum_efficiency, load_resistance):
        detector_voltage = signal * self.constants.charge_electron * detector_gain * quantum_efficiency * \
                           load_resistance
        return detector_voltage

    def photon_noise_generator(self, signal):
        signal = self.poisson(signal)
        return signal

    def _shot_noise_setup(self, signal, detector_gain, load_resistance, noise_index, bandwidth,
                          expected_value=0):
        expected_value = signal
        variance = numpy.array(numpy.sqrt(2 * self.constants.charge_electron * detector_gain *
                                          numpy.power(detector_gain, noise_index) * bandwidth) * load_resistance *
                               numpy.sqrt(signal))
        return expected_value, variance

    def shot_noise_generator(self, signal, detector_gain, load_resistance, noise_index, bandwidth, quantum_efficiency):
        expected_value, variance = self._shot_noise_setup(signal, detector_gain, load_resistance, noise_index,
                                                          bandwidth, quantum_efficiency)
        noised_signal = self.normal(expected_value, variance)
        return noised_signal

    def _johnson_noise_setup(self, detector_temperature, bandwidth, load_resistance, expected_value=0):
        variance = numpy.sqrt(4 * self.constants.Boltzmann * detector_temperature * bandwidth * load_resistance)
        return expected_value, variance

    def johnson_noise_generator(self, detector_temperature, bandwidth, load_resistance, signal_size):
        expected_value, variance = self._johnson_noise_setup(detector_temperature, bandwidth, load_resistance)
        johnson_noise = self.normal(expected_value, variance, signal_size)
        return johnson_noise

    def _voltage_noise_setup(self, voltage_noise, load_resistance, load_capacity, internal_capacity, expected_value=0):
        variance = voltage_noise * numpy.sqrt(1 / (2 * numpy.pi * load_resistance *
                                                    (load_capacity + internal_capacity))) + voltage_noise * \
                    (1 + internal_capacity / load_capacity) * numpy.sqrt(1.57 * 1 / (2 * numpy.pi * load_resistance *
                                                                                       load_capacity))
        return expected_value, variance

    def voltage_noise_generator(self, voltage_noise, load_resistance, load_capacity, internal_capacity, signal_size):
        expected_value, variance = self._voltage_noise_setup(voltage_noise, load_resistance, load_capacity,
                                                             internal_capacity)
        noise = self.normal(expected_value, variance, signal_size)
        return noise

    def _dark_noise_setup(self, dark_current, bandwidth, load_resistance, expected_value=0):
        variance = numpy.sqrt(2 * self.constants.charge_electron * dark_current * bandwidth) * load_resistance
        expected_value = dark_current * load_resistance
        return expected_value, variance

    def dark_noise_generator(self, dark_current, bandwidth, load_resistance, signal_size):
        expected_value, variance = self._dark_noise_setup(dark_current, bandwidth, load_resistance)
        noise = self.normal(expected_value, variance, signal_size)
        return noise


class APD(Noise):

    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        assert isinstance(detector_parameters, etree._ElementTree), 'Expected data type for <detector_parameters> ' \
                                                                    'is etree._ElementTree.'
        assert detector_parameters.getroot().find('head').find('type').text == 'apd', \
            'The detector type to be set is APD. Please check input data.'
        self.detector_temperature = float(detector_parameters.getroot().find('body').find('temperature').text)
        self.detector_gain = float(detector_parameters.getroot().find('body').find('gain').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.noise_index = float(detector_parameters.getroot().find('body').find('noise_index').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.dark_current = float(detector_parameters.getroot().find('body').find('dark_current').text)
        self.load_resistance = float(detector_parameters.getroot().find('body').find('load_resistance').text)
        self.load_capacity = float(detector_parameters.getroot().find('body').find('load_capacity').text)
        self.voltage_noise = float(detector_parameters.getroot().find('body').find('voltage_noise').text)
        self.internal_capacity = float(detector_parameters.getroot().find('body').find('internal_capacity').text)
        self.sampling_frequency = float(detector_parameters.getroot().find('body').find('sampling_frequency').text)
        self.signal_to_background = float(detector_parameters.getroot().find('body').find('signal_to_background').text)

    def add_noise_to_signal(self, signal):
        size = self.signal_size(signal)
        prepared_signal = self.signal_preparation(signal, self.sampling_frequency)
        background = self.background_noise_generator(prepared_signal, self.signal_to_background)
        background_noised_signal = prepared_signal + background
        detector_voltage = self.detector_transfer(background_noised_signal, self.detector_gain, self.quantum_efficiency,
                                                  self.load_resistance)
        shot_noised_signal = self.shot_noise_generator(detector_voltage, self.detector_gain, self.load_resistance,
                                                       self.noise_index, self.bandwidth, self.quantum_efficiency)
        shot_noise = shot_noised_signal - detector_voltage
        dark_noise = self.dark_noise_generator(self.dark_current, self.bandwidth, self.load_resistance, size)
        voltage_noise = self.voltage_noise_generator(self.voltage_noise, self.load_resistance,
                                                     self.load_capacity, self.internal_capacity, size)
        johnson_noise = self.johnson_noise_generator(self.detector_temperature, self.bandwidth, self.load_resistance,
                                                     size)
        noised_signal = shot_noised_signal + dark_noise + voltage_noise + johnson_noise
        return noised_signal


class PMT(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        assert isinstance(detector_parameters, etree._ElementTree), 'Expected data type for <detector_parameters> ' \
                                                                    'is etree._ElementTree.'
        assert detector_parameters.getroot().find('head').find('type').text == 'pmt', \
            'The detector type to be set is PMT. Please check input data.'
        self.detector_temperature = float(detector_parameters.getroot().find('body').find('temperature').text)
        self.dynode_number = float(detector_parameters.getroot().find('body').find('dynode_number').text)
        self.dynode_gain = float(detector_parameters.getroot().find('body').find('dynode_gain').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.dark_current = float(detector_parameters.getroot().find('body').find('dark_current').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.sampling_frequency = float(detector_parameters.getroot().find('body').find('sampling_frequency').text)

    def add_noise_to_signal(self):
        pass


class PP(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        pass

    def add_noise_to_signal(self):
        pass


class Detector(APD, PMT, PP):
    def __init__(self, detector_type='apd', parameters=None, data_path=None):
        assert isinstance(detector_type, str), 'Expected data type for <detector_type> is str.'
        self.detector_type = detector_type
        if self.detector_type is 'apd':
            APD.__init__(self, self.__get_detector_parameters(parameters, data_path))
        elif self.detector_type is 'pmt':
            PMT.__init__(self, self.__get_detector_parameters(parameters, data_path))
        elif self.detector_type is 'pp':
            PP.__init__(self, self.__get_detector_parameters(parameters, data_path))
        else:
            raise ValueError('The requested detector type:' + self.detector_type + ' is not yet supported')

    def __get_detector_parameters(self, parameters, data_path):
        if isinstance(parameters, etree._ElementTree):
            self.data_path = 'From external workflow.'
        elif data_path is None:
            self.data_path = 'detector/'+self.detector_type+'_default.xml'
            parameters = ut.GetData(data_path_name=self.data_path).data
        elif isinstance(data_path, str):
            self.data_path = data_path
            parameters = ut.GetData(data_path_name=self.data_path).data
        else:
            raise ValueError('Expected data type for <data_path> is str or None, '
                             'in which case default values will be loaded')
        assert isinstance(parameters, etree._ElementTree), 'Expected data type for parameters is etree._ElementTree.'
        return parameters
