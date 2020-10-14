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

    @staticmethod
    def signal_length(signal):
        return signal.size

    @staticmethod
    def _photon_flux_to_photon_number(signal, sampling_frequency):
        return signal / sampling_frequency

    @staticmethod
    def background_addition(signal, signal_to_background):
        return signal + signal.mean() / signal_to_background

    def photon_flux_to_detector_voltage(self, signal, detector_gain, quantum_efficiency, load_resistance):
        return signal * self.constants.charge_electron * detector_gain * quantum_efficiency * load_resistance

    def generate_photon_noise(self, signal):
        return self.poisson(signal)

    def _shot_noise_setup(self, signal, detector_gain, load_resistance, noise_index, bandwidth):
        """
        :return: mean (I_det) and STD (R_l * sqrt(2*q*I_det*M*F*B)), where F = M exp(x)
        """
        return signal, numpy.array(numpy.sqrt(2 * self.constants.charge_electron * signal *
                                   numpy.power(detector_gain, noise_index + 1) * bandwidth) * load_resistance)

    def shot_noise_generator(self, signal, detector_gain, load_resistance, noise_index, bandwidth):
        expected_value, variance = self._shot_noise_setup(signal, detector_gain,
                                                          load_resistance, noise_index, bandwidth)
        return self.normal(expected_value, variance)

    def _johnson_noise_setup(self, detector_temperature, bandwidth, load_resistance, expected_value=0):
        """
        :return: mean (0) and STD (sqrt(4kBTR_l))
        """
        return expected_value, numpy.sqrt(4 * self.constants.Boltzmann * detector_temperature *
                                          bandwidth * load_resistance)

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
        size = self.signal_length(signal)
        prepared_signal = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        background_noised_signal = self.background_addition(prepared_signal, self.signal_to_background)
        detector_voltage = self.photon_flux_to_detector_voltage(background_noised_signal, self.detector_gain,
                                                                self.quantum_efficiency, self.load_resistance)
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
        self.dynode_number = int(detector_parameters.getroot().find('body').find('dynode_number').text)
        self.dynode_gain = float(detector_parameters.getroot().find('body').find('dynode_gain').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.signal_to_background = float(detector_parameters.getroot().find('body').find('signal_to_background').text)
        self.dark_current = float(detector_parameters.getroot().find('body').find('dark_current').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.sampling_frequency = float(detector_parameters.getroot().find('body').find('sampling_frequency').text)

    def _dynode_noise_generator(self, signal):
        for i in range(0, self.dynode_number):
            signal = self.normal(signal * self.dynode_gain, numpy.sqrt(signal * self.dynode_gain))
        return signal

    def _pmt_dark_noise_generator(self, signal_size):
        expected_value = self.dark_current / (self.sampling_frequency * self.constants.charge_electron)
        dark_electrons = self.poisson(expected_value, signal_size)
        return dark_electrons

    def _pmt_transfer(self, signal):
        emitted_electrons = self.poisson(signal * self.quantum_efficiency)
        return emitted_electrons

    def add_noise_to_signal(self, signal):
        size = self.signal_length(signal)
        prepared_signal = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        emitted_photons = self.generate_photon_noise(prepared_signal)
        background_noised_signal = self.background_addition(emitted_photons, self.signal_to_background)
        emitted_electrons = self._pmt_transfer(background_noised_signal)
        dark_electrons = self._pmt_dark_noise_generator(size)
        primary_electrons = emitted_electrons + dark_electrons
        secondary_electrons = self._dynode_noise_generator(primary_electrons)
        noised_signal = secondary_electrons * self.constants.charge_electron * self.sampling_frequency
        return noised_signal


class PPD(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        assert isinstance(detector_parameters, etree._ElementTree), 'Expected data type for <detector_parameters> ' \
                                                                    'is etree._ElementTree.'
        assert detector_parameters.getroot().find('head').find('type').text == 'ppd', \
            'The detector type to be set is PPD. Please check input data.'
        self.detector_temperature = float(detector_parameters.getroot().find('body').find('temperature').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.dark_current = float(detector_parameters.getroot().find('body').find('dark_current').text)
        self.load_resistance = float(detector_parameters.getroot().find('body').find('load_resistance').text)
        self.load_capacity = float(detector_parameters.getroot().find('body').find('load_capacity').text)
        self.voltage_noise = float(detector_parameters.getroot().find('body').find('voltage_noise').text)
        self.internal_capacity = float(detector_parameters.getroot().find('body').find('internal_capacity').text)
        self.sampling_frequency = float(detector_parameters.getroot().find('body').find('sampling_frequency').text)
        self.signal_to_background = float(detector_parameters.getroot().find('body').find('signal_to_background').text)

    def _ppd_transfer(self, signal):
        detector_current = self.poisson(signal * self.quantum_efficiency) * self.constants.charge_electron * \
                           self.sampling_frequency
        return detector_current

    def add_noise_to_signal(self, signal):
        size = self.signal_length(signal)
        prepared_signal = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        emitted_photons = self.generate_photon_noise(prepared_signal)
        background_noised_signal = self.background_addition(emitted_photons, self.signal_to_background)
        detector_current = self._ppd_transfer(background_noised_signal)
        dark_noise = self.dark_noise_generator(self.dark_current, self.bandwidth, self.load_resistance, size)
        voltage_noise = self.voltage_noise_generator(self.voltage_noise, self.load_resistance,
                                                     self.load_capacity, self.internal_capacity, size)
        johnson_noise = self.johnson_noise_generator(self.detector_temperature, self.bandwidth, self.load_resistance,
                                                     size)
        noised_signal = detector_current + dark_noise + voltage_noise + johnson_noise
        return noised_signal


class Detector(object):
    def __new__(cls, detector_type='apd', parameters=None, data_path=None):
        assert isinstance(detector_type, str), 'Expected data type for <detector_type> is str.'
        if detector_type is 'apd':
            return APD(cls.__get_detector_parameters(parameters, data_path, detector_type))
        elif detector_type is 'pmt':
            return PMT(cls.__get_detector_parameters(parameters, data_path, detector_type))
        elif detector_type is 'ppd':
            return PPD(cls.__get_detector_parameters(parameters, data_path, detector_type))
        else:
            raise ValueError('The requested detector type:' + detector_type + ' is not yet supported')

    @staticmethod
    def __get_detector_parameters(parameters, data_path, detector_type):
        if isinstance(parameters, etree._ElementTree):
            print('Detector parameters received from external source.')
            return parameters
        elif data_path is None:
            return ut.GetData(data_path_name='detector/'+detector_type+'_default.xml').data
        elif isinstance(data_path, str):
            return ut.GetData(data_path_name=data_path).data
        else:
            raise ValueError('Expected data type for <data_path> is str or None, '
                             'in which case default values will be loaded')
