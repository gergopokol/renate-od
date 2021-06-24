import numpy
import pandas
import utility.getdata as ut
import math
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

    def photon_flux_to_detector_voltage(self, signal, detector_gain, quantum_efficiency, load_resistance,
                                        sampling_frequency):
        return signal * self.constants.charge_electron * detector_gain * quantum_efficiency * load_resistance * \
               sampling_frequency

    def generate_photon_noise(self, signal):
        return self.poisson(signal)

    def generate_gaussian_noise(self, signal, snr):
        average = signal.avg()
        noised_signal = self.normal(signal, average/snr)
        return noised_signal

    def _shot_noise_setup(self, signal, detector_gain, load_resistance, noise_index, bandwidth):
        """
        :return: mean (U_det) and STD (sqrt(2*q*U_det*M*F*B*R_L)), where F = M exp(x)
        """
        return signal, numpy.array(numpy.sqrt(2 * self.constants.charge_electron * signal *
                                   numpy.power(detector_gain, noise_index + 1) * bandwidth * load_resistance))

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
        return self.normal(expected_value, variance, signal_size)

    @staticmethod
    def _voltage_noise_setup(voltage_noise, load_resistance, load_capacity, internal_capacity, expected_value=0):
        return expected_value,\
            voltage_noise * numpy.sqrt(1 / (2 * numpy.pi * load_resistance * (load_capacity + internal_capacity))) + \
            voltage_noise * (1 + internal_capacity / load_capacity) * \
            numpy.sqrt(1.57 * 1 / (2 * numpy.pi * load_resistance * load_capacity))

    def voltage_noise_generator(self, voltage_noise, load_resistance, load_capacity, internal_capacity, signal_size):
        expected_value, variance = self._voltage_noise_setup(voltage_noise, load_resistance,
                                                             load_capacity, internal_capacity)
        return self.normal(expected_value, variance, signal_size)

    def _dark_noise_setup(self, dark_current, bandwidth, load_resistance):
        """
        :return: mean (I_dark*R_l), STD(sqrt(2*q*I_dark*B)*R_l)
        """
        return dark_current * load_resistance, \
            numpy.sqrt(2 * self.constants.charge_electron * dark_current * bandwidth) * load_resistance

    def dark_noise_generator(self, dark_current, bandwidth, load_resistance, signal_size):
        expected_value, variance = self._dark_noise_setup(dark_current, bandwidth, load_resistance)
        return self.normal(expected_value, variance, signal_size)

    def derive_background_emission_in_photon_count(self, signal, sbr):
        expected_background = numpy.ones(len(signal)) * signal.mean() / sbr
        return self.generate_photon_noise(expected_background)

    def _mppc_gaussian_noise_init(self, signal, dark_count_rate, photon_detection_efficiency, sampling_frequency):
        mean = signal * photon_detection_efficiency
        std = numpy.sqrt(signal * photon_detection_efficiency + dark_count_rate / sampling_frequency)
        return mean, std

    def mppc_gaussian_noise_generator(self, signal, dark_count_rate, sampling_frequency, photon_detection_efficiency):
        mean, std = self._mppc_gaussian_noise_init(signal, dark_count_rate, photon_detection_efficiency,
                                                   sampling_frequency)
        noisy_signal = self.normal(mean, std)
        return noisy_signal * sampling_frequency

    def mppc_transfer(self, signal, photon_detection_efficiency, detector_gain, quenching_resistance):
        detector_voltage = signal * photon_detection_efficiency * detector_gain * quenching_resistance \
                           * self.constants.charge_electron
        return detector_voltage


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

    def _apd_noiseless_transfer(self, signal):
        detector_voltage = (signal * (1 + 1 / self.signal_to_background) * self.quantum_efficiency
                            * self.constants.charge_electron * self.detector_gain
                            + self.dark_current) * self.load_resistance
        return detector_voltage

    def add_noise_to_signal(self, signal):
        size = self.signal_length(signal)
        prepared_signal = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        background_noised_signal = self.background_addition(prepared_signal, self.signal_to_background)
        detector_voltage = self.photon_flux_to_detector_voltage(background_noised_signal, self.detector_gain,
                                                                self.quantum_efficiency, self.load_resistance,
                                                                self.sampling_frequency)
        shot_noised_signal = self.shot_noise_generator(detector_voltage, self.detector_gain, self.load_resistance,
                                                       self.noise_index, self.bandwidth)
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
        self.load_resistance = float(detector_parameters.getroot().find('body').find('load_resistance').text)

    def _pmt_noiseless_transfer(self, signal):
        detector_current = signal * (1 + 1 / self.signal_to_background) * self.quantum_efficiency \
                           * self.constants.charge_electron * (self.dynode_gain ** self.dynode_number) \
                           + self.dark_current
        return detector_current

    def _pmt_shot_noise_generation(self, signal):
        mean, variance = self._pmt_shot_noise_setup(signal)
        return self.normal(mean, variance)

    def _pmt_shot_noise_setup(self, signal):
        mean = signal * self.sampling_frequency * self.dynode_gain ** self.dynode_number * self.quantum_efficiency * \
               self.constants.charge_electron
        std = numpy.sqrt(2*self.constants.charge_electron*mean*(self.dynode_gain/(self.dynode_gain-1)) *
                         self.bandwidth*self.dynode_gain**self.dynode_number)
        return mean, std

    def _pmt_dark_noise_generation(self, signal):
        mean = numpy.ones(len(signal))*self.dark_current
        std = numpy.sqrt(4*self.constants.charge_electron*mean*self.dynode_gain**self.dynode_number*
                         (self.dynode_gain/(self.dynode_gain-1))*self.bandwidth)
        return self.normal(mean, std)

    def _photo_cathode_electron_generation(self, signal):
        emitted_electrons = self.poisson(signal * self.quantum_efficiency).astype(float)
        return emitted_electrons

    def _dynode_noise_generator(self, signal, dynode_number, dynode_gain):
        for i in range(dynode_number):
            signal = numpy.abs(signal)
            for j in range(len(signal)):
                if signal[j] * dynode_gain > 10:
                    signal[j] = self.normal(signal[j] * dynode_gain, math.sqrt(signal[j] * dynode_gain))
                else:
                    signal[j] = numpy.float(self.poisson(signal[j] * dynode_gain))
        return signal

    def _thermionic_dark_electron_generator(self, signal_length):
        expected_cathode_electron_count = self.dark_current / (self.constants.charge_electron * self.dynode_gain ** self.dynode_number)
        electron_generation_rate = expected_cathode_electron_count / self.sampling_frequency
        if electron_generation_rate >= 1:
            return self.poisson(numpy.ones(signal_length)*electron_generation_rate)
        else:
            electron_generation = self.uniform(0, 1, signal_length)
            electron_generation[electron_generation <= electron_generation_rate] = 1
            electron_generation[electron_generation > electron_generation_rate] = 0
            return electron_generation

    def _pmt_gaussian_noise_generator(self, signal):
        expected_emission_photon_count = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        total_expected_photon_count = self.background_addition(expected_emission_photon_count,
                                                               self.signal_to_background)
        anode_photon_current = self._pmt_shot_noise_generation(total_expected_photon_count)
        anode_dark_current = self._pmt_dark_noise_generation(total_expected_photon_count)
        noisy_voltage_signal = (anode_photon_current + anode_dark_current) * self.load_resistance + \
                               self.johnson_noise_generator(self.detector_temperature, self.bandwidth,
                                                            self.load_resistance, len(signal))
        return noisy_voltage_signal

    def _pmt_poisson_noise_generator(self, signal):
        expected_emission_photon_count = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        emission_photon_count = self.generate_photon_noise(expected_emission_photon_count)
        background_photon_count = self.derive_background_emission_in_photon_count(expected_emission_photon_count,
                                                                                  self.signal_to_background)
        emitted_electrons = self._photo_cathode_electron_generation(emission_photon_count + background_photon_count)
        dark_electrons = self._thermionic_dark_electron_generator(self.signal_length(signal))
        anode_electron_count = self._dynode_noise_generator(emitted_electrons + dark_electrons, self.dynode_number,
                                                            self. dynode_gain)
        anode_current = anode_electron_count * self.constants.charge_electron * self.sampling_frequency
        return self.load_resistance * anode_current + self.johnson_noise_generator(self.detector_temperature,
                                                                                   self.bandwidth, self.load_resistance,
                                                                                   self.signal_length(anode_current))

    def add_noise_to_signal(self, signal, noise_type='poisson'):
        if noise_type == 'poisson':
            return self._pmt_poisson_noise_generator(signal)
        elif noise_type == 'gaussian':
            return self._pmt_gaussian_noise_generator(signal)
        else:
            raise ValueError('The requested noise type does not exist or is not implemented.', noise_type)


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

    def _ppd_noiseless_transfer(self, signal):
        detector_voltage = (signal * (1 + 1 / self.signal_to_background) * self.quantum_efficiency
                            * self.constants.charge_electron + self.dark_current) * self.load_resistance
        return detector_voltage

    def _ppd_transfer(self, signal):
        detector_voltage = self.poisson(signal * self.quantum_efficiency) * self.constants.charge_electron * \
                           self.sampling_frequency * self.load_resistance
        return detector_voltage

    def add_noise_to_signal(self, signal):
        size = self.signal_length(signal)
        prepared_signal = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        emitted_photons = self.generate_photon_noise(prepared_signal)
        background_noised_signal = self.background_addition(emitted_photons, self.signal_to_background)
        detector_voltage = self._ppd_transfer(background_noised_signal)
        dark_noise = self.dark_noise_generator(self.dark_current, self.bandwidth, self.load_resistance, size)
        voltage_noise = self.voltage_noise_generator(self.voltage_noise, self.load_resistance,
                                                     self.load_capacity, self.internal_capacity, size)
        johnson_noise = self.johnson_noise_generator(self.detector_temperature, self.bandwidth, self.load_resistance,
                                                     size)
        noised_signal = detector_voltage + dark_noise + voltage_noise + johnson_noise
        return noised_signal


class MPPC(Noise):
    def __init__(self, detector_parameters):
        Noise.__init__(self)
        self.__setup_detector_parameters(detector_parameters)

    def __setup_detector_parameters(self, detector_parameters):
        assert isinstance(detector_parameters, etree._ElementTree), 'Expected data type for <detector_parameters> ' \
                                                                    'is etree._ElementTree.'
        assert detector_parameters.getroot().find('head').find('type').text == 'mppc', \
            'The detector type to be set is MPPC, Please check input data.'
        self.detector_gain = float(detector_parameters.getroot().find('body').find('detector_gain').text)
        self.quantum_efficiency = float(detector_parameters.getroot().find('body').find('quantum_efficiency').text)
        self.bandwidth = float(detector_parameters.getroot().find('body').find('bandwidth').text)
        self.dark_count_rate = float(detector_parameters.getroot().find('body').find('dark_count_rate').text)
        self.sampling_frequency = float(detector_parameters.getroot().find('body').find('sampling_frequency').text)
        self.signal_to_background = float(detector_parameters.getroot().find('body').find('signal_to_background').text)
        self.photon_detection_efficiency = float(detector_parameters.getroot().find('body').
                                                 find('photon_detection_efficiency').text)
        self.terminal_capacitance = float(detector_parameters.getroot().find('body').find('terminal_capacitance').text)
        self.total_pixel_count = float(detector_parameters.getroot().find('body').find('total_pixel_count').text)

    def add_noise_to_signal(self, signal):
        prepared_signal = self._photon_flux_to_photon_number(signal, self.sampling_frequency)
        emitted_photons = self.generate_photon_noise(prepared_signal)
        background_noised_signal = self.background_addition(emitted_photons, self.signal_to_background)
        noisy_signal = self.mppc_gaussian_noise_generator(background_noised_signal, self.dark_count_rate,
                                                          self.sampling_frequency, self.photon_detection_efficiency)
        return noisy_signal


class Detector(object):
    def __new__(cls, detector_type='apd', parameters=None, data_path=None):
        assert isinstance(detector_type, str), 'Expected data type for <detector_type> is str.'
        if detector_type is 'apd':
            return APD(cls.__get_detector_parameters(parameters, data_path, detector_type))
        elif detector_type is 'pmt':
            return PMT(cls.__get_detector_parameters(parameters, data_path, detector_type))
        elif detector_type is 'ppd':
            return PPD(cls.__get_detector_parameters(parameters, data_path, detector_type))
        elif detector_type is 'mppc':
            return MPPC(cls.__get_detector_parameters(parameters, data_path, detector_type))
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
