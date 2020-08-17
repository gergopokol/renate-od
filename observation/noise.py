import numpy
import scipy
import utility.getdata as ut
import pandas
import utility.constants as const


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


class Generator:
    def __init__(self, DispersionType, Mean, Deviation):
        self.DispersionType = DispersionType
        self.Mean = Mean
        self.Deviation = Deviation

    def Generate(self):
        if self.DispersionType == "Poisson":
            PoissonRandNumb = numpy.random.poisson(self.Mean)
            return PoissonRandNumb

        if self.DispersionType == "Gaussian":
            GaussianRandNumb = numpy.random.normal(loc=self.Mean, scale=self.Deviation)
            return GaussianRandNumb

    pass

class Parameters:
    def __init__(self, Gain, QuantumEfficiency, NoiseIndex, Bandwidth, LoadResistance, LoadCapacity, DarkCurrent,
                 Temperature, VoltageNoise, InternalCapacities):
        self.Gain = Gain
        self.QuantumEfficiency = QuantumEfficiency
        self.NoiseIndex = NoiseIndex
        self.Bandwidth = Bandwidth
        self.LoadResistance = LoadResistance
        self.LoadCapacity = LoadCapacity
        self.DarkCurrent = DarkCurrent
        self.Temperature = Temperature
        self.VoltageNoise = VoltageNoise
        self.InternalCapacities = InternalCapacities


class Noise(Generator, Parameters):
    def __init__(self, NoiseType):
        self.NoiseType = NoiseType

    def NoiseGeneration(self):
        p = Parameters()
        c = const.Constants()
        if self.NoiseType == "Johnson":
            Mean = 0
            Deviation = scipy.sqrt(4 * c.Boltzmanns_constant * p.Temperature * p.Bandwidth * p.LoadResistance)
            g = Generator("Gaussian", Mean, Deviation)
            Johnsonnoise = g.Generate()
            return Johnsonnoise
        if self.NoiseType == "Voltage":
            Mean = 0
            Deviation = p.VoltageNoise * scipy.sqrt(1 / (2 * scipy.pi * p.LoadResistance * (p.LoadCapacity + p.InternalCapacities))) + p.VoltageNoise * (
                        1 + p.InternalCapacities / p.LoadCapacity) * scipy.sqrt(1.57 * 1 / (2 * scipy.pi * p.LoadResistance* p.LoadCapacity))
            g = Generator("Gaussian", Mean, Deviation)
            Voltagenoise = g.Generate()
            return Voltagenoise
        if self.NoiseType == "Dark":
            Mean = 0
            Deviation = scipy.sqrt(2 * c.charge_electron * p.DarkCurrent * p.Bandwidth) * p.LoadResistance
            g = Generator("Gaussian", Mean, Deviation)
            Darknoise = g.Generate()
            return Darknoise
