import matplotlib.pyplot
import pandas
from lxml import etree
import utility
from matplotlib.backends.backend_pdf import PdfPages
import datetime


class BeamletProfiles:
    def __init__(self, param_path='output/beamlet/beamlet_test.xml', key=['profiles']):
        self.param_path = param_path
        self.param = utility.getdata.GetData(data_path_name=self.param_path).data
        self.access_path = self.param.getroot().find('body').find('beamlet_profiles').text
        self.key = key
        self.profiles = utility.getdata.GetData(data_path_name=self.access_path, data_key=self.key).data
        self.title = None

    def set_x_range(self, x_min=None, x_max=None):
        self.x_limits = [x_min, x_max]

    def plot_benchmark_populations(self):
        axis = matplotlib.pyplot.subplot()
        self.setup_RENATE_benchmark(axis)
        matplotlib.pyplot.show()

    def setup_RENATE_benchmark(self, axis):
        number_of_levels = self.get_number_of_levels(self.profiles)
        keys = self.profiles.keys()
        level_keys = []
        for i in range(len(keys)):
            if 'level' in keys[i]:
                level_keys.append(keys[i])
        for level in range(number_of_levels):
            label = level_keys[level]
            axis.plot(self.profiles['beamlet grid'], self.profiles[label], label=label)
        axis.set_yscale('log', nonposy='clip')
        axis.set_xlabel('Distance [m]')
        axis.set_ylabel('Relative electron population [-]')
        axis.legend(loc='best', ncol=1)
        self.title = 'Beamlet profiles'
        axis.set_title(self.title)
        axis.grid()
        return axis

    def plot_linear_emission_density(self):
        axis_dens = matplotlib.pyplot.subplot()
        self.setup_density_axis(axis_dens)
        axis_dens.set_xlabel('Distance [m]')
        axis_em = axis_dens.twinx()
        self.setup_linear_emission_density_axis(axis_em)
        matplotlib.pyplot.show()

    def setup_linear_emission_density_axis(self, axis):
        axis.plot(self.profiles['beamlet grid'], self.profiles['linear_emission_density'],
                  label='Linear emission density', color='r')
        axis.set_ylabel('Linear emission density [ph/sm]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='upper right')
        return axis

    def plot_attenuation(self):
        axis_dens = matplotlib.pyplot.subplot()
        self.setup_density_axis(axis_dens)
        axis_dens.set_xlabel('Distance [m]')
        axis_em = axis_dens.twinx()
        self.setup_linear_density_attenuation_axis(axis_em)
        matplotlib.pyplot.show()

    def setup_linear_density_attenuation_axis(self, axis):
        axis.plot(self.profiles['beamlet grid'], self.profiles['linear_density_attenuation'],
                  label='Linear density attenuation', color='r')
        axis.set_ylabel('Linear density [1/m]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='upper right')
        return axis

    def plot_relative_populations(self):
        axis = matplotlib.pyplot.subplot()
        self.setup_population_axis(axis, kind='relative')
        matplotlib.pyplot.show()

    def plot_populations(self):
        axis = matplotlib.pyplot.subplot()
        self.setup_population_axis(axis)
        matplotlib.pyplot.show()

    def plot_all_profiles(self):
        fig1 = matplotlib.pyplot.figure()
        grid = matplotlib.pyplot.GridSpec(3, 1)
        ax1 = matplotlib.pyplot.subplot(grid[0, 0])
        ax1 = self.setup_density_axis(ax1)
        ax2 = ax1.twinx()
        self.setup_temperature_axis(ax2)
        self.title = 'Plasma profiles'
        ax1.set_title(self.title)
        ax3 = matplotlib.pyplot.subplot(grid[1:, 0])
        self.setup_population_axis(ax3)
        fig1.tight_layout()
        matplotlib.pyplot.show()

    def benchmark(self, benchmark_param_path='../data/beamlet/IMAS_beamlet_test_profiles_Li.xml', key=['profiles']):
        benchmark_param = utility.getdata.GetData(data_path_name=benchmark_param_path).data
        benchmark_path = benchmark_param.getroot().find('body').find('beamlet_profiles').text
        benchmark_profiles = utility.getdata.GetData(data_path_name=benchmark_path, data_key=key).data
        fig1 = matplotlib.pyplot.figure()
        ax1 = matplotlib.pyplot.subplot()
        ax1 = self.setup_population_axis(ax1)
        ax1 = self.setup_benchmark_axis(benchmark_profiles, axis=ax1)
        ax1.legend(loc='best', ncol=2)
        self.title = 'Beamlet profiles - benchmark'
        ax1.set_title(self.title)
        ax1.grid()
        fig1.tight_layout()
        matplotlib.pyplot.show()

    def setup_density_axis(self, axis):
        axis.plot(self.profiles['beamlet grid'], self.profiles['electron']
                  ['density']['m-3'], label='Density', color='b')
        if hasattr(self, 'x_limits'):
            axis.set_xlim(self.x_limits)
        axis.set_ylabel('Density [1/m3]')
        axis.yaxis.label.set_color('b')
        axis.legend(loc='upper left')
        axis.grid()
        return axis

    def setup_temperature_axis(self, axis):
        axis.plot(self.profiles['beamlet grid'], self.profiles['electron']['temperature']['eV'], color='r',
                  label='Electron_temperature')
        axis.plot(self.profiles['beamlet grid'], self.profiles['ion1']['temperature']['eV'], '--', label='Ion_temperature',
                  color='m')
        axis.set_ylabel('Temperature [eV]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='lower right')
        axis.grid()
        return axis

    def setup_population_axis(self, axis, kind='absolute'):
        pandas_key, axis_name = self.set_axis_parameters(kind)
        number_of_levels = self.get_number_of_levels(self.profiles)
        for level in range(number_of_levels):
            label = pandas_key + str(level)
            axis.plot(self.profiles['beamlet grid'], self.profiles[label], label=label)
        if hasattr(self, 'x_limits'):
            axis.set_xlim(self.x_limits)
        axis.set_yscale('log', nonposy='clip')
        axis.set_xlabel('Distance [m]')
        axis.set_ylabel(axis_name)
        axis.legend(loc='best', ncol=1)
        self.title = 'Beamlet profiles'
        axis.set_title(self.title)
        axis.grid()
        return axis

    @staticmethod
    def set_axis_parameters(kind):
        assert isinstance(kind, str)
        if kind == 'absolute':
            return 'level ', 'Linear density [1/m]'
        elif kind == 'relative':
            return 'rel.pop ', 'Relative linear density [-]'
        else:
            raise ValueError('Requested plotting format not accepted')

    def setup_benchmark_axis(self, benchmark_profiles, axis):
        benchmark_profiles = benchmark_profiles
        number_of_levels = self.get_number_of_levels(benchmark_profiles)
        for level in range(number_of_levels):
            label = 'level ' + str(level)
            axis.plot(benchmark_profiles['beamlet grid'], benchmark_profiles[label], '--', label=label+' ref.')
        return axis

    @staticmethod
    def get_number_of_levels(profiles):
        if len(profiles.filter(like='RENATE', axis=1).keys()) == 0:
            return len(profiles.filter(like='level', axis=1).keys())
        else:
            return len(profiles.filter(like='RENATE', axis=1).keys())

    def save_figure(self, file_path='data/output/beamlet/test_plot.pdf'):
        with PdfPages(file_path) as pdf:
            pdf.savefig()
            d = pdf.infodict()
            d['Title'] = self.title
            d['Keywords'] = 'Source hdf5 file: ' + self.access_path + ', source xml file: ' + self.param_path
            d['ModDate'] = datetime.datetime.today()
