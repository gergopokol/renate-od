import matplotlib.pyplot
import pandas


class Profiles:
    def __init__(self, access_path):
        self.profiles = pandas.read_hdf(access_path)
        self.plot_populations()

    def plot_populations(self):
        for level in range(9):
            label = 'level ' + str(level)
            matplotlib.pyplot.plot(self.profiles['beamlet_grid'], self.profiles[label], label=label)
            matplotlib.pyplot.yscale('log', nonposy='clip')
            matplotlib.pyplot.ylim((1e-5, 1))
        matplotlib.pyplot.legend(loc='best', bbox_to_anchor=(0, 0.5), ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()
