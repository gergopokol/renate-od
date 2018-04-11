import matplotlib.pyplot

from imas_wrapper.wrapper_1d import BeamletFromIds


class Visualize:
    def __init__(self, profiles='None'):
        if profiles == 'None':
            bfi = BeamletFromIds()
            bfi.compute_beamevolution()
            self.profiles = bfi.profiles
        self.plot_populations()

    def plot_populations(self):
        for level in range(9):
            label = 'level ' + str(level)
            matplotlib.pyplot.plot(self.profiles['beamlet_grid'], self.profiles[label], label=label)
            matplotlib.pyplot.yscale('log', nonposy='clip')
            matplotlib.pyplot.ylim((1e-5, 1))
        matplotlib.pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5), ncol=1)
        matplotlib.pyplot.xlabel('x')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()
