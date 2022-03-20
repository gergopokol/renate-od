import os
import sys
import h5py
import matplotlib.pyplot
import numpy
import scipy

class fluctuations:

    def __init__(self):
        # Class properties storing info to be plotted
        appliance_name = "Undefined"
        shot_number = 0
        time_index = 0
        R_axis = numpy.array
        Z_axis = numpy.array
        density = numpy.array
        temperature = numpy.array

    def load_fluctuations(self):
        # Here comes a loader for 2D fluctuating fields using get_data_from_hdf5
        pass

    def get_data_from_hdf5(file_name, source_name):
        try:
            hdf5_id = h5py.File(file_name, 'r')
        except IOError:
            print("File not found!", file_name)
            quit()
        else:
            data = hdf5_id[source_name].value
            hdf5_id.close()
            return data

    def load_synthetic_signals(self):
        # Here comes a loader for simulated synthetic signals from .sav,
        # https://docs.scipy.org/doc/scipy/reference/io.html
        pass


    def generate_fluctuations_movie(self):
        # Here comes a composite movie generator using other routines
        pass

    def generate_fluctuations_figure(self):
        # Here comes the figure generator to be used in fluctuations_movie
        pass

    def plot_synthetic_signals(self):
        # Here comes the figure generator to be used in fluctuations_movie
        pass
