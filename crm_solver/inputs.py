import numpy


class Inputs:
    initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    step_interval = 0.1
    step_number = 100
    beam_energy = 60
    beam_species = 'Li'
    electron_temperature = numpy.linspace(1,6,step_number)*1e3
    proton_temperature = numpy.linspace(1,6,step_number)*1e3
    density = numpy.linspace(1,6,step_number)*1e20

    number_of_levels = len(initial_condition)
    steps = numpy.linspace(0, step_interval, step_number)



