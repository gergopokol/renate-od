import numpy


class Inputs:
    initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    step_interval = 0.1
    step_number = 100
    beam_energy = 60
    beam_species = 'Li'
    electron_temperature = 3e3
    proton_temperature = 2e2
    density = 6e19

    number_of_levels = len(initial_condition)
    steps = numpy.linspace(0, step_interval, step_number)


class Inputs1:
    initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    step_interval = 0.1
    step_number = 100
    beam_energy = 60
    beam_species = 'Li'
    electron_temperature = 3e3
    proton_temperature = 3e2
    density = 6e19

    number_of_levels = len(initial_condition)
    steps = numpy.linspace(0, step_interval, step_number)


class Inputs2:
    initial_condition = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    step_interval = 0.1
    step_number = 30
    beam_energy = 60
    beam_species = 'Li'
    electron_temperature = 3e3
    proton_temperature = 3e2
    density = 6e19

    number_of_levels = len(initial_condition)
    steps = numpy.linspace(0, step_interval, step_number)
