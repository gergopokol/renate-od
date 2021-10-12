import scipy.constants as c
from utility.exceptions import InputError


class Particle(object):
    def __init__(self, label=str, mass_number=int, atomic_number=int, charge=0, mass=None):
        if isinstance(label, str):
            self.label = label
        else:
            raise InputError('Label is expected to be of string type.')
        if (atomic_number >= 0) and isinstance(atomic_number, int):
            self.atomic_number = atomic_number
        else:
            raise InputError('The atomic number is expected to be a positive integer.')
        if (mass_number >= 0) and isinstance(mass_number, int) and (self.atomic_number <= mass_number):
            self.mass_number = mass_number
        else:
            raise InputError('The mass number must be a positive integer and larger than the atomic number.')
        if isinstance(charge, int) and (charge <= self.atomic_number) and (charge >= -1):
            self.charge = charge
        else:
            raise InputError('The charge '+str(charge) +
                             ' of the particle can not exceed the atomic number: ' + str(self.atomic_number))
        if mass is None:
            if self.label == 'e':
                self.mass = c.electron_mass
            else:
                self.mass = c.proton_mass * self.atomic_number + \
                            c.neutron_mass * (self.mass_number - self.atomic_number)
        else:
            self.mass = mass

    def update_mass(self, mass):
        self.mass = mass

    def update_charge(self, charge):
        self.charge = charge

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return 'Particle: ' + str(self.label) + ' with Z = ' + str(self.atomic_number) + \
               ' N = ' + str(self.mass_number) + ' charge = ' + str(self.charge)


class Transition(object):
    def __init__(self):
        pass
