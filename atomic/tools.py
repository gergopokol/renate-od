from utility.constants import Constants
from utility.exceptions import InputError
#from atomic.coefficients import ATOMIC_SOURCES
import numpy as np
import h5py
import scipy.constants as sc
from renate_od.atomic import cross_section

CONST = Constants()


class Particle(object):
    def __init__(self, label='', charge=0, mass_number=0, atomic_number=0, mass=None):
        if isinstance(label, str):
            self.label = label
        else:
            raise InputError('Label is expected to be of string type.')
        if isinstance(atomic_number, int):
            self.atomic_number = atomic_number
        if isinstance(mass_number, int) and (self.atomic_number - mass_number) >= 0:
            self.mass_number = mass_number
            self.neutron_number = self.mass_number - self.atomic_number
        if self.label is 'e':
            self.charge = -1
        elif isinstance(charge, int):
            self.charge = charge
        else:
            raise InputError('The charge '+str(charge) + ' of the particle must exceed -1 and be an integer.')
        if mass is None and self.label == 'e':
            self.mass = CONST.electron_mass
        else:
            self.mass = mass

    def update_mass(self, mass):
        self.mass = mass

    def update_atomic_number(self, atomic_number):
        self.atomic_number = atomic_number

    def update_mass_number(self, mass_number):
        self.mass_number = mass_number
        self.neutron_number = self.mass_number - self.atomic_number

    def update_charge(self, charge):
        self.charge = charge

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return 'Particle: ' + str(self.label) + '\t (q,Z,A) =  (' + str(self.charge) + ',' + str(self.atomic_number) + \
               ',' + str(self.mass_number) + ')'


class Ion(Particle):
    def __init__(self, label='', charge=int, mass=None, atomic_number=int, mass_number=int):
        Particle.__init__(self, label=label, mass=mass, charge=charge)
        if (atomic_number >= 0) and isinstance(atomic_number, int) and (atomic_number >= self.charge):
            self.atomic_number = atomic_number
        else:
            raise InputError('The atomic number is expected to be a positive integer '
                             'and larger or equal to the charge.')
        if (mass_number >= 0) and isinstance(mass_number, int) and (self.atomic_number <= mass_number):
            self.mass_number = mass_number
        else:
            raise InputError('The mass number must be a positive integer and larger than the atomic number.')
        if mass is None:
            self.mass = CONST.proton_mass * self.atomic_number + \
                CONST.neutron_mass * (self.mass_number - self.atomic_number)
        else:
            self.mass = mass

    def __repr__(self):
        return 'Ion: ' + str(self.label) + '\t Z = ' + str(self.atomic_number) + \
            ' N = ' + str(self.mass_number) + ' charge = ' + str(self.charge)


class Atom(Ion):
    def __init__(self, label='', mass_number=int, atomic_number=int, mass=None):
        Ion.__init__(self, label=label, mass_number=mass_number, mass=mass, atomic_number=atomic_number, charge=0)

    def __repr__(self):
        return 'Atom: ' + str(self.label) + '\t Z = ' + str(self.atomic_number) + ' N = ' + str(self.mass_number)

    def __add__(self, other):
        self.atomic_number += other.atomic_number
        self.mass_number += other.mass_number
        self.label += other.label
        self.mass += other.mass

    def __mul__(self, other):
        self.atomic_number *= other
        self.mass_number *= other
        self.label = str(other) + self.label
        self.mass *= other


class Molecule(Atom):
    def __init__(self, atoms=list, mass=None):
        Atom.__init__(self, label='', mass_number=0, atomic_number=0, mass=mass)
        for atom in atoms:
            self.mass += atom.mass
            self.mass_number += atom.mass_number
            self.atomic_number += atom.atomic_number
            self.label += atom.label

    def __repr__(self):
        return 'Molecule: ' + str(self.label) + '\t Protons = ' + str(self.atomic_number) + \
               ' Neutrons = ' + str(self.mass_number - self.atomic_number)


class IonizedMolecule(Molecule):
    def __init__(self, atoms=list, mass=None, charge=int):
        Molecule.__init__(self, atoms=atoms, mass=mass)
        self.update_charge(charge=charge)

    def __repr__(self):
        return 'Ionized Molecule: ' + str(self.label) + '\t Protons = ' + str(self.atomic_number) + \
               ' Nucleons = ' + str(self.mass_number) + ' Charge = ' + str(self.charge)


class Transition(object):
    def __init__(self, projectile=Particle, target=Particle, from_level=str, to_level=None, trans=str):
        self.projectile = projectile
        self.target = target
        if isinstance(from_level, str) and isinstance(trans, str):
            self.from_level = from_level
            if trans in ['ex', 'de-ex', 'eloss', 'ion', 'cx']:
                self.name = trans
            else:
                InputError('The provided transition is not supported. '
                           'Supported transitions are: ex, de-ex, eloss, ion and cx.')
            if (to_level is None) or isinstance(to_level, str):
                self.to_level = to_level
            else:
                InputError('The provided end state for the electron transition is not valid. Str or None is expected.')
        else:
            InputError('Expected input data format for <from_level>, <to_level> and <trans> to be of str type.')

    def __str__(self):
        if self.name in ['cx', 'eloss', 'ion']:
            return self.from_level + '-' + self.name
        else:
            return self.from_level + '-' + self.to_level

    def __repr__(self):
        return 'Collision of: '+str(self.projectile)+' + '+str(self.target)+' with transition: ' + \
               self.name+' | from level: '+self.from_level+' | to_level: '+str(self.to_level)


class RENATE_H_hdf_generator():

    def __init__(self, atomic_dict):
        self.atomic_dict = atomic_dict
        self.max_level = 6
        self.einsteins = np.array([[0.0000e+00, 4.6986e+08, 5.5751e+07, 1.2785e+07, 4.1250e+06,
                                    1.6440e+06],
                                   [0.0000e+00, 0.0000e+00, 4.4101e+07, 8.4193e+06, 2.5304e+06,
                                    9.7320e+05],
                                   [0.0000e+00, 0.0000e+00, 0.0000e+00, 8.9860e+06, 2.2008e+06,
                                    7.7829e+05],
                                   [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.6993e+06,
                                    7.7110e+05],
                                   [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                    1.0254e+06],
                                   [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                    0.0000e+00]], dtype='float')

    def __build_rate_matrix(self, mx_type, projectile, target):
        if mx_type == 'collisional':
            matrix = np.zeros((self.max_level, self.max_level, len(self.T)))
            for i in range(len(self.T)):
                matrix[:, :, i] = self.__get_rate_matrix(projectile, target, self.T[i])
            return matrix
        if mx_type == 'eloss':
            matrix = np.zeros((self.max_level, len(self.T)))
            for i in range(len(self.T)):
                matrix[:, i] = self.__get_eloss_rate_matrix(projectile, target, self.T[i])
            return matrix

    def __get_rate_matrix(self, projectile, target, t):
        E_range = self.__get_energy_range(t, target.mass, projectile.mass)
        rate_matrix = np.zeros((self.max_level, self.max_level), dtype='float')
        levels = np.arange(self.max_level)
        for i in levels:
            for j in levels:
                if i != j:
                    if j > i:
                        transtype = 'ex'
                    else:
                        transtype = 'de-ex'
                    trans = Transition(projectile=projectile, target=target,
                                       from_level=str(i+1), to_level=str(j+1), trans=transtype)
                    crossec = cross_section.CrossSection(transition=trans, impact_energy=E_range, atomic_dict=self.atomic_dict)
                    rate = cross_section.RateCoeff(transition=trans, crossection=crossec)
                    rate_matrix[i, j] = rate.generate_rate(temperature=t, beamenergy=self.beamenergy)
        return rate_matrix

    def __get_eloss_rate_matrix(self, projectile, target, t):
        E_range = self.__get_energy_range(t, target.mass, projectile.mass)
        rate_matrix = np.zeros((self.max_level), dtype='float')
        levels = np.arange(self.max_level)
        for i in levels:
            trans = Transition(projectile=projectile, target=target,
                               from_level=str(i+1), to_level='eloss', trans='eloss')
            crossec = cross_section.CrossSection(transition=trans, impact_energy=E_range, atomic_dict=self.atomic_dict)
            rate = cross_section.RateCoeff(transition=trans, crossection=crossec)
            rate_matrix[i] = rate.generate_rate(temperature=t, beamenergy=self.beamenergy)
        return rate_matrix

    def __get_energy_range(self, t, m_t, m_b, minE=13.6, maxE=1e7, N=1000, k=3):
        w = np.sqrt(2*t*sc.eV/m_t)
        vb = np.sqrt(2*self.beamenergy*sc.eV/m_b)
        v_min = max(vb-k*w, 0)
        v_max = vb+k*w
        roi_start = max(0.5*m_t*v_min**2/sc.eV, minE)
        roi_max = min(0.5*m_t*v_max**2/sc.eV, maxE)
        E_range = np.linspace(roi_start, roi_max, N)
        return E_range

    def write_hdf(self, beamenergy, temperature, path='rate_coeffs_'):
        H = Atom(label='H', mass_number=1, atomic_number=1)
        el = Particle('e', charge=-1)
        prot = Ion(label='1H1+', mass_number=1, atomic_number=1, charge=1)
        self.beamenergy = beamenergy
        self.T = temperature

        self.filename = path+str(int(self.beamenergy/1000))+'_H.h5'
        rate_data = h5py.File(self.filename, "w")
        rate_data.create_dataset('Beam energy', (), dtype='<i2', data=self.beamenergy)
        self.atomic_levels = rate_data.create_dataset('Atomic Levels', dtype='|S3', data=[b'1l', b'2l', b'3l', b'4l', b'5l', b'6l'])
        self.beam_type = rate_data.create_dataset('Beam type', dtype='|S2', data=b'H')
        rate_data.create_dataset('Einstein Coeffs', dtype='<f4', data=self.einsteins)
        rate_data.create_dataset('Temperature axis', dtype='<f8', data=self.T)
        self.impurity_collisions = rate_data.create_dataset('Impurity Collisions', dtype='<i2', data=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.impur_neutral = rate_data.create_dataset('Collisional Coeffs/Impurity Neutral Collisions', dtype='<f8', data=np.zeros((10, 6, 6, 400)))
        self.electron_neutral = self.__build_rate_matrix('collisional', H, el)
        rate_data.create_dataset('Collisional Coeffs/Electron Neutral Collisions', dtype='<f8', data=self.electron_neutral)
        self.proton_neutral = self.__build_rate_matrix('collisional', H, prot)
        rate_data.create_dataset('Collisional Coeffs/Proton Neutral Collisions', dtype='<f8', data=self.proton_neutral)
        e_eloss = self.__build_rate_matrix('eloss', H, el)
        p_eloss = self.__build_rate_matrix('eloss', H, prot)
        eloss = np.zeros((12, 6, 400), dtype='float')
        eloss[0, :, :] = e_eloss
        eloss[1, :, :] = p_eloss
        self.eloss = eloss
        rate_data.create_dataset('Collisional Coeffs/Electron Loss Collisions', dtype='<f8', data=self.eloss)
        rate_data.close()
        return self.filename
