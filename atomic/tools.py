from utility.constants import Constants
from utility.exceptions import InputError
import numpy as np
import h5py
import scipy.constants as sc


class RENATE_H_hdf_generator():

    def __init__(self, beamenergy, temperature, atomic_dict):
        self.beamenergy = beamenergy
        self.T = temperature
        self.atomic_dict = atomic_dict
        self.max_level = 6
        self.impur_list = [('He', 2, 4), ('Z', 3, 6), ('Be', 4, 9), ('B', 5, 11), ('C', 6, 12), ('Z', 7, 14), ('O', 8, 16),
                           ('Z', 9, 18), ('Z', 10, 20), ('Z', 11, 22)]
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
        print(mx_type+' '+str(projectile)+'-->'+str(target))
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

    def __build_impurity_rate_matrix(self, mx_type, projectile):
        matrix = []
        for impurity in self.impur_list:
            target = Ion(label=impurity[0], mass_number=impurity[2], atomic_number=impurity[1], charge=impurity[1])
            matrix.append(self.__build_rate_matrix(mx_type, projectile, target))
        return np.array(matrix)

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

    def __get_energy_range(self, t, m_t, m_b, minE=13.6, N=1000, k=3):
        w = np.sqrt(2*t*sc.eV/m_t)
        vb = np.sqrt(2*self.beamenergy*sc.eV/m_b)
        v_min = max(vb-k*w, 0)
        v_max = vb+k*w
        roi_start = max(0.5*m_t*v_min**2/sc.eV, minE)
        roi_max = 0.5*m_t*v_max**2/sc.eV
        E_range = np.linspace(roi_start, roi_max, N)
        return E_range

    def write_hdf(self, path='rate_coeffs_'):
        H = Atom(label='H', mass_number=1, atomic_number=1)
        el = Particle('e', charge=-1)
        prot = Ion(label='1H1+', mass_number=1, atomic_number=1, charge=1)

        self.filename = path+str(int(self.beamenergy/1000))+'_H.h5'
        rate_data = h5py.File(self.filename, "w")
        rate_data.create_dataset('Beam energy', (), dtype='<i2', data=self.beamenergy)
        self.atomic_levels = rate_data.create_dataset('Atomic Levels', dtype='|S3', data=[b'1l', b'2l', b'3l', b'4l', b'5l', b'6l'])
        self.beam_type = rate_data.create_dataset('Beam type', dtype='|S2', data=b'H')
        rate_data.create_dataset('Einstein Coeffs', dtype='<f4', data=self.einsteins)
        rate_data.create_dataset('Temperature axis', dtype='<f8', data=self.T)
        self.impurity_collisions = rate_data.create_dataset('Impurity Collisions', dtype='<i2', data=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.impur_neutral = self.__build_impurity_rate_matrix('collisional', H)
        rate_data.create_dataset('Collisional Coeffs/Impurity Neutral Collisions', dtype='<f8', data=self.impur_neutral)
        self.electron_neutral = self.__build_rate_matrix('collisional', H, el)
        rate_data.create_dataset('Collisional Coeffs/Electron Neutral Collisions', dtype='<f8', data=self.electron_neutral)
        self.proton_neutral = self.__build_rate_matrix('collisional', H, prot)
        rate_data.create_dataset('Collisional Coeffs/Proton Neutral Collisions', dtype='<f8', data=self.proton_neutral)
        eloss = np.zeros((12, 6, 400), dtype='float')
        eloss[0, :, :] = self.__build_rate_matrix('eloss', H, el)
        eloss[1, :, :] = self.__build_rate_matrix('eloss', H, prot)
        eloss[2:, :, :] = self.__build_impurity_rate_matrix('eloss', H)
        self.eloss = eloss
        rate_data.create_dataset('Collisional Coeffs/Electron Loss Collisions', dtype='<f8', data=self.eloss)
        rate_data.close()
        return self.filename
