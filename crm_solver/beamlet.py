import os
import sys
import matplotlib.pyplot as plt
import h5py
import copy
import utility
from utility.constants import Constants
import pandas
import numpy as np
from copy import deepcopy
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
from atomic.atomic_db import AtomicDB
from utility.input import BeamletInput

class Beamlet:
    def __init__(self, param=None, profiles=None, components=None, atomic_db=None,
                 solver='numerical', data_path="beamlet/testimp0001.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.__read_beamlet_param(data_path)
        self.profiles = profiles
        self.components = components
        self.atomic_db = atomic_db
        if not (isinstance(self.components, pandas.DataFrame) and isinstance(self.profiles, pandas.DataFrame)):
            self.__read_beamlet_profiles()
        if atomic_db is None:
            self.atomic_db = AtomicDB(param=self.param, components=self.components)
        self.const = Constants()
        self.coefficient_matrix = None
        self.initial_condition = None
        self.calculate_beamevolution(solver)

    def __read_beamlet_param(self, data_path):
        self.param = utility.getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('Beamlet.param read from file: ' + data_path)

    def __read_beamlet_profiles(self):
        hdf5_path = self.param.getroot().find('body').find('beamlet_source').text
        self.components = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['components']).data
        assert isinstance(self.components, pandas.DataFrame)
        print('Beamlet.imp_components read from file: ' + hdf5_path)
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['profiles']).data
        assert isinstance(self.profiles, pandas.DataFrame)
        print('Beamlet.imp_profiles read from file: ' + hdf5_path)

    def __initialize_ode(self):
        self.coefficient_matrix = CoefficientMatrix(self.param, self.profiles, self.components, self.atomic_db)
        self.initial_condition = [self.__get_linear_density()] + [0.] * (self.atomic_db.atomic_ceiling - 1)

    def __get_linear_density(self):
        current = float(self.param.getroot().find('body').find('beamlet_current').text)
        return current / (self.atomic_db.velocity * self.const.charge_electron)

    def __solve_numerically(self, store_data=True):
        if self.coefficient_matrix is None or self.initial_condition is None:
            self.__initialize_ode()

        ode = Ode(coeff_matrix=self.coefficient_matrix.matrix, init_condition=self.initial_condition)
        numerical = ode.calculate_numerical_solution(self.profiles['beamlet grid']['distance']['m'])
        if store_data:
            for level in range(self.atomic_db.atomic_ceiling):
                label = 'level ' + self.atomic_db.inv_atomic_dict[level]
                self.profiles[label] = numerical[:, level]
            return
        else:
            return numerical

    def calculate_beamevolution(self, solver):
        assert isinstance(solver, str)
        if solver == 'numerical':
            self.__solve_numerically()
        elif solver == 'analytical':
            raise NotImplementedError('Analytical solver not yet implemented.')
        elif solver == 'disregard':
            print('Beam evolution not calculated.')
            return
        else:
            raise Exception('The numerical solver: ' + solver + ' is not supported. '
                                                                'Supported solvers are: numerical, analytical, disregard.')

    def __was_beamevolution_performed(self):
        try:
            dummy = self.profiles['level ' + self.atomic_db.set_default_atomic_levels()[2]]
            return True
        except KeyError:
            return False

    def compute_linear_emission_density(self, to_level=None, from_level=None):
        if to_level is None or from_level is None:
            from_level, to_level, ground_level, transition_label = self.atomic_db.set_default_atomic_levels()
        if isinstance(to_level, str) and isinstance(from_level, str):
            if self.atomic_db.atomic_dict[to_level] >= self.atomic_db.atomic_dict[from_level]:
                raise Exception('Dude! Please stop screwing around. '
                                'Electrons spontaneously transit from higher to lower atomic states.')
        else:
            raise Exception('The expected input for atomic transitions are strings. '
                            'Bundled-n for H,D,T beam species ex:[1, 2, ... 6]. '
                            'l-n resolved labels for Li ex: [2s, 2p, ... 4f] and Na ex: [3s, 3p, ... 5s]')
        if self.__was_beamevolution_performed():
            transition_label = from_level + '-->' + to_level
            self.profiles[transition_label] = \
                self.profiles['level ' + from_level] * self.atomic_db.spontaneous_trans[self.atomic_db.atomic_dict
                [to_level], self.atomic_db.atomic_dict[from_level]]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def compute_linear_density_attenuation(self):
        if self.__was_beamevolution_performed():
            self.profiles['linear_density_attenuation'] = self.profiles['level ' + self.atomic_db.inv_atomic_dict[0]]
            for level in range(1, self.atomic_db.atomic_ceiling):
                self.profiles['linear_density_attenuation'] += self.profiles['level ' +
                                                                             self.atomic_db.inv_atomic_dict[level]]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def compute_relative_populations(self, reference_level=None):
        if self.__was_beamevolution_performed():
            if reference_level is None:
                reference_level = self.atomic_db.set_default_atomic_levels()[2]
            assert isinstance(reference_level, str)
            for level in range(self.atomic_db.atomic_ceiling):
                self.profiles['rel.pop ' + self.atomic_db.inv_atomic_dict[level]] = \
                    self.profiles['level ' + self.atomic_db.inv_atomic_dict[level]] / \
                    self.profiles['level ' + reference_level]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def copy(self, object_copy='full'):
        if not isinstance(object_copy, str):
            raise TypeError('The expected data type for <object_copy> is str.')
        if object_copy == 'full':
            return deepcopy(self)
        elif object_copy == 'without-results':
            beamlet = deepcopy(self)
            beamlet.profiles = self._copy_profiles_input()
            return beamlet
        else:
            raise ValueError('The <object_copy> variable does not support ' + object_copy)

    def _copy_profiles_input(self):
        profiles = np.zeros((1 + len(self.components) * 2, len(self.profiles)))
        type_labels = []
        property_labels = []
        unit_labels = []

        profiles[0, :] = self.profiles['beamlet grid']['distance']['m']
        type_labels.append('beamlet grid')
        property_labels.append('distance')
        unit_labels.append('m')
        count = 1

        for component in self.components.T:
            type_labels.append(str(component))
            property_labels.append('density')
            unit_labels.append('m-3')
            profiles[count, :] = self.profiles[str(component)]['density']['m-3']

            type_labels.append(str(component))
            property_labels.append('temperature')
            unit_labels.append('eV')
            profiles[count + 1, :] = self.profiles[str(component)]['temperature']['eV']

            count += 2

        profiles = np.swapaxes(profiles, 0, 1)
        row_index = [i for i in range(len(self.profiles))]
        column_index = pandas.MultiIndex.from_arrays([type_labels, property_labels, unit_labels],
                                                     names=['type', 'property', 'unit'])
        return pandas.DataFrame(data=profiles, columns=column_index, index=row_index)

    def fluctuation_response(self, type_of_fluct = 'Gauss', num_of_fluct = 1, positions = [], absolute_fluct = False ,
                             fluct_size = [0.1], fwhm = [0.01], component = 'electron', Temp_fluct = False, rel_pop = False, diagnostics = False):
        root = self.param.getroot()
        beamlet_species_child = root.find(".//beamlet_species")
        if beamlet_species_child.attrib == 'He': #Vizsgált elektronpálya eltolások
            H = 1
        else:
            H = 0
        if(len(fluct_size) != num_of_fluct):
            raise ValueError('The number of given amplitudes does not equal the number of fluctuations provided')
        if(len(fwhm) != num_of_fluct):
            raise ValueError('The number of given amplitudes does not equal the number of fluctuations provided')
        if (positions == []):
            prof_length = max(self.profiles['beamlet grid']['distance']['m'])
            for i in range(num_of_fluct):
                positions.append(prof_length/num_of_fluct * (0.5+i))
        else:
            if(len(positions) != num_of_fluct):
                print(len(positions))
                print(num_of_fluct)
                raise ValueError('The number of given positions does not equal the number of fluctuations provided')
        levels = list(self.atomic_db.atomic_dict.keys())
        self.compute_linear_emission_density(levels[1+H], levels[2+H])
        response = []
        self.compute_linear_emission_density(levels[1+H], levels[2+H])
        original = self.profiles[levels[2+H] + '-->' + levels[1+H]]
        pos = []
        for i in range(num_of_fluct):
            for j in range(len(self.profiles[str(component)]['density']['m-3'])):
                    if (self.profiles['beamlet grid']['distance']['m'][j] > positions[i]):
                        pos = j
                        break
            if absolute_fluct:
                relative_amp = fluct_size[i]/self.profiles[str(component)]['density']['m-3'][pos]
            else:
                relative_amp = fluct_size[i]
                f,beam = self.fluctuation_addition(type_of_fluct, relative_amp, fwhm[i], pos, component, Temp_fluct, H, diagnostics)
            if diagnostics:
                levels = list(beam.atomic_db.atomic_dict.keys())
                diff = beam.profiles[levels[2+H] + '-->' + levels[1+H]] - original
                maximum = max(diff)
                for k in range(len(diff)):
                    if diff[k] >= maximum:
                        orig_max = original[k]
                response.append(diff)
            else:
                diff = f-original
                maximum = max(diff)
                for i in range(len(diff)):
                    if diff[i] >= maximum:
                        orig_max = original[i]
                response.append(diff)
        if rel_pop:
            if num_of_fluct == 1:
                beam.compute_relative_populations()
                return beam, response
            else:
                raise ValueError('The relative population output only works for singular fluctuations at the moment. Please handle individual fluctuations separately.')
        return response, orig_max

    def fluctuation_addition(self, type_of_fluct, relative_amp, fwhm, pos, component, Temp_fluct, H, diagnostics=False):
        beamlet = self.copy(object_copy='without-results')
        beamlet.add_density_fluctuation(type_of_fluct, relative_amp, fwhm, pos, component, Temp_fluct)
        beamlet.__initialize_ode()
        beamlet.calculate_beamevolution(solver='numerical')
        levels = list(beamlet.atomic_db.atomic_dict.keys())
        if diagnostics:
            label_font = 14
            title_font = 16
            tick_font = 12
            plt.plot(beamlet.profiles['beamlet grid']['distance']['m'],
                     beamlet.profiles[str(component)]['density']['m-3'],
                     label='Fluktuáló Plazma sűrűség')
            plt.plot(beamlet.profiles['beamlet grid']['distance']['m'],self.profiles[str(component)]['density']['m-3'],
                     label='Stacioner Plazma sűrűség')
            plt.title("Nyalábmenti sűrűségprofil", fontsize=title_font, fontweight='bold')
            plt.xlabel("Nyalábmenti távolság (m)", fontsize=label_font, fontweight= 'bold')
            plt.ylabel("Plazma sűrűség (1/m^3)", fontsize=label_font, fontweight='bold')
            plt.tick_params(axis='both', labelsize=tick_font)
            plt.legend()
            plt.show()
            beamlet.compute_linear_emission_density(levels[1+H], levels[2+H])
            plt.plot(beamlet.profiles['beamlet grid']['distance']['m'],beamlet.profiles[levels[2+H] + '-->' + levels[1+H]],
                     label='Fluktuáló Emisszió profil')
            plt.plot(beamlet.profiles['beamlet grid']['distance']['m'], self.profiles[levels[2+H] + '-->' + levels[1+H]],
                     label='Stacioner Emisszió profil')
            plt.title("Fényválasz profil", fontsize=title_font, fontweight='bold')
            plt.xlabel("Nyalábmenti távolság (m)", fontsize=label_font, fontweight='bold')
            plt.ylabel("Fényválasz intenzitás (photon/(s*m))", fontsize=label_font, fontweight='bold')
            plt.tick_params(axis='both', labelsize=tick_font)
            plt.legend()
            plt.show()
            response = beamlet.profiles[levels[2 + H] + '-->' + levels[1 + H]]
            return response,beamlet
        else:
            beamlet.compute_linear_emission_density(levels[1+H],levels[2+H])
            response = beamlet.profiles[levels[2+H] + '-->' + levels[1+H]]  # a szint még kérdéses, temp solution
            return response,beamlet

    def add_density_fluctuation(self, type_of_fluct, relative_amplitude, fwhm, pos,
                                component, Temp_fluct, hole = False):  # lyuk számolás implementálása
        density_amp = relative_amplitude * self.profiles[str(component)]['density']['m-3'][pos]
        temp_amp = relative_amplitude * self.profiles[str(component)]['temperature']['eV'][pos]
        position = self.profiles['beamlet grid']['distance']['m'][pos]
        if hole:
            sign = -1
        else:
            sign = 1

        if (type_of_fluct == 'Gauss'):
            theta = fwhm / (2 * np.sqrt(2 * np.log(2)))
            for i in range(len(self.profiles['beamlet grid']['distance']['m'])):
                dx = abs(position - self.profiles['beamlet grid']['distance']['m'][i])
                self.profiles[str(component)]['density']['m-3'][i] = self.profiles[str(component)]['density']['m-3'][
                                                                         i] + sign*density_amp * np.exp(
                    -1 * 0.5 * pow(dx, 2) / pow(theta, 2))
                if Temp_fluct:
                    self.profiles[str(component)]['temperature']['eV'][i] = \
                    self.profiles[str(component)]['temperature']['eV'][i] + sign*temp_amp * np.exp(-1 * 0.5 * pow(dx, 2) / pow(theta, 2))
        elif (type_of_fluct == 'Hann'):
            L = 1 / density_amp
            for i in range(len(self.profiles['beamlet grid']['distance']['m'])):
                dx = abs(position - self.profiles['beamlet grid']['distance']['m'][i])
                if dx < L:
                    self.profiles[str(component)]['density']['m-3'][i] = self.profiles[component]['density']['m-3'][
                                                                             i] + sign*density_amp * pow(
                        np.cos(np.pi * dx * density_amp), 2)
                    if Temp_fluct:
                        self.profiles[str(component)]['temperature']['eV'][i] = \
                        self.profiles[component]['temperature']['eV'][i] + sign*temp_amp * pow(np.cos(np.pi * dx * density_amp),2)
        else:
            raise ValueError('This function does not support ' + type_of_fluct + ' type fluctuations.')

    def save_output(self):
        rel_pops = []
        for level in range(self.atomic_db.atomic_ceiling):
            rel_pops.append(self.profiles['rel.pop ' + self.atomic_db.inv_atomic_dict[level]])
        return rel_pops
