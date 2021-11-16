import pandas as pd
from lxml import etree
import numpy as np
from crm_solver.atomic_db import AtomicDB


class AtomicInput(object):
    def __init__(self, energy, projectile, source, current, param_name):
        self.energy = energy
        self.projectile = projectile
        self.data_source = source
        self.current = current
        self.param_name = param_name
        self.components = None
        self.param = None
        self.component_col = ['q', 'Z', 'A', 'Molecule']
        self.component_index = []
        self.component_data = []
        self._build_param()

    def _build_param(self):
        xml = etree.Element('xml')
        head = etree.SubElement(xml, 'head')
        id_tag = etree.SubElement(head, 'id')
        id_tag.text = self.param_name
        body_tag = etree.SubElement(xml, 'body')
        beamlet_energy = etree.SubElement(body_tag, 'beamlet_energy', {'unit': 'keV'})
        beamlet_energy.text = str(self.energy)
        beamlet_species = etree.SubElement(body_tag, 'beamlet_species')
        beamlet_species.text = self.projectile
        beamlet_source = etree.SubElement(body_tag, 'beamlet_source')
        beamlet_source.text = self.data_source
        beamlet_current = etree.SubElement(body_tag, 'beamlet_current', {'unit': 'A'})
        beamlet_current.text = str(self.current)
        self.param = etree.ElementTree(element=xml)

    def _build_components(self):
        if len(self.component_data) == 0:
            raise ValueError('The component data base was not initiated.')
        else:
            self.components = pd.DataFrame(np.array(self.component_data), columns=self.component_col,
                                           index=self.component_index)

    def _next_element(self, name):
        return name + str(len([component for component in self.component_index if name in component])+1)

    def add_target_component(self, charge, atomic_number, mass_number, molecule_name=None):
        element = [charge, atomic_number, mass_number, molecule_name]
        if charge == -1:
            if 'electron' in self.component_index:
                print('Already a component.')
            else:
                self.component_index.append('electron')
                self.component_data.append(element)
        elif charge == 0:
            if element in self.component_data:
                print('Already a component.')
            else:
                self.component_index.append(self._next_element('neutral'))
                self.component_data.append(element)
        elif charge >= 1:
            if element in self.component_data:
                print('Already a component.')
            else:
                self.component_index.append(self._next_element('ion'))
                self.component_data.append(element)

    def get_atomic_db_input(self):
        self._build_components()
        return self.param, self.components

    def generate_atomic_db(self, atomic_source='renate', max_level=False, rate_type='default'):
        return AtomicDB(atomic_source=atomic_source, param=self.param, components=self.components,
                        atomic_ceiling=max_level, rate_type=rate_type)


class BeamletInput(AtomicInput):
    def __init__(self, energy, projectile, source, current, param_name):
        AtomicInput.__init__(self, energy=energy, projectile=projectile, source=source,
                             current=current, param_name=param_name)
        self.profiles = None
        self.grid = None
        self.profile_data = []
        self.type_labels = []
        self.property_labels = []
        self.unit_labels = []

    def _build_profiles(self):
        profiles = np.swapaxes(np.array(self.profile_data), 0, 1)
        row_index = [i for i in range(len(self.grid))]
        column_index = pd.MultiIndex.from_arrays([self.type_labels, self.property_labels, self.unit_labels],
                                                 names=['type', 'property', 'unit'])
        self.profiles = pd.DataFrame(data=profiles, columns=column_index, index=row_index)

    def add_grid(self, grid):
        self.type_labels.append('beamlet grid')
        self.property_labels.append('distance')
        self.unit_labels.append('m')
        self.grid = np.array(grid)
        self.profile_data.append(self.grid)

    def add_target_profiles(self, charge, atomic_number, mass_number, molecule_name, density, temperature):
        if self.grid is None:
            print('The profiles grid has not been added first.')
        elif (len(self.grid) != len(density)) or (len(self.grid) != len(temperature)):
            print('The density or temperature profiles do not match with the provided grid.')
        else:
            self.add_target_component(charge=charge, atomic_number=atomic_number,
                                      mass_number=mass_number, molecule_name=molecule_name)

            self.type_labels.append(self.component_index[-1])
            self.property_labels.append('density')
            self.unit_labels.append('m-3')
            self.profile_data.append(density)

            self.type_labels.append(self.component_index[-1])
            self.property_labels.append('temperature')
            self.unit_labels.append('eV')
            self.profile_data.append(temperature)

    def get_beamlet_input(self):
        self._build_components()
        self._build_profiles()
        return self.param, self.components, self.profiles

    def from_text(self, source, charges, atomic_numbers, mass_numbers, molecular_numbers):
        pass
