import pandas as pd
from lxml import etree
import numpy as np


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

    def add_target_component(self, charge, atomic_number, mass_number, molecule_name):
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


class BeamletInput(AtomicInput):
    def __init__(self, energy, projectile, source, current, param_name):
        AtomicInput.__init__(self, energy=energy, projectile=projectile, source=source,
                             current=current, param_name=param_name)
        self.profiles = None
        self.grid = None
        self.profile_data = []

    def _build_profiles(self):
        pass

    def add_grid(self, grid):
        self.grid = np.array(grid)

    def add_target_profiles(self, charge, atomic_number, mass_number, molecule_name, density, temperature):
        self.add_target_component(charge=charge, atomic_number=atomic_number,
                                  mass_number=mass_number, molecule_name=molecule_name)

    def get_beamlet_input(self):
        pass

    def from_text(self, source, charges, atomic_numbers, mass_numbers, molecular_numbers):
        pass
