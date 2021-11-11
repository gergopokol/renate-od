import pandas as pd
from lxml import etree
import numpy as np


class BeamletInput(object):
    def __init__(self, energy, projectile, source):
        self.energy = energy
        self.projectile = projectile
        self.source = source

    def _build_param(self):
        pass

    def _build_components(self):
        pass

    def _build_profiles(self):
        pass

    def add_grid(self, grid):
        self.grid = np.array(grid)

    def add_target(self, charge, atomic_number, mass_number, molecular_number, density, temperature):
        pass

    def make_input(self):
        pass

    def from_text(self, source, charges, atomic_numbers, mass_numbers, molecular_numbers):
        pass
