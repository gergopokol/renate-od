from lxml import etree
from utility.getdata import GetData
from crm_solver.beamlet import Beamlet
from imas_utility.idscoreprof import CoreprofIds
from imas_utility.idsequilibrium import EquilibriumIds


class BeamletFromIds:
    def __init__(self, input_source='beamlet/test_imas.xml'):
        self.access_path = input_source

        self.read_imas_xml()
        self.machine = self.param.getroot().find('body').find('imas_machine').text
        self.user = self.param.getroot().find('body').find('imas_user').text
        self.shotnumber = int(self.param.getroot().find('body').find('imas_shotmuner').text)
        self.runnumber = int(self.param.getroot().find('body').find('imas_runmuner').text)

        self.profile_source = self.param.getroot().find('body').find('beamlet_profiles').text
        self.load_imas_profiles()

        self.equilibrium_source = self.param.getroot().find('body').find('device_magnetic_geometry').text
        self.load_imas_equilibrium()

    def get_beamlet_energy(self, energy=None):
        if isinstance(energy, int):


    def get_beamlet_current(self, current=False):
        pass

    def get_beamlet_species(self, species=False):
        pass

    def load_imas_profiles(self):
        if self.profile_source is 'core_profiles':
            self.run_prof = CoreprofIds(self.shotnumber, self.runnumber)
        else:
            print('There is no input protocol for data stored in ' + self.profile_source + ' IDS')

    def load_imas_equilibrium(self):
        if self.equilibrium_source is 'equilibrium':
            self.run_equi = EquilibriumIds(self.shotnumber, self.runnumber)
        else:
            print('There is no input protocol for data stored in ' + self.equilibrium_source + ' IDS')

    def read_imas_xml(self):
        self.param = GetData(data_path_name=self.access_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('ElementTree read to dictionary from: ' + self.access_path)
