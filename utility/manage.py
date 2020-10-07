import os
import re
from lxml import etree
from crm_solver.beamlet import Beamlet
from utility.writedata import WriteData
from utility.putdata import PutData
from utility.getdata import GetData
from utility.accessdata import AccessData
from utility.exceptions import RenateAuthorizedUserError


class Version(object):
    def __init__(self, value):
        if not isinstance(value, str):
            raise TypeError('The expected variable type for Version control is str.')
        version = value.split(sep='.')
        if not version.__len__() == 3:
            raise ValueError('The expected data value is point delimited with 3 expressions. Ex: 1.1.0 ')
        for val in version:
            assert isinstance(int(val), int), 'Delimited values are expected to be convertible to <int>.'
        self.major_version = int(version[0])
        self.minor_version = int(version[1])
        self.bugfix_version = int(version[2])
        self.version = version

    def __repr__(self):
        return 'Version: ' + self.version[0] + '.' + self.version[1] + '.' + self.version[2]

    def __eq__(self, other):
        return (self.major_version == other.major_version) and (self.minor_version == other.minor_version) and \
               (self.bugfix_version == other.bugfix_version)

    def __ne__(self, other):
        return (self.major_version != other.major_version) or (self.minor_version != other.minor_version) or \
               (self.bugfix_version != other.bugfix_version)

    def __lt__(self, other):
        for i in range(self.version.__len__()):
            if self.version[i] < other.version[i]:
                return True
            elif self.version[i] < other.version[i]:
                return False
            elif i+1 == self.version.__len__():
                return False

    def __gt__(self, other):
        for i in range(self.version.__len__()):
            if self.version[i] > other.version[i]:
                return True
            elif self.version[i] < other.version[i]:
                return False
            elif i+1 == self.version.__len__():
                return False

    def __str__(self):
        return self.version[0] + '.' + self.version[1] + '.' + self.version[2]

    def release_major_version(self):
        self.major_version += 1
        self.minor_version = 0
        self.bugfix_version = 0
        self.version = [str(self.major_version), str(self.minor_version), str(self.bugfix_version)]

    def release_minor_version(self):
        self.minor_version += 1
        self.bugfix_version = 0
        self.version[1] = str(self.minor_version)
        self.version[2] = str(self.bugfix_version)

    def release_bugfix_version(self):
        self.bugfix_version += 1
        self.version[2] = str(self.bugfix_version)


class CodeInfo(object):
    def __init__(self):
        self.code_info_path = os.path.join(os.path.dirname(__file__), '..', 'RENATE-OD.info')
        self._load_code_info()
        self._check_authorised_user()

    def _load_code_info(self):
        self.code_info_tree = etree.parse(self.code_info_path)
        self.code_name = self.code_info_tree.find('head').find('code_name').text
        self.code_abbreviation = self.code_info_tree.find('head').find('code_abbreviation').text
        self.code_license = self.code_info_tree.find('body').find('code_license').text
        self.code_version = Version(self.code_info_tree.find('body').find('code_version').text)
        self.code_install_requirements_path = self.code_info_tree.find('body').find('code_install_requirement').text
        self.code_package = self.code_info_tree.find('body').find('code_package').text
        self.code_git_link = self.code_info_tree.find('body').find('code_git_link').text
        self._load_classifiers()
        self._load_install_requirements()

    def _load_classifiers(self):
        list_of_classifiers = ['classifier_audience', 'classifier_natural_language',
                               'classifier_programming_language', 'classifier_topic']
        self.classifiers = []
        for classifier in list_of_classifiers:
            self.classifiers.append(self.code_info_tree.find('body').find(classifier).text)

    def _load_install_requirements(self):
        requirements_path = os.path.join(os.path.dirname(__file__), '..', self.code_install_requirements_path)
        self.code_requirements = {}
        with open(requirements_path) as requirements_file:
            lines = requirements_file.readlines()

            for line in lines:
                line_content = re.split('==|<=|>=', line)
                self.code_requirements.update({line_content[0]: Version(line_content[1])})

    def _check_authorised_user(self):
        access = AccessData(data_path_name=None)
        if access.private_key is None:
            self.authorization = False
        else:
            self.authorization = True

    def update_data(self, attribute, value=None):
        if not isinstance(attribute, str):
            raise TypeError('The <attribute> input is expected to be a str.')
        if self.authorization:
            if hasattr(self, attribute):
                if value is not None:
                    setattr(self, attribute, value)
                self.code_info_tree.find('body').find(attribute).text = str(getattr(self, attribute))
                self.code_info_tree.write(self.code_info_path)
                print('Updates XML file by rewriting: ' + attribute + ' with: ' + str(getattr(self, attribute)))
                return True
            else:
                raise AttributeError('The CodeInfo object has no attribute: ' + attribute)
        else:
            raise RenateAuthorizedUserError('You are not an authorized user to perform changes on Code information!')


class Release(object):
    def __init__(self):
        self.info = CodeInfo()
        self.test_cases = ['scenario-standard_plasma-H_energy-100_beam-H_profile',
                           'scenario-standard_plasma-H_energy-100_beam-D_profile',
                           'scenario-standard_plasma-H_energy-100_beam-T_profile',
                           'scenario-standard_plasma-H_energy-100_beam-Li_profile',
                           'scenario-standard_plasma-H_energy-100_beam-Na_profile']
        self.test_path = 'test_dataset/crm_systemtests'
        self.write = WriteData()

    def _update_code_version(self, release, version):
        if release is None and version is None:
            raise ValueError('No input for code Version or Release type was given! Please try again.')
        if isinstance(release, str):
            if release == 'major':
                self.info.code_version.release_major_version()
                self.info.update_data('code_version')
            elif release == 'minor':
                self.info.code_version.release_minor_version()
                self.info.update_data('code_version')
            elif release == 'bugfix':
                self.info.code_version.release_bugfix_version()
                self.info.update_data('code_version')
            else:
                raise ValueError('Requested version release protocol: ' + release + ' is not supported.')
        elif isinstance(version, str):
            ver = Version(version)
            if ver > self.info.code_version:
                self.info.update_data('code_version', ver)
            else:
                raise ValueError('The provided version number: ' + str(ver) + ' is lower or equal to the '
                                 'actual: ' + str(self.info.code_version) + ' version number.')
        else:
            raise TypeError('Neither <version> or <release> variables provided are of '
                            '<str> type or of supported value')

    def _compute_all_benchmarks(self):
        for test_case in self.test_cases:
            test_path = self.test_path + '/actual/' + test_case + '.xml'
            reference = Beamlet(data_path=test_path, solver='disregard')
            actual_source = reference.copy(object_copy='without-results')
            actual = Beamlet(param=actual_source.param, profiles=actual_source.profiles,
                             components=actual_source.components, atomic_db=actual_source.atomic_db, solver='numerical')
            actual.compute_linear_density_attenuation()
            actual.compute_linear_emission_density()
            actual.compute_relative_populations()
            self.write.write_beamlet_profiles(actual, subdir='release/')

    def _relocate_benchmarks_from_actual_to_archive_on_server(self):
        pass

    def _upload_actual_benchmarks(self):
        pass

    def execute_release(self, release=None, version=None):
        self._update_code_version(release, version)
        self._compute_all_benchmarks()
        self._relocate_benchmarks_from_actual_to_archive_on_server()
        self._upload_actual_benchmarks()
