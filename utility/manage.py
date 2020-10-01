import os
from lxml import etree
from crm_solver.beamlet import Beamlet
from utility.writedata import WriteData
from utility.putdata import PutData
from utility.getdata import GetData
from utility.accessdata import AccessData


class CodeInfo(object):
    def __init__(self):
        self.code_info_path = os.path.join(os.path.dirname(__file__), '..', 'RENATE-OD.info')
        self._load_code_info()
        self._check_authorised_user()

    def _load_code_info(self):
        code_info_tree = etree.parse(self.code_info_path)
        self.code_name = code_info_tree.find('head').find('name').text
        self.code_abbr = code_info_tree.find('head').find('abbreviation').text
        self.code_license = code_info_tree.find('body').find('license').text
        self.code_version = code_info_tree.find('body').find('version').text
        self.code_install_requirements = code_info_tree.find('body').find('install_requirements').text
        self.code_package = code_info_tree.find('body').find('package_name').text
        self.code_git_link = code_info_tree.find('body').find('git').text
        self.classifier_audience = code_info_tree.find('classifiers').find('audience').text
        self.classifier_natural_language = code_info_tree.find('classifiers').find('natural_language').text
        self.classifier_programming_language = code_info_tree.find('classifiers').find('programming_language').text
        self.classifier_topic = code_info_tree.find('classifiers').find('topic').text

    def _check_authorised_user(self):
        access = AccessData(data_path_name=None)
        if access.private_key is None:
            self.authorization = False
        else:
            self.authorization = True

    def update_data(self, attribute, value):
        pass


class Release(object):
    def __init__(self):
        pass

    def _execute_all_beam_evolution_benchmarks(self):
        pass

    def execute_release(self, version):
        pass
