import os
from lxml import etree

DEFAULT_SETUP = 'getdata_setup.xml'


class AccessData(object):
    def __init__(self, data_key, data_path_name):
        self.data_key = data_key
        self.access_path = ''
        self.read_setup()

        if data_path_name is not None:
            self.data_path_name = data_path_name
            self.path_setup()

    def read_setup(self, setup_path_name=None):

        if not setup_path_name:
            setup_path_name = os.path.join(os.path.dirname(__file__), DEFAULT_SETUP)

        tree = etree.parse(setup_path_name)
        body = tree.getroot().find('body')
        self.dummy_directory = body.find('dummy_directory').text
        self.common_local_data_directory = os.path.join(os.path.dirname(__file__), '..',
                                                        body.find('common_local_data_directory').text)
        self.user_local_data_directory = os.path.join(os.path.dirname(__file__), '..',
                                                      body.find('user_local_data_directory').text)
        self.server_private_address = body.find('server_private_address').text
        self.private_key = body.find('private_key').text
        self.server_public_address = body.find('server_public_address').text
        self.contact_address = body.find('contact_address').text

    def path_setup(self):
        self.common_local_data_path = os.path.join(self.common_local_data_directory, self.data_path_name)
        self.user_local_data_path = os.path.join(self.user_local_data_directory, self.data_path_name)
        self.user_local_dummy_path = os.path.join(self.user_local_data_directory,
                                                  self.dummy_directory, self.data_path_name)

    def check_user_local_dummy_path(self):
        if os.path.isfile(self.user_local_dummy_path):
            self.access_path = self.user_local_dummy_path
            print('Warning: Dummy data is used from the user local directory: ' + self.user_local_dummy_path)
            return True
        else:
            print('Data is NOT present in the user local dummy directory: ' + self.user_local_dummy_path)
            return False

    def check_common_local_data_path(self):
        if os.path.isfile(self.common_local_data_path):
            self.access_path = self.common_local_data_path
            print('Data is located in the common local directory: ' + self.common_local_data_path)
            return True
        else:
            print('Data is NOT located in the common local directory: ' + self.common_local_data_path)
            return False

    def check_user_local_data_path(self):
        if os.path.isfile(self.user_local_data_path):
            self.access_path = self.user_local_data_path
            print('Data is located in the user local directory: ' + self.user_local_data_path)
            return True
        else:
            print('Data is NOT present in the user local directory: ' + self.user_local_data_path)
            return False

    def contact_us(self):
        print('\nFor further info and data please contact us: \n\tmailto:' + self.contact_address)
