import os
from lxml import etree
import urllib
import paramiko

DEFAULT_SETUP = 'getdata_setup.xml'


class AccessData(object):
    def __init__(self, data_path_name):
        self.access_path = ''
        self._read_setup()
        self._set_private_connection()

        if data_path_name is not None:
            self.data_path_name = data_path_name
            self._path_setup()

    def _read_setup(self, setup_path_name=None):

        if not setup_path_name:
            setup_path_name = os.path.join(os.path.dirname(__file__), DEFAULT_SETUP)

        tree = etree.parse(setup_path_name)
        body = tree.getroot().find('body')
        self.dummy_directory = body.find('dummy_directory').text
        self.common_local_data_directory = os.path.join(os.path.dirname(__file__), '..',
                                                        body.find('common_local_data_directory').text)
        self.user_local_data_directory = os.path.join(os.path.dirname(__file__), '..',
                                                      body.find('user_local_data_directory').text)
        self.server_address = body.find('server_address').text
        self.server_user = body.find('user_name').text
        self.server_private_access = body.find('server_private_data').text
        self.server_public_access = body.find('server_public_data').text
        self.server_public_address = body.find('server_public_address').text
        self.contact_address = body.find('contact_address').text
        self.private_key_path = body.find('private_key').text

    def _set_private_connection(self):
        self._set_private_key()
        self.ssh_connection = paramiko.SSHClient()
        self.ssh_connection.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def _set_private_key(self):
        key_path = os.path.join(os.path.dirname(__file__), '..', self.private_key_path)
        if os.path.isfile(key_path):
            try:
                self.private_key = paramiko.RSAKey.from_private_key_file(key_path)
            except paramiko.SSHException:
                self.private_key = None
        else:
            self.private_key = None

    def _path_setup(self):
        self._local_path_setup()
        self._server_path_setup()

    def _local_path_setup(self, local_path=None):
        if local_path is None:
            local_path = self.data_path_name
        elif not isinstance(local_path, str):
            raise TypeError('File local path input is expected to be of str type.')
        self.common_local_data_path = os.path.join(self.common_local_data_directory, local_path)
        self.user_local_data_path = os.path.join(self.user_local_data_directory, local_path)
        self.user_local_dummy_path = os.path.join(self.user_local_data_directory,
                                                  self.dummy_directory, local_path)

    def _server_path_setup(self, server_path=None):
        if server_path is None:
            server_path = self.data_path_name
        elif not isinstance(server_path, str):
            raise TypeError('File local path input is expected to be of str type.')
        self.server_public_path = self.server_public_address + '/' + server_path
        self.server_private_path = self.server_private_access + '/' + server_path

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

    def check_public_server_data_path(self):
        request = urllib.request.Request(self.server_public_path)
        try:
            response = urllib.request.urlopen(request)
            return True
        except urllib.error.HTTPError:
            return False

    def check_private_server_data_path(self):
        if self.private_key is not None:
            try:
                self.ssh_connection.connect(self.server_address, username=self.server_user, pkey=self.private_key)
                sftp = self.ssh_connection.open_sftp()
                message = sftp.stat(self.server_private_path)
                status = True
            except FileNotFoundError:
                status = False
            finally:
                sftp.close()
                self.ssh_connection.close()
                return status
        else:
            return False

    def contact_us(self):
        print('\nFor further info and data please contact us: \n\tmailto:' + self.contact_address)
