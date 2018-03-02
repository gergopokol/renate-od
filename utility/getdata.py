import os
import urllib.request
import utility.settings


class GetData:
    """
    This class is to access and load data from files. It looks for data in the following order:
    1. Common local data path
    2. User's local data path
    3. Remote private data path (downloads if accessable)
    4. Local user dummy data path
    5. Remote public dummy data path (downloads if accessable)

    data_path_name should include relative path from RENATE-OD root directory
    Paths are read from utility/getdata_setup.xml file.
    """

    def __init__(self,
                 data_path_name="test.txt",
                 source=""):

        # Read info from setup file.
        setup = utility.settings.Settings(filename='utility/getdata_setup.xml')
        self.dummy_directory = setup.return_text(element_name='dummy_directory')
        self.common_local_data_directory = setup.return_text(element_name='common_local_data_directory')
        self.user_local_data_directory = setup.return_text(element_name='user_local_data_directory')
        self.server_private_address = setup.return_text(element_name='server_private_address')
        self.private_key = setup.return_text(element_name='private_key')
        self.server_public_address = setup.return_text(element_name='server_public_address')
        self.contact_address = setup.return_text(element_name='contact_address')

        self.data_path_name = data_path_name
        self.source = source
        self.common_local_data_path = self.common_local_data_directory + '/' + data_path_name
        self.user_local_data_path = self.user_local_data_directory + '/' + data_path_name
        self.user_local_dummy_path = self.user_local_data_directory + '/' + self.dummy_directory + '/' + data_path_name

        self.access_path = ''
        self.data = ''
        self.read_data()

    def read_data(self):
        """
        Reads data into the self.data property. Data can be narrowed down by the self.source variable.
        :return: True if successful
        """
        if self.get_data():
            extension = os.path.splitext(self.data_path_name)[1]
            if extension == '.hd5':
                import utility.get_data_from_hdf5
                self.data = utility.get_data_from_hdf5.get_data_from_hdf5(self.access_path, self.source)
                return True
            elif extension == '.txt':
                file = open(self.access_path, 'r')
                self.data = file.read()
                return True
            else:
                raise NameError('Unknown extension. Unable to load!')

    def get_data(self):
        """
        Looks for data file at different locations.
        :return: True if successful
        """
        if self.check_common_local_data_path():
            return True
        elif self.check_user_local_data_path():
            return True
        elif self.get_private_data():
            return True
        elif self.check_user_local_dummy_path():
            return True
        elif self.get_public_data():
            return True
        else:
            print('Error: No data source available!')
            self.contact_with_us()
            return False

    def contact_with_us(self):
        print('\nFor further info and data please contact us: \n\tmailto:' + self.contact_address)

    def check_common_local_data_path(self):
        if os.path.isfile(self.common_local_data_path):
            self.access_path = self.common_local_data_path
            print('Data is located in the common local directory (' + self.access_path + ')')
            return True
        else:
            return False

    def check_user_local_data_path(self):
        if os.path.isfile(self.user_local_data_path):
            self.access_path = self.user_local_data_path
            print('Data is located in the user local directory (' + self.access_path + ')')
            return True
        else:
            return False

    def get_private_data(self):
        server_private_path = self.server_private_address + "/" + self.data_path_name
        print('Attempting to download from server: ' + server_private_path)
        try:
            if os.name == 'posix':
                scp_answer = os.system('scp -i "%s" "%s" "%s"' % (self.private_key, server_private_path,
                                                                  self.user_local_data_path))
            else:
                scp_answer = os.system('pscp -scp -i "%s" "%s" "%s"' % (self.private_key, server_private_path,
                                                                        self.user_local_data_path))
        except:
            scp_answer = 1
        if scp_answer == 0:
            self.access_path = self.user_local_data_path
            print('Data has been downloaded from the server to the user local directory (' + self.access_path + ')')
            return True
        else:
            print('Warning: Could not read data from server: ' + server_private_path + '!')
            return False

    def check_user_local_dummy_path(self):
        if os.path.isfile(self.user_local_dummy_path):
            self.access_path = self.user_local_dummy_path
            print('Warning: Dummy data is used from the user local directory (' + self.access_path + ')!')
            return True
        else:
            return False

    def get_public_data(self):
        server_public_path = self.server_public_address + "/" + self.data_path_name
        print('Attempting to download dummy data from public server: ' + server_public_path)
        try:
            self.ensure_dir(self.user_local_dummy_path)
            urllib.request.urlretrieve(server_public_path, self.user_local_dummy_path)
            self.access_path = self.user_local_dummy_path
            print('Warning: Dummy data has been downloaded to the user local directory (' + self.access_path + ')!')
            return True
        except:
            print('Warning: Could not read data from ' + server_public_path + '!')
            return False

    @staticmethod
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
