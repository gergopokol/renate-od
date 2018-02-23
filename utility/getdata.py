import os
import urllib.request


class GetData:
    def __init__(self,
                 data_path_name="test.txt"):

        # Set constants that can later be read from setup files
        self.dummy_directory = "dummy"
        self.common_local_data_directory = "/data"
        self.user_local_data_directory = "data"
        self.server_private_address = "data@deep.reak.bme.hu:~/private_html/renate-od"
        self.private_key = "data/deep-data.ppk"
        self.server_public_address = "http://deep.reak.bme.hu/~data/renate-od"
        self.contact_address = "pokol@reak.bme.hu"

        # Assemble paths
        self.data_path_name = data_path_name
        self.common_local_data_path = self.common_local_data_directory + '/' + data_path_name
        self.user_local_data_path = self.user_local_data_directory + '/' + data_path_name
        self.user_local_dummy_path = self.user_local_data_directory + '/' + self.dummy_directory + '/' + data_path_name

        self.read_data()

    def read_data(self):
        if self.get_data():
            pass # Read to desired format (pandas.DataFrame?)

    def get_data(self):
        if self.check_common_local_data_path():
            return True
        if self.check_user_local_data_path():
            return True
        if self.get_private_data():
            return True
        if self.check_user_local_dummy_path():
            return True
        if self.get_public_data():
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
                scp_answer = os.system('scp -i "%s" "%s" "%s"' % (self.private_key, server_private_path, self.user_local_data_path))
            else:
                scp_answer = os.system('pscp -scp -i "%s" "%s" "%s"' % (self.private_key, server_private_path, self.user_local_data_path))
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
            print('Warning: Dummy data is used in the user local directory (' + self.access_path + ')!')
            return True
        else:
            return False

    def get_public_data(self):
        server_public_path = self.server_public_address + "/" + self.data_path_name
        print('Attempting to download from server: ' + server_public_path)
        try:
            urllib.request.urlretrieve(server_public_path, self.user_local_dummy_path)
            self.access_path = self.user_local_dummy_path
            print('Warning: Dummy data has been downloaded to the user local directory (' + self.access_path + ')!')
            return True
        except:
            print('Warning: Could not read data from ' + server_public_path + '!')
            return False
