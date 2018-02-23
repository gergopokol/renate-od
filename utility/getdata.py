import os
import urllib.request


class GetData:
    def __init__(self,
                 data_path_name="test.txt"):

        # Set constants that can later be read from setup files
        self.dummy_directory = "dummy"
        self.common_local_data_directory = "/data"
        self.user_local_data_directory = "../data"
        self.server_private_address = "data@deep.reak.bme.hu:private_html/renate-od"
        self.server_public_address = "deep.reak.bme.hu/~data/renate-od"
        self.contact_address = "pokol@reak.bme.hu"

        # Assemble paths
        self.data_path_name = data_path_name
        self.common_local_data_path = self.common_local_data_directory + os.sep + data_path_name
        self.user_local_data_path = self.user_local_data_directory + os.sep + data_path_name
        self.user_local_dummy_path = self.user_local_data_directory + os.sep + self.dummy_directory

        self.access_path = ''

    def get_data(self):
        if self.check_common_local_data_path():
            return
        if self.check_user_local_data_path():
            return
        if self.get_private_data():
            return
        if self.check_user_local_dummy_path():
            return
        if self.get_public_data():
            return
        else:
            self.contact_with_us()

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
        if os.name == 'posix':
            scp_answer = os.system('scp "%s" "%s"' % (server_private_path, self.user_local_data_path))
        else:
            scp_answer = os.system('winscp.exe "%s" "%s"' % (server_private_path, self.user_local_data_path))
        if scp_answer == 1:
            self.access_path = self.user_local_data_path
            print('Data has been downloaded from the server to the user local directory (' + self.access_path + ')')
            return True
        else:
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
        try:
            urllib.request.urlretrieve(server_public_path, self.user_local_dummy_path)
            self.access_path = self.user_local_dummy_path
            print('Warning: Dummy data has been downloaded to the user local directory (' + self.access_path + ')!')
            return True
        except:
            print('ERROR: Download was not successful!')
            return False


GetData("data.txt")
