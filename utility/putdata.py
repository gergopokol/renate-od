from utility.accessdata import AccessData


class PutData(AccessData):
    def __init__(self):
        AccessData.__init__(self, data_path_name=None)

    def to_server(self, data_path, server_type='public'):
        self.data_path_name = data_path
        self.path_setup()

        if not isinstance(server_type, str):
            raise TypeError('The requested server type is a str information. Provided data is not.')

        if server_type == 'public':
            return
        elif server_type == 'private':
            return
        else:
            raise ValueError('The requested server type <'+server_type+'> where data is to be ported does not exist!')

    def delete_from_server(self, data_path, server_type='public'):
        self.data_path_name = data_path
        self.path_setup()

        if not isinstance(server_type, str):
            raise TypeError('The requested server type is a str information. Provided data is not.')

        if server_type == 'public':
            return
        elif server_type == 'private':
            return
        else:
            raise ValueError('The requested server type <'+server_type+'> from which data is to be deleted '
                                                                       'does not exist!')

    def move_file(self, from_path, to_path, server_type='public'):
        pass
