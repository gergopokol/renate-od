# van-e a közös lokálisban
# van-e lokálisan
# ssh-val letölti
# error handling, ha nincs:
# van-e lokálisan dummy data -> only test get contact
# ha nincs, public-ból leszedi a dummyt -> only test get contact
# ha nincs hiba: get contact

import os
import urllib.request

def get_data(data_name):

    dummy_name = "dummy.txt"
    common_local_data_directory = "."
    user_local_data_directory = "."
    server_data_ip = "data@deep.reak.bme.hu"
    server_data_subdir = "renate-od"
    server_public_url = "deep.reak.bme.hu/~data"

    common_local_data_path = common_local_data_directory + os.sep + data_name
    user_local_data_path = user_local_data_directory + os.sep + data_name

    common_local_dummy_path = common_local_data_directory + os.sep + dummy_name
    user_local_dummy_path = user_local_data_directory + os.sep + dummy_name

    server_data_path = server_data_ip + ":" + server_data_subdir + "/" + data_name
    server_public_path = server_public_url + "/" + server_data_subdir + "/" + data_name

    if os.path.isfile(common_local_data_path):
        return_path = common_local_data_path
        print('Data is located in the common local directory (' + return_path + ')')

    else:
        if os.path.isfile(user_local_data_path):
            return_path = user_local_data_path
            print('Data is located in the user local directory (' + return_path + ')')

        else:
            if os.name == 'posix':
                scp_answer = os.system('scp "%s" "%s"' % (server_data_path, user_local_data_path))

            else:
                scp_answer = os.system('winscp.exe "%s" "%s"' % (server_data_path, user_local_data_path))

            if scp_answer==1:
                return_path = user_local_data_path
                print('Data is downloaded from the server to the user local directory (' + return_path + ')')

            else:
                if os.path.isfile(common_local_dummy_path):
                    return_path = common_local_dummy_path
                    print('Common dummy data is used (' + return_path + ')')
                    contact_with_us()

                elif os.path.isfile(user_local_dummy_path):
                    return_path = user_local_dummy_path
                    print('User dummy data is used (' + return_path + ')')
                    contact_with_us()

                else:
                    try:
                        urllib.request.urlretrieve(server_public_path, user_local_dummy_path)
                        return_path = user_local_dummy_path
                        print('Dummy data is downloaded to the user local directory (' + return_path + ')')
                        contact_with_us()

                    except:
                        print('ERROR: Download was not successful!')
                        contact_with_us()

def contact_with_us():
    print('\nFor further info and data please contact us: \n\tmailto:pokol@reak.bme.hu')

get_data("data.txt")