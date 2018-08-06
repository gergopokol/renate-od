from utility.getdata import GetData


class WriteData:
    def __init__(self, root_path = "data/output/beamlet/"):
        self.root_path = root_path

    def write_beamlet_profiles(self, param, profiles):
        output_path = param.getroot().find('head').find('id').text
        h5_output_path = self.root_path + output_path + ".h5"
        xml_output_path = self.root_path + output_path + ".xml"
        GetData.ensure_dir(h5_output_path)
        try:
            profiles.to_hdf(path_or_buf=h5_output_path, key="profiles")
            param.write(xml_output_path)
            print('Beamlet profile data written to file: ' + output_path)
        except:
            print('Beamlet profile data could NOT be written to file: ' + output_path)
            raise

