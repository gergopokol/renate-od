from utility.getdata import GetData
from lxml import etree



class WriteData:
    def __init__(self, root_path="data/output/"):
        self.root_path = root_path

    def write_beamlet_profiles(self, param, profiles, subdir = 'beamlet/'):
        output_path = param.getroot().find('head').find('id').text
        h5_output_path = self.root_path + subdir + output_path + ".h5"
        xml_output_path = self.root_path + subdir +output_path + ".xml"
        GetData.ensure_dir(h5_output_path)
        try:
            profiles.to_hdf(path_or_buf=h5_output_path, key="profiles")
            if not isinstance(param.getroot().find('body').find('beamlet_profiles'), etree._Element):
                new_element = etree.Element('beamlet_profiles')
                new_element.text = h5_output_path
                new_element.set('unit', '-')
                param.getroot().find('body').append(new_element)
            param.write(xml_output_path)
            print('Beamlet profile data written to file: ' + output_path)
        except:
            print('Beamlet profile data could NOT be written to file: ' + output_path)
            raise

    def write_photon_emission_profile(self, obs_param, emission_profiles, subdir='emission/'):
        output_path = obs_param.getroot().find('head').find('id').text
        h5_output_path = self.root_path + subdir + output_path + ".h5"
        xml_output_path = self.root_path + subdir + output_path + ".xml"
        GetData.ensure_dir(h5_output_path)
        try:
            emission_profiles.to_hdf(path_or_buf=h5_output_path, key='emission_profiles')
            if not isinstance(obs_param.getroot().find('body').find('emission_profiles'), etree._Element):
                new_element = etree.Element('emission_profiles')
                new_element.text = h5_output_path
                new_element.set('unit', '-')
                obs_param.getroot().find('body').append(new_element)
            obs_param.write(xml_output_path)
            print('Photon emission profile data written to file: ' + output_path)
        except:
            print('Photon emission profile data could NOT be written to file: ' + output_path)
            raise