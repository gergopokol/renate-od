from utility.getdata import GetData
import utility.convert as uc


class NeutralDB(object):
    def __init__(self, param=None, resolved='bundled_n'):
        self.param = param
        self.resolved = resolved
        self.__load_neutral_cross_section()
        self.__get_atomic_levels()

    def __get_atomic_levels(self):
        self.atomic_levels = self.neutral_cross_sections.shape[0]

    def __load_neutral_cross_section(self):
        path = self.__set_neutral_cross_section_path()
        self.neutral_cross_sections = GetData(data_path_name=path, data_format="array").data

    def __set_neutral_cross_section_path(self, target='H'):
        cross_section_path = 'atomic_data/' + self.param.getroot().find('body').find('beamlet_species').text + \
                             '/cross_sections/neutral/'
        file_name = target + '_' + self.resolved + '_' + \
            self.param.getroot().find('body').find('beamlet_energy').text + '.txt'
        return cross_section_path + file_name

    def get_neutral_impact_loss(self, from_level):
        return uc.convert_from_cm2_to_m2(self.neutral_cross_sections[from_level, from_level])

    def get_neutral_impact_transition(self, from_level, to_level):
        return uc.convert_from_cm2_to_m2(self.neutral_cross_sections[from_level, to_level])
