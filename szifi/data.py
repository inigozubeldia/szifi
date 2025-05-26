from szifi import params
import importlib.util

class input_data:

    def __init__(self,params_szifi=params.params_szifi_default,params_data=params.params_data_default):

        path_to_survey = params_szifi["survey_file"]
        spec_cat = importlib.util.spec_from_file_location("cat_module",path_to_survey)
        cat_module = importlib.util.module_from_spec(spec_cat)
        spec_cat.loader.exec_module(cat_module)
        input_data_survey_in = cat_module.input_data_survey
        input_data_instance = input_data_survey_in(params_szifi=params_szifi,params_data=params_data)

        self.data = input_data_instance.data
