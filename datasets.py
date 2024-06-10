ds_name_to_n_feats_mapping = {
    'markov_3feat11t5g_more_balanced':3,
    'markov_3feat11t5g_less_balanced':3,
    'markov_32feat_11t5g_more_balanced':32,
    'markov_32feat_11t5g_less_balanced':32,
    'real_NCT00364013':25,
}

from dataset_11t5g_markov import DatasetMarkovSurvSurf, DataModuleMarkovSurvSurf
from dataset_NCT00364013 import DataModuleNCT00364013SurvCurv