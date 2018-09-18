

##############
# PARAMETERS #
##############

data_path = './saved_data/20180918_174028/activity/activity.h5'

#############
# LIBRARIES #
#############

import numpy as np
import pandas as pd


################
# DATA_LOADING #
################

df = pd.HDFStore(data_path)
df = pd.DataFrame(df['activity'])

