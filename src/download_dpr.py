'''
This script downloads level 2 GPM DPR swath data from both radar frequencies (Ku-Band and Ka-Band).

The data is downloaded from NASA GES DISC : https://disc.gsfc.nasa.gov/datasets/GPM_2AKu_07/summary?keywords=2A%20GPM%20DPR%20dual%20frequency

A NASA GES DISC user profile is necessary. 

Email: kukulies@ucar.edu 

'''
import sys 
import numpy as np 
from datetime import datetime 
# the pansat library is used to download the data: https://github.com/SEE-GEO/pansat
import pansat         
from pansat.time import TimeRange
from pansat.products.satellite.gpm import l2b_gpm_cmb, l2a_gpm_dpr
from pansat.download.providers.ges_disc import GesDiscProviderDay 

# add identity for GES DISC 
from pansat.download.accounts import add_identity, get_identity, delete_identity
import warnings
warnings.filterwarnings('ignore')

destination = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]

time_range = TimeRange(start, end)
files = l2a_gpm_dpr.download(time_range, destination = destination)



