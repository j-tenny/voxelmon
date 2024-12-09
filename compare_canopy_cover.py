import pandas as pd
from voxelmon.utils import get_files_list
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

single_files = get_files_list(r'D:\DataWork\pypadResults\Plot_Summary','.csv')
single = []

for file in single_files:
    df = pd.read_csv(file)
    df['plot'] = file[-15:-4]
    single.append(df)
single = pd.concat(single)

multi_files = get_files_list(r'D:\DataWork\pypadResultsMulti\Plot_Summary','.csv')
multi = []

for file in multi_files:
    df = pd.read_csv(file)
    df['plot'] = file[-15:-4]
    multi.append(df)
multi = pd.concat(multi)

df = pd.merge(single,multi,on='plot',suffixes=('_single','_multi'))

cc_model = smf.ols('canopy_cover_multi~canopy_cover_single',df).fit_bayesian()
print(cc_model.summary())

plt.scatter(df['canopy_cover_single'],df['canopy_cover_multi'])
plt.axline([0,cc_model.params[0]],slope=cc_model.params[1])
plt.show()

slope_model = smf.ols('terrain_slope_multi~terrain_slope_single',df).fit_bayesian()
print(slope_model.summary())

plt.scatter(df['terrain_slope_single'],df['terrain_slope_multi'])
plt.show()