import pandas as pd
import time
import numpy as np
from pathlib import Path
import pickle
from voxelmon import PtxBlk360G1_Group,CanopyBulkDensityModel,get_files_list,plot_side_view,smooth
import statsmodels.formula.api as smf
import seaborn as sns

cell_size = .1
grid_radius = 12 # Distance from grid center to edge
plot_radius = 11.3
max_grid_height = 30 # Height of grid above coordinate [0,0,0]
max_occlusion = .8

input_folder = '../TontoNF/TLS/PTX/AllScans/'
keyword = 'Med Density 1'
export_folder = 'D:/DataWork/pypadResultsMulti/'
surface_biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\SubplotBiomassEstimates.csv')
biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\AZ_CanProf_Output.csv')
biomass_classes = list(biomass.columns[2:-1])
#biomassClasses = ['PIPO','PINYON','JUNIPER','OAKS','ARPU','OTHER']
sigma = 1.2
feature = 'pad'
process = True
by_veg_type = False
leave_one_out = True
generate_test_figure = False
generate_figures = False

def get_class(plot_name):
    code = int(plot_name[1:3])
    if code ==5:
        return 'CHAP'
    elif code <=12:
        return 'PJO'
    else:
        return 'PPO'

def get_plot_id(filename):
    import os
    return os.path.basename(filename).split('-')[0]

def score(y_obs,y_pred,print_output=True,refit=True):
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import numpy as np
    y_pred = pd.DataFrame(y_pred).to_numpy()
    y_obs = pd.DataFrame(y_obs).to_numpy()
    if refit:
        lm = LinearRegression()
        lm = lm.fit(y_pred,y_obs)
        y_pred = lm.predict(y_pred)

    y_obs = y_obs.flatten()
    y_pred = y_pred.flatten()

    r2 = metrics.r2_score(y_obs,y_pred)
    rmse = metrics.root_mean_squared_error(y_obs,y_pred)
    mae = metrics.mean_absolute_error(y_obs,y_pred)

    m = y_obs.mean()
    rrmse = rmse / m

    mape = np.abs((y_obs - y_pred) / y_obs).mean()
    wmape = np.abs((y_obs - y_pred)).sum() / y_obs.sum()
    wmpe = (y_obs - y_pred).sum() / y_obs.sum()

    if print_output:
        print('R^2: ', r2)
        print('RMSE: ',rmse)
        print('RRMSE: ',rrmse) #relative rmse
        print('MAE: ',mae)
        print('MeanObs: ',m) #mean of observed values
        print('MAPE: ', mape) #mean absolute percent error
        print('WMAPE: ', wmape) #mean absolute percent error weighted by y_obs
        print('WMPE: ', wmpe)  # mean percent error weighted by y_obs
        print()
    return {'r2':r2,'rmse':rmse,'rrmse':rrmse}




files_scan1 = get_files_list(input_folder, keyword)
files_all = get_files_list(input_folder, '.ptx')
files_grouped = []
for fileScan1 in files_scan1:
    plot_id = get_plot_id(fileScan1)
    files_grouped.append([file for file in files_all if plot_id in file])
start_time_all = time.time()
i=1
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
files_grouped = files_grouped[6:]

if process:


    for filegroup in files_grouped:
        start_time = time.time()
        print("Starting file ", i, " of ", len(files_grouped))
        base_file_name = get_plot_id(filegroup[0])

        profile,plot_summary = PtxBlk360G1_Group(filegroup).execute_default_processing(export_folder=export_folder, plot_name=base_file_name, cell_size=cell_size,
                                                                              plot_radius=plot_radius, max_height=max_grid_height, max_occlusion=max_occlusion,
                                                                              sigma1=.1, min_pad_foliage=.01, max_pad_foliage=6)


        plt.plot(profile['pad'], profile['height'])
        plt.show()

        print("Finished file ", i, " of ", len(files_grouped), " in ", round(time.time() - start_time, 3), " seconds")
        i += 1

    print("Finished all files in ",round(time.time()-start_time_all)," seconds")

from scipy.ndimage import gaussian_filter1d,uniform_filter1d
import math
import statsmodels.formula.api as smf

biomass_list = []
for plot in biomass['Plot_ID'].unique():
    biomass_filter = biomass[(biomass['Plot_ID']==plot) & (biomass['Height_m']>=1)].copy()
    for col in biomass_classes:
        biomass_filter[col] = smooth(biomass_filter[col],sigma=sigma)
    biomass_filter['TOTAL'] = smooth(biomass_filter['TOTAL'],sigma=sigma)
    biomass_list.append(biomass_filter)
biomass = pd.concat(biomass_list)
biomass['CLASS'] = [get_class(string) for string in biomass['Plot_ID']]
biomass['heightBin'] = (biomass['Height_m'] / cell_size).round().astype(int)

tls_summary_files = get_files_list(export_folder + '/PAD_Summary', '.csv')
i=0
for file in tls_summary_files:
    df_tls = pd.read_csv(file)
    df_tls.insert(0,'Plot_ID',os.path.basename(file).strip('.csv'))
    df_tls = df_tls[df_tls['height']>=.2]
    df_tls[feature] = smooth(df_tls[feature], sigma=sigma)
    if i == 0:
        df_all = df_tls
        i+=1
    else:
        df_all = pd.concat([df_all,df_tls])
df_all = df_all.merge(biomass,on=['Plot_ID','heightBin'],how='outer')
df_all['CLASS'] = [get_class(string) for string in df_all['Plot_ID']]
df_all = df_all.fillna(0)
df_all = df_all[~((df_all[feature]<.005) & (df_all['TOTAL']<.005))]

outliers = ['T0523061403']
df_all = df_all[~(df_all['Plot_ID'].isin(outliers))]

### Surface fuels ###
surface_biomass_plot = surface_biomass.pivot_table(index='PLOT_NAME', values=['LOAD_LITTER_DUFF', 'LOAD_DOWN_WOODY', 'LOAD_STANDING', 'LOAD_TOTAL_SURFACE'], aggfunc='mean')
surface_biomass_plot['LOAD_DOWNED'] = surface_biomass_plot['LOAD_LITTER_DUFF'] + surface_biomass_plot['LOAD_DOWN_WOODY']
plot_canopy = df_all[df_all['height'] >= .1].pivot_table(index='Plot_ID', aggfunc={'CLASS': 'first', 'foliage': 'sum', 'pad': 'sum'})
plot_surface = df_all[(df_all['height'] >= .1) & (df_all['height'] < 1)].pivot_table(index='Plot_ID', values=['foliage', 'pad'], aggfunc='mean')
surface_biomass_plot = surface_biomass_plot.join(plot_canopy)
surface_biomass_plot = surface_biomass_plot.join(plot_surface, lsuffix='_canopy', rsuffix='_surface')
surface_biomass_plot = surface_biomass_plot.dropna()

sns.scatterplot(surface_biomass_plot, x='pad_canopy', y='LOAD_DOWNED', hue='CLASS')
plt.show()

lm = smf.ols('LOAD_DOWNED ~ pad_canopy:CLASS', surface_biomass_plot).fit()
print(lm.summary())
print('RMSE = ',(lm.resid**2).mean()**.5)

sns.scatterplot(surface_biomass_plot, x='pad_surface', y='LOAD_STANDING', hue='CLASS')
plt.show()

lm = smf.ols('LOAD_STANDING ~ pad_surface:CLASS', surface_biomass_plot).fit()
print(lm.summary())
print('RMSE = ',(lm.resid**2).mean()**.5)

df_all = df_all[df_all['height']>=1]
outliers = ['T0523061403','T1423071101']
df_all = df_all[~(df_all['Plot_ID'].isin(outliers))]

results = []
if by_veg_type:
    for class_id in df_all['CLASS'].unique():
        df_class = df_all[df_all['CLASS'] == class_id]
        plots = pd.DataFrame(df_class['Plot_ID'].unique())
        if leave_one_out:
            nfolds = len(plots)
        else:
            nfolds = 10
        kvals = np.arange(nfolds).repeat(int(math.ceil(len(plots)/nfolds)))
        np.random.shuffle(kvals)
        plots['k'] = kvals[:len(plots)]
        for k in range(nfolds):
            train_plots = plots.loc[plots['k'] != k,0]
            test_plots = plots.loc[plots['k'] == k,0]
            train_data = df_class[(df_class['Plot_ID'].isin(train_plots))]
            test_data = df_class[(df_class['Plot_ID'].isin(test_plots))]
            model = CanopyBulkDensityModel()
            model.fit(train_data, train_data, biomassCols=biomass_classes, cellSize=cell_size, sigma=sigma, lidarValueCol=feature)
            pred = model.predict(test_data, lidarValueCol=feature)
            results.append(pred)
        with open(class_id + '.model', 'wb') as f:
            pickle.dump(model,f)
else:
    df_class = df_all
    plots = pd.DataFrame(df_class['Plot_ID'].unique())
    if leave_one_out:
        nfolds = len(plots)
    else:
        nfolds = 3
    kvals = np.arange(nfolds).repeat(int(math.ceil(len(plots) / nfolds)))
    np.random.shuffle(kvals)
    plots['k'] = kvals[:len(plots)]
    for k in range(nfolds):
        train_plots = plots.loc[plots['k'] != k, 0]
        test_plots = plots.loc[plots['k'] == k, 0]
        train_data = df_class[(df_class['Plot_ID'].isin(train_plots))]
        test_data = df_class[(df_class['Plot_ID'].isin(test_plots))]
        model = CanopyBulkDensityModel()
        model.fit(lidarProfile = train_data, biomassProfile=train_data, biomassCols=biomass_classes, cellSize=cell_size, sigma=sigma,
                  lidarValueCol=feature, fitIntercept=True, twoStageFit=True)
        pred = model.predict(test_data, lidarValueCol=feature)
        results.append(pred)
    with open('combined.model', 'wb') as f:
        pickle.dump(model, f)

results = pd.concat(results)
results['volTotal'] = math.pi * grid_radius ** 2 * cell_size
results = results.fillna(0)
results['biomass'] = results['TOTAL']
plots = results['Plot_ID'].unique()

for plot in plots:
    results_filter = results[results['Plot_ID']==plot]
    results_filter.to_csv(export_folder + '\\' + 'Final_Profile' + '\\' + plot + '.csv')


import matplotlib.pyplot as plt

results_plot = results.pivot_table(index='Plot_ID',aggfunc={'biomass':['mean','max'],
                                                            'biomassPred':['mean','max'],
                                                            'occluded':'mean', 'volTotal':'sum'}).reset_index()

results_plot['biomass','sum'] = results_plot['biomass','mean'] * results_plot['volTotal','sum'] / (math.pi * plot_radius ** 2 / 10000)
results_plot['biomassPred','sum'] = results_plot['biomassPred','mean'] * results_plot['volTotal','sum'] / (math.pi * plot_radius ** 2 / 10000)

results_plot['CLASS'] = [get_class(string) for string in results_plot['Plot_ID']]

print("Sum of biomass (canopy fuel load)")
score_cfl = score(results_plot['biomass','sum'],results_plot['biomassPred','sum'],refit=False)

print("Max of biomass (max canopy bulk density)")
score_cbd = score(results_plot['biomass','max'],results_plot['biomassPred','max'],refit=False)

print("Max of biomass (max canopy bulk density) no chaparral")
score(results_plot['biomass','max'][results_plot['CLASS']!='CHAP'],results_plot['biomassPred','max'][results_plot['CLASS']!='CHAP'],refit=False)

f, [ax1,ax2] = plt.subplots(ncols=2, constrained_layout=True)
sns.scatterplot(x=results_plot['biomass','sum'],y=results_plot['biomassPred','sum'],
                hue=results_plot['CLASS'],palette=['orange','green','blue'],ax=ax1)
sns.scatterplot(x=results_plot['biomass','max'],y=results_plot['biomassPred','max'],
                hue=results_plot['CLASS'],palette=['orange','green','blue'],ax=ax2)
ax1.axline([0,0],slope=1)
ax2.axline([0,0],slope=1)
ax1.set_ylim(ax1.get_xlim())
ax2.set_ylim(ax2.get_xlim())
ax1.set_title('r^2 = '+str(round(score_cfl['r2'],2))+' rmse = '+str(round(score_cfl['rmse'],2)))
ax2.set_title('r^2 = '+str(round(score_cbd['r2'],2))+' rmse = '+str(round(score_cbd['rmse'],2)))
plt.savefig(export_folder + '_'.join(['Results', feature, str(cell_size), str(max_occlusion), str(sigma)]) + '.pdf')
plt.show()

if generate_test_figure:
    import polars as pl
    plotname = 'T1423080202'
    summary_filepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
    basename = summary_filepath.name
    dempath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['DEM', basename]))
    pointspath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['Points', basename]))
    pts = pl.read_csv(pointspath)
    demPts = pl.read_csv(dempath)
    results_filter = results[(results['Plot_ID']==plotname) & (results['height']>0)]
    f,[ax1,ax2] = plt.subplots(ncols=2,sharey=True,width_ratios=[2.5,1],figsize=[8,4])
    [arr,arr_extents] = plot_side_view(pts,direction=0,demPtsNormalize=demPts,returnData=True)
    ax1.imshow(arr,extent=arr_extents,aspect=2)
    ax2.plot(results_filter['biomass'],results_filter['height'],label='Traditional Estimate')
    ax2.plot(results_filter['biomassPred'],results_filter['height'],label='Lidar Estimate')
    ymax = max(ax1.get_ylim()[1],ax2.get_ylim()[1],14)
    ax1.set_ylim([0,ymax])
    ax2.set_ylim([0,ymax])
    ax2.set_yticks(ax1.get_yticks())
    ax2.legend()
    plt.show()

if generate_figures:
    import polars as pl
    plots = results['Plot_ID'].unique()
    np.random.shuffle(plots)
    for plotname in plots:
        summary_filepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
        basename = summary_filepath.name
        dempath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['DEM', basename]))
        pointspath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['Points', basename]))
        pts = pl.read_csv(pointspath)
        demPts = pl.read_csv(dempath)
        results_filter = results[(results['Plot_ID']==plotname) & (results['height']>0)]
        f,[ax1,ax2] = plt.subplots(ncols=2,sharey=True,figsize=[8,4])
        [arr,arr_extents] = plot_side_view(pts,direction=0,demPtsNormalize=demPts,returnData=True)
        ax1.imshow(arr,extent=arr_extents,aspect=2)
        ax2.plot(results_filter['biomass'],results_filter['height'],label='Traditional Estimate')
        ax2.plot(results_filter['biomassPred'],results_filter['height'],label='Lidar Estimate')
        #ymax = max(ax1.get_ylim()[1],ax2.get_ylim()[1],14)
        ymax = max(results_filter['height'].max()+1,15)
        ax1.set_ylim([0,ymax])
        ax2.set_ylim([0,ymax])
        ax2.set_yticks(ax1.get_yticks())
        ax2.legend()
        ax1.set_title(plotname)
        ax1.set_ylabel('Height (m)')
        ax1.set_xlabel('Easting (m)')
        ax2.set_xlabel('Canopy Bulk Density (kg/m^3)')
        plt.savefig(export_folder + plotname + '.pdf')
        plt.show()

