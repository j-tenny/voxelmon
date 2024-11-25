#%%#####################################################################################################################
import pandas as pd
import time
import numpy as np
from pathlib import Path
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.formula.api as smf
from voxelmon import PtxBlk360G1, CanopyBulkDensityModel, get_files_list, plot_side_view

########################################################################################################################

### Setup Run Parameters ###
cell_size = .1 # Voxel side length
grid_radius = 12  # Distance from voxel grid center to edge
plot_radius = 11.3  # Radius of circular plot after clipping
max_grid_height = 30  # Expected max vegetation height above coordinate [0,0,0]
max_occlusion = .8 # Threshold to determine when to classify a voxel as occluded

input_folder = '../TontoNF/TLS/PTX/AllScans/' # Folder containing scans in .PTX format
keyword = 'Med Density 1' # Keyword used to filter selected scans from other files. Just want the center scan here.
export_folder = 'D:/DataWork/pypadResults/' # Root folder for results
biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\AZ_CanProf_Output.csv') # Conventional estimates of canopy bulk density by species group by height bin by plot. Produced by BuildCanopyProfile.R
biomass_classes = list(biomass.columns[2:-1]) # Names of the species groups used in biomass estimates
surface_biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\SubplotBiomassEstimates.csv') # Estimated surface fuels by fuel component by subplot by plot
sigma = 1.2 # Sigma value used in gaussian smoothing filter applied to canopy bulk density profiles
feature = 'pad' # Can be 'pad' to train model with plant area density or 'foliage' to train model with foliage volume
process = False # Process (or reprocess) ptx files to produce PAD estimates
by_veg_type = False # True: Estimate leaf mass per area seperately for each vegetation type. False: combine all veg types when estimating LMA
leave_one_out = False # True: Use leave-one-out cross validation. False: Use k-fold cross validation
nfolds = 10 # Number of folds used in k-fold cross validation. Set to 1 to train with all data (can't test).
bootstrap_confidence_intervals = True
generate_test_figure = False # Show figure from individual random plot.
generate_figures = False # Generate figures for all plots
variance_stats = False # Generate stats for effects of terrain, occlusion, height

#%%#####################################################################################################################
### Define Helper Functions ###
def get_class(plot_name):
    # Determine vegetation type from plot name
    code = int(plot_name[1:3])
    if code == 5:
        return 'CHAP'
    elif code <= 12:
        return 'PJO'
    else:
        return 'PPO'

def score(y_obs, y_pred, print_output=True, refit=True):
    # Generate model scoring metrics for model comparison
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import numpy as np
    y_pred = pd.DataFrame(y_pred).to_numpy()
    y_obs = pd.DataFrame(y_obs).to_numpy()
    if refit:
        lm = LinearRegression()
        lm = lm.fit(y_pred, y_obs)
        y_pred = lm.predict(y_pred)

    y_obs = y_obs.flatten()
    y_pred = y_pred.flatten()

    r2 = metrics.r2_score(y_obs, y_pred)
    rmse = metrics.root_mean_squared_error(y_obs, y_pred)
    mae = metrics.mean_absolute_error(y_obs, y_pred)

    m = y_obs.mean()
    rrmse = rmse / m

    mape = np.abs((y_obs - y_pred) / y_obs).mean()
    wmape = np.abs((y_obs - y_pred)).sum() / y_obs.sum()
    wmpe = (y_obs - y_pred).sum() / y_obs.sum()

    if print_output:
        print('R^2: ', r2)
        print('RMSE: ', rmse)
        print('RRMSE: ', rrmse)  # relative rmse
        print('MAE: ', mae)
        print('MeanObs: ', m)  # mean of observed values
        print('MAPE: ', mape)  # mean absolute percent error
        print('WMAPE: ', wmape)  # mean absolute percent error weighted by y_obs
        print('WMPE: ', wmpe)  # mean percent error weighted by y_obs
        print()
    return {'r2': r2, 'rmse': rmse, 'rrmse': rrmse}

### Setup file processing ###
files = get_files_list(input_folder, keyword)
start_time_all = time.time()
i = 1

warnings.filterwarnings("ignore", category=RuntimeWarning)
files = files[6:] # Skip Tonto Basin desert files not included in study

# Define base extents of voxel grid
base_extents = [-grid_radius, -grid_radius, -grid_radius, grid_radius, grid_radius, max_grid_height]


if process:
    ### Process (or reprocess) ptx files to calculate plant area density ###
    for ptx_file in files:
        start_time = time.time()
        print("Starting file ", i, " of ", len(files))

        # Get plot name from file name
        base_file_name = os.path.splitext(os.path.basename(ptx_file))[0].split('-')[0]

        # Read ptx file
        ptx = PtxBlk360G1(ptx_file, applyTranslation=False, applyRotation=True, dropNull=False)

        # Calculate plant area density
        profile, summary = ptx.execute_default_processing(export_folder=export_folder, plot_name=base_file_name, cell_size=cell_size,
                                                          plot_radius=plot_radius, plot_radius_buffer=grid_radius-plot_radius,
                                                          max_height=max_grid_height, max_occlusion=max_occlusion,
                                                          sigma1=0, min_pad_foliage=.01, max_pad_foliage=6)

        # Show vertical profile
        plt.plot(profile['pad'], profile['height'])
        plt.show()

        print("Finished file ", i, " of ", len(files), " in ", round(time.time() - start_time, 3), " seconds")
        i += 1

    print("Finished all files in ", round(time.time() - start_time_all), " seconds")

### Read and concatenate vertical profiles from all plots ###
tls_summary_files = get_files_list(export_folder + '/PAD_Summary', '.csv')
i = 0
for file in tls_summary_files:
    df_tls = pd.read_csv(file)
    df_tls.insert(0, 'Plot_ID', os.path.basename(file).strip('.csv'))
    if i == 0:
        df_all = df_tls
        i += 1
    else:
        df_all = pd.concat([df_all, df_tls])

### Format dataframe containing conventional biomass estimates ###
# Add column for height bin as integer
biomass['heightBin'] = (biomass['Height_m'] / cell_size).round().astype(int)
biomass = biomass.drop(columns='Height_m')
### Merge TLS plant area density profiles with conventional bulk density profiles
df_all = df_all.merge(biomass, on=['Plot_ID', 'heightBin'], how='outer')
# Add column for vegetation type
df_all['CLASS'] = [get_class(string) for string in df_all['Plot_ID']]

### Fill empty cells with 0, remove height bins where TLS and conventional biomass are both ~0
df_all = df_all.fillna(0)
df_all = df_all[~((df_all[feature] < .005) & (df_all['TOTAL'] < .005))]

### Remove plots that are unexplained outliers
#outliers = ['T0523061403']
outliers = []
df_all = df_all[~(df_all['Plot_ID'].isin(outliers))]

#%%#####################################################################################################################
### Train and Test Surface Fuel Models ###

# Summarize conventional surface fuel load estimates by plot
surface_biomass_plot = surface_biomass.pivot_table(index='PLOT_NAME',
                                                   values=['LOAD_LITTER', 'LOAD_DOWN_WOODY', 'LOAD_STANDING',
                                                        'LOAD_TOTAL_SURFACE'], aggfunc='mean')

# Combine litter and downed woody debris
surface_biomass_plot['LOAD_DOWNED'] = surface_biomass_plot['LOAD_LITTER'] + surface_biomass_plot['LOAD_DOWN_WOODY']

# Summarize TLS data by plot (height .1m to top of canopy)
plot_canopy = df_all[df_all['height'] >= .1].pivot_table(index='Plot_ID', aggfunc={'CLASS': 'first', 'foliage': 'sum', 'pad': 'sum'})
# Calculate plant area index (sum of plant area density profile * height bin size)
plot_canopy['pai'] = plot_canopy['pad'] * cell_size

# Summarize TLS data by plot (height .1m to 1m
plot_surface = df_all[(df_all['height'] >= .1) & (df_all['height'] < 1)].pivot_table(index='Plot_ID',
                                                                                     values=['foliage', 'pad'],
                                                                                     aggfunc='mean')
# Merge conventional surface fuel estimates, TLS canopy metrics, TLS surface metrics
surface_biomass_plot = surface_biomass_plot.join(plot_canopy)
surface_biomass_plot = surface_biomass_plot.join(plot_surface, lsuffix='_canopy', rsuffix='_surface')
surface_biomass_plot = surface_biomass_plot.dropna()

# Model downed woody debris as a function of canopy plant area index by vegetation type
lm_downed = smf.ols('LOAD_DOWNED ~ pai:CLASS', surface_biomass_plot).fit()
print(lm_downed.summary())
print('RMSE = ', (lm_downed.resid ** 2).mean() ** .5)

# Model standing (live) surface fuel load as a function of near-surface plant area density
lm_standing = smf.ols('LOAD_STANDING ~ pad_surface', surface_biomass_plot).fit()
print(lm_standing.summary())
print('RMSE = ', (lm_standing.resid ** 2).mean() ** .5)

# Produce surface fuel regression figures
f, [ax1, ax2] = plt.subplots(ncols=2, constrained_layout=True, figsize=[7, 4])
sns.scatterplot(surface_biomass_plot, x='pai', y='LOAD_DOWNED', hue='CLASS', palette=['orange', 'green', 'blue'],
                ax=ax1)
sns.scatterplot(surface_biomass_plot, x='pad_surface', y='LOAD_STANDING', hue='CLASS',
                palette=['orange', 'green', 'blue'], ax=ax2)
ax1.axline((0, lm_downed.params.iloc[0]), slope=lm_downed.params.iloc[1], color='orange', linestyle='--')
ax1.axline((0, lm_downed.params.iloc[0]), slope=lm_downed.params.iloc[2], color='green', linestyle='--')
ax1.axline((0, lm_downed.params.iloc[0]), slope=lm_downed.params.iloc[3], color='blue', linestyle='--')
ax2.axline((0, lm_standing.params.iloc[0]), slope=lm_standing.params.iloc[1], color='black', linestyle='--')
f.tight_layout(pad=.5, w_pad=2.5)
ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax1.set_xlabel('LAI ($m^2$/$m^2$)')
ax1.set_ylabel('Downed Surface Fuel Load (kg/$m^2$)')
ax1.text(.05,.93,'$R^2$ = ' + str(lm_downed.rsquared.round(2)),transform=ax1.transAxes)
ax1.text(.05,.88,'RMSE = ' + str(((lm_downed.resid ** 2).mean() ** .5).round(2))+ ' kg/$m^2$',transform=ax1.transAxes)
ax2.set_xlabel('Surface LAD ($m^2$/$m^3$)')
ax2.set_ylabel('Standing Surface Fuel Load (kg/$m^2$)')
ax2.text(.05,.93,'$R^2$ = ' + str(lm_standing.rsquared.round(2)),transform=ax2.transAxes)
ax2.text(.05,.88,'RMSE = ' + str(((lm_standing.resid ** 2).mean() ** .5).round(2)) + ' kg/$m^2$',transform=ax2.transAxes)
plt.savefig(export_folder + '\\SurfaceFuelPAD.pdf')
plt.show()

#%%#####################################################################################################################
### MODEL CANOPY FUELS ###
# Remove surface fuel and outliers
df_all = df_all[df_all['height'] >= 1]
outliers = ['T1423071101']
df_all = df_all[~(df_all['Plot_ID'].isin(outliers))]

# Train model
results = []
if by_veg_type:
    # Calculate effective leaf mass per area separately for each vegetation type
    for classId in df_all['CLASS'].unique():
        # Filter to this vegetation type
        df_class = df_all[df_all['CLASS'] == classId]
        # Get list of plot names
        plots = pd.DataFrame(df_class['Plot_ID'].unique())
        if leave_one_out:
            # If using loo cross validation, overwrite nfolds with nplots
            nfolds = len(plots)
        # Randomly assign k values for each plot
        kvals = np.arange(nfolds).repeat(int(math.ceil(len(plots) / nfolds)))
        np.random.shuffle(kvals)
        plots['k'] = kvals[:len(plots)]
        # Train/test with each fold
        for k in range(nfolds):
            # Train test split
            train_plots = plots.loc[plots['k'] != k, 0]
            test_plots = plots.loc[plots['k'] == k, 0]
            train_data = df_class[(df_class['Plot_ID'].isin(train_plots))]
            test_data = df_class[(df_class['Plot_ID'].isin(test_plots))]
            # Train model on training set
            model = CanopyBulkDensityModel()
            model.fit(train_data, biomassCols=biomass_classes, sigma=sigma, plotIdCol='Plot_ID',
                      lidarValueCol=feature, minHeight=1, classIdCol='CLASS', fitIntercept=True, twoStageFit=True)
            # Get predictions for test set
            pred = model.predict_and_test(test_data, biomassCols=biomass_classes, lidarValueCol=feature, classIdCol='CLASS', resultCol='biomassPred')
            results.append(pred)
        # Train and save final model based on all training data
        model = CanopyBulkDensityModel()
        model.fit(df_class, biomassCols=biomass_classes, sigma=sigma, plotIdCol='Plot_ID',
                  lidarValueCol=feature, minHeight=1, classIdCol='CLASS', fitIntercept=True, twoStageFit=True)
        model.to_file(classId+'.model')
        print(f'Effective Leaf Area Density {classId} Model:')
        print(model.mass_ratio)
        print()
else:
    # Calculate effective leaf mass per area combining data from all vegetation types
    df_class = df_all
    # Get list of plot names
    plots = pd.DataFrame(df_class['Plot_ID'].unique())
    if leave_one_out:
        # If using loo cross validation, overwrite nfolds with nplots
        nfolds = len(plots)
    # Randomly assign k values for each plot
    kvals = np.arange(nfolds).repeat(int(math.ceil(len(plots) / nfolds)))
    np.random.shuffle(kvals)
    plots['k'] = kvals[:len(plots)]
    # Train/test with each fold
    for k in range(nfolds):
        # Train test split
        train_plots = plots.loc[plots['k'] != k, 0]
        test_plots = plots.loc[plots['k'] == k, 0]
        train_data = df_class[(df_class['Plot_ID'].isin(train_plots))]
        test_data = df_class[(df_class['Plot_ID'].isin(test_plots))]
        # Train model on training set
        model = CanopyBulkDensityModel()
        model.fit(train_data, biomassCols=biomass_classes, sigma=sigma, plotIdCol='Plot_ID',
                  lidarValueCol=feature, minHeight=1, classIdCol='CLASS', fitIntercept=True, twoStageFit=True)
        # Get predictions for test set
        pred = model.predict_and_test(test_data, biomassCols=biomass_classes, lidarValueCol=feature, classIdCol='CLASS', resultCol='biomassPred')
        results.append(pred)
    # Train and save final model based on all training data
    model = CanopyBulkDensityModel()
    model.fit(df_class, biomassCols=biomass_classes, sigma=sigma, plotIdCol='Plot_ID',
              lidarValueCol=feature, minHeight=1, classIdCol='CLASS', fitIntercept=True, twoStageFit=True)
    model.to_file('combined.model')
    print('Effective Leaf Area Density Combined Model:')
    print(model.mass_ratio)
    print()

# Format dataframe with predictions
results = pd.concat(results)
results['volTotal'] = math.pi * grid_radius ** 2 * cell_size
results = results.fillna(0)
results['biomass'] = results['CBD_TOTAL'] # Conventional estimate of canopy bulk density at a height bin, all species combined
plots = results['Plot_ID'].unique()

# Write results to csv files (separately for each plot)
for plot in plots:
    results_filter = results[results['Plot_ID'] == plot]
    results_filter.to_csv(export_folder + '\\' + 'Final_Profile' + '\\' + plot + '.csv')

# Aggregate canopy bulk density along profile
results_plot = results.pivot_table(index='Plot_ID', aggfunc={'biomass': ['mean', 'max'],
                                                             'biomassPred': ['mean', 'max'],
                                                             'occluded': 'mean', 'volTotal': 'sum'}).reset_index()

# Calculate total canopy fuel load
results_plot['biomass', 'sum'] = results_plot['biomass', 'mean'] * results_plot['volTotal', 'sum'] / (
        math.pi * plot_radius ** 2)
results_plot['biomassPred', 'sum'] = results_plot['biomassPred', 'mean'] * results_plot['volTotal', 'sum'] / (
        math.pi * plot_radius ** 2)

results_plot['CLASS'] = [get_class(string) for string in results_plot['Plot_ID']]

# Print accuracy results

print("Canopy fuel load (all veg types)")
score_cfl = score(results_plot['biomass', 'sum'], results_plot['biomassPred', 'sum'], refit=True)

print("Max CBD (all veg types)")
score_cbd = score(results_plot['biomass', 'max'], results_plot['biomassPred', 'max'], refit=True)


print("Canopy fuel load (CHAP only)")
results_plot_filter = results_plot[results_plot['CLASS']=='CHAP']
score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'], refit=True)
print("Max CBD (CHAP only)")
score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'], refit=True)

print("Canopy fuel load (PJO only)")
results_plot_filter = results_plot[results_plot['CLASS']=='PJO']
score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'], refit=True)
print("Max CBD (PJO only)")
score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'], refit=True)

print("Canopy fuel load (PPO only)")
results_plot_filter = results_plot[results_plot['CLASS']=='PPO']
score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'], refit=True)
print("Max CBD (PPO only)")
score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'], refit=True)

print("Canopy fuel load (CHAP + PJO only)")
results_plot_filter = results_plot[results_plot['CLASS']!='PPO']
score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'], refit=True)
print("Max CBD (CHAP + PJO only)")
score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'], refit=True)

print("Canopy fuel load (CHAP + PPO only)")
results_plot_filter = results_plot[results_plot['CLASS']!='PJO']
score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'], refit=True)
print("Max CBD (CHAP + PPO only)")
score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'], refit=True)

print("Canopy fuel load (PJO + PPO only)")
results_plot_filter = results_plot[results_plot['CLASS']!='CHAP']
score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'], refit=True)
print("Max CBD (PJO + PPO only)")
score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'], refit=True)


# Produce regression accuracy plots for total canopy fuel load and max canopy bulk density
f, [ax1, ax2] = plt.subplots(ncols=2, constrained_layout=True, figsize=[7, 4])
sns.scatterplot(x=results_plot['biomass', 'sum'], y=results_plot['biomassPred', 'sum'],
                hue=results_plot['CLASS'], palette=['orange', 'green', 'blue'], ax=ax1)
sns.scatterplot(x=results_plot['biomass', 'max'], y=results_plot['biomassPred', 'max'],
                hue=results_plot['CLASS'], palette=['orange', 'green', 'blue'], ax=ax2)
ax1.axline([0, 0], slope=1)
ax2.axline([0, 0], slope=1)
ax1.set_ylim(ax1.get_xlim())
ax2.set_ylim(ax2.get_xlim())
f.tight_layout(pad=.5, w_pad=2.5)
ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax1.set_xlabel('Conventional Estimate Canopy Fuel Load (kg/$m^2$)')
ax1.set_ylabel('Lidar Estimate Canopy Fuel Load (kg/$m^2$)')
ax1.text(.05,.93,'$R^2$ = ' + str(round(score_cfl['r2'], 2)),transform=ax1.transAxes)
ax1.text(.05,.88,'RMSE = ' + str(round(score_cfl['rmse'], 2)) + ' kg/$m^2$',transform=ax1.transAxes)
ax2.set_xlabel('Conventional Estimate $CBD_{max}$ (kg/$m^3$)')
ax2.set_ylabel('Lidar Estimate $CBD_{max}$ (kg/$m^3$)')
ax2.text(.05,.93,'$R^2$ = ' + str(round(score_cbd['r2'], 2)),transform=ax2.transAxes)
ax2.text(.05,.88,'RMSE = ' + str(round(score_cbd['rmse'], 2)) + ' kg/$m^3$',transform=ax2.transAxes)

plt.savefig(export_folder + '_'.join(['Results', feature, str(cell_size), str(max_occlusion), str(sigma)]) + '.pdf')
plt.show()

if bootstrap_confidence_intervals:
    nsamples = 5000
    # Get list of plot names
    plots = pd.DataFrame(df_all['Plot_ID'].unique()).to_numpy().flatten()
    coef_all = []
    for sample_ind in range(nsamples):
        resampled_df = []
        plot_indices = np.random.randint(0,plots.shape[0],plots.shape[0])
        for plot_i in plot_indices:
            resampled_df.append(df_class[df_all['Plot_ID']==plots[plot_i]])
        resampled_df = pd.concat(resampled_df).reset_index(drop=True)

        # Train model
        if by_veg_type:
            # Calculate effective leaf mass per area separately for each vegetation type
            for classId in resampled_df['CLASS'].unique():
                # Filter to this vegetation type
                df_class = resampled_df[resampled_df['CLASS'] == classId]
                # Train and save final model based on all training data
                model = CanopyBulkDensityModel()
                model.fit(df_class, biomassCols=biomass_classes, sigma=sigma, plotIdCol='Plot_ID',
                          lidarValueCol=feature, minHeight=1, classIdCol='CLASS', fitIntercept=True, twoStageFit=True)
                coef = model.mass_ratio
                coef['CLASS'] = classId
                coef_all.append(coef)

        else:
            # Calculate effective leaf mass per area combining data from all vegetation types
            # Train and save final model based on all training data
            model = CanopyBulkDensityModel()
            model.fit(resampled_df, biomassCols=biomass_classes, sigma=sigma, plotIdCol='Plot_ID',
                      lidarValueCol=feature, minHeight=1, classIdCol='CLASS', fitIntercept=True, twoStageFit=True)
            coef_all.append(model.mass_ratio)
    results = pd.DataFrame(coef_all)


    def quantile_05(series):
        return series.quantile(0.05)
    def quantile_95(series):
        return series.quantile(0.95)

    if by_veg_type:
        results_summary = results.pivot_table(aggfunc=['median','std', quantile_05, quantile_95], index='CLASS')
        print(results_summary)
    else:
        results_summary = results.agg(['median', 'std', quantile_05, quantile_95])
        print(results_summary)

#%%#####################################################################################################################
if generate_test_figure:
    import polars as pl

    plotname = 'T1423071101'
    summary_filepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
    basename = summary_filepath.name
    dempath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['DEM', basename]))
    pointspath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['Points', basename]))
    pts = pl.read_csv(pointspath)
    dem_pts = pl.read_csv(dempath)
    results_filter = results[(results['Plot_ID'] == plotname) & (results['height'] > 0)]
    f, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, width_ratios=[2.5, 1], figsize=[8, 4])
    [arr, arr_extents] = plot_side_view(pts, direction=0, demPtsNormalize=dem_pts, returnData=True)
    ax1.imshow(arr, extent=arr_extents, aspect=2)
    ax2.plot(results_filter['biomass'], results_filter['height'], label='Conventional Estimate')
    ax2.plot(results_filter['biomassPred'], results_filter['height'], label='Lidar Estimate')
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1], 14)
    ax1.set_ylim([0, ymax])
    ax2.set_ylim([0, ymax])
    ax2.set_yticks(ax1.get_yticks())
    ax2.legend()
    plt.show()

# Generate figures for each plot
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
        dem_pts = pl.read_csv(dempath)
        results_filter = results[(results['Plot_ID'] == plotname) & (results['height'] > 0)]
        f, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=[8, 4])
        [arr, arr_extents] = plot_side_view(pts, direction=0, demPtsNormalize=dem_pts, returnData=True)
        ax1.imshow(arr, extent=arr_extents, aspect=2)
        ax2.plot(results_filter['biomass'], results_filter['height'], label='Conventional Estimate')
        ax2.plot(results_filter['biomassPred'], results_filter['height'], label='Lidar Estimate')
        # ymax = max(ax1.get_ylim()[1],ax2.get_ylim()[1],14)
        ymax = max(results_filter['height'].max() + 1, 15)
        ax1.set_ylim([0, ymax])
        ax2.set_ylim([0, ymax])
        ax2.set_yticks(ax1.get_yticks())
        ax2.xaxis.set_major_locator(plt.MultipleLocator(.01))
        ax2.legend()
        ax1.set_title(plotname)
        ax1.set_ylabel('Height (m)')
        ax1.set_xlabel('Easting (m)')
        ax2.set_xlabel('Canopy Bulk Density (kg/m^3)')
        plt.savefig(export_folder + plotname + '.pdf')
        plt.show()

#%%#####################################################################################################################
# Look for patterns in the residuals
if variance_stats:
    from sklearn.preprocessing import robust_scale

    residuals = pd.concat(
        [results['biomassPred'] - results['biomass'], abs(results['biomassPred'] - results['biomass']),
         results['biomass'], results['occluded'], results['height'], results['CLASS']], axis=1)
    residuals.columns = ['resid', 'resid_abs', 'biomass', 'occlusion', 'height', 'veg_class']
    residuals.describe()

    residuals_plot = pd.concat([results_plot['biomassPred', 'sum'] - results_plot['biomass', 'sum'],
                                results_plot['biomassPred', 'max'] - results_plot['biomass', 'max'],
                                abs(results_plot['biomassPred', 'sum'] - results_plot['biomass', 'sum']),
                                abs(results_plot['biomassPred', 'max'] - results_plot['biomass', 'max']),
                                results_plot['biomass', 'sum'], results_plot['biomass', 'max'],
                                results_plot['occluded'], results_plot['CLASS']], axis=1)
    residuals_plot.columns = ['resid_sum', 'resid_max', 'resid_abs_sum', 'resid_abs_max', 'biomass_sum', 'biomass_max',
                              'occlusion', 'class']
    print(residuals_plot.describe())

    residuals.iloc[:, :-1] = robust_scale(residuals.iloc[:, :-1])
    residuals_plot.iloc[:, :-1] = robust_scale(residuals_plot.iloc[:, :-1])

    residuals['height_class'] = ((residuals['height'] // .25) * .25).astype(int).astype(str)
    sns.kdeplot(residuals, x='occlusion', y='resid')
    sns.scatterplot(residuals, x='occlusion', y='resid', hue='veg_class', alpha=.1, size=.5)
    plt.show()

    print(smf.ols('resid_abs~occlusion+biomass', residuals).fit().profile())
    print(smf.ols('resid_abs~occlusion*C(veg_class,Sum)+biomass*C(veg_class,Sum)', residuals).fit().profile())
    print(smf.ols('resid~occlusion+biomass', residuals).fit().profile())
    print(smf.ols('resid~occlusion*C(veg_class,Sum)+biomass*C(veg_class,Sum)', residuals).fit().profile())

    sns.scatterplot(residuals_plot, x='biomass_sum', y='resid_sum', hue='class')
    plt.show()
    sns.scatterplot(residuals_plot, x='occlusion', y='resid_sum', hue='class')
    plt.show()
    print(smf.ols('resid_abs~occlusion+biomass_sum', residuals_plot).fit().profile())
    print(smf.ols('resid_sum~occlusion+biomass_sum', residuals_plot).fit().profile())

    sns.scatterplot(residuals_plot, x='occlusion', y='resid_max', hue='class')
    plt.show()
    print(smf.ols('resid_max~occlusion+biomass_max', residuals_plot[results_plot['CLASS'] == 'PPO']).fit().profile())
