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
from multiprocessing import freeze_support, set_start_method
from voxelmon import PtxBlk360G1, PtxBlk360G1_Group, BulkDensityProfileModelFitter, get_files_list, plot_side_view, calculate_species_proportions,smooth

########################################################################################################################

def main():
    ### Setup Run Parameters ###
    cell_size = .1 # Voxel side length
    grid_radius = 12  # Distance from voxel grid center to edge
    plot_radius = 11.3  # Radius of circular plot after clipping
    max_grid_height = 30  # Expected max vegetation height above coordinate [0,0,0]
    max_occlusion = .8 # Threshold to determine when to classify a voxel as occluded

    input_folder = '../TontoNF/TLS/PTX/AllScans/' # Folder containing scans in .PTX format
    keyword = 'Med Density 1' # Keyword used to filter selected scans from other files. Just want the center scan here.
    export_folder = 'D:/DataWork/TontoFinalResultsMulti/' # Root folder for results
    biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\AZ_CanProf_Output.csv') # Conventional estimates of canopy bulk density by species group by height bin by plot. Produced by BuildCanopyProfile.R
    biomass_classes = list(biomass.columns[2:-1]) # Names of the species groups used in biomass estimates
    surface_biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\SubplotBiomassEstimates.csv') # Estimated surface fuels by fuel component by subplot by plot
    profile_smoothing_factor = .005 # Value used as input to voxelmon.utils.smooth() when smoothing profiles
    feature = 'pad' # Can be 'pad' to train model with plant area density or 'foliage' to train model with foliage volume
    prior_mean = np.array([0.141,0.28,0.141,0.49,.333])
    prior_std = prior_mean*.05
    sigma_residuals = 1
    sigma_intercept = .005
    fit_intercept = False
    two_stage_fit = True
    process = False # Process (or reprocess) ptx files to produce PAD estimates
    by_veg_type = False # True: Estimate leaf mass per area seperately for each vegetation type. False: combine all veg types when estimating LMA
    leave_one_out = False # True: Use leave-one-out cross validation. False: Use k-fold cross validation
    nfolds = 10 # Number of folds used in k-fold cross validation. Set to 1 to train with all data (can't test).
    bootstrap_confidence_intervals = False
    generate_test_figure = False # Show figure from individual random plot.
    generate_figures = True # Generate figures for all plots

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

    def get_plot_id(filepath):
        filepath = Path(filepath)
        id = filepath.name.split('-')[0]
        return id

    def score(y_obs, y_pred, print_output=True):
        # Generate model scoring metrics for model comparison
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics
        import numpy as np
        import statsmodels.api as sm
        y_pred = pd.DataFrame(y_pred).to_numpy()
        y_obs = pd.DataFrame(y_obs).to_numpy()
        mean_error = (y_pred - y_obs).mean()
        mse = ((y_pred - y_obs)**2).mean()
        bias_sq = mean_error**2
        variance = mse - bias_sq
        rmse = mse**.5
        m = y_obs.mean()
        rrmse = rmse / m

        y_obs = sm.add_constant(y_obs)
        lm = sm.OLS(y_pred, y_obs).fit()
        pvalue = lm.pvalues[1]
        r2 = lm.rsquared


        if print_output:
            print('p-value: ', pvalue)
            print('R^2: ', r2)
            print('RMSE: ', rmse)
            print('MeanObs: ', m)  # mean of observed values
            print('RRMSE: ', rrmse)  # relative rmse
            print('MSE', mse)
            print('Bias: ',mean_error)
            print('BiasSq: ',bias_sq)
            print('Variance: ', variance)
            print()
        return {'r2': r2, 'rmse': rmse, 'rrmse': rrmse}

    ### Setup file processing ###
    files_scan1 = get_files_list(input_folder, keyword)
    files_all = get_files_list(input_folder, '.ptx')
    files_grouped = []
    for fileScan1 in files_scan1:
        plot_id = get_plot_id(fileScan1)
        files_grouped.append([file for file in files_all if plot_id in file])
    start_time_all = time.time()
    i=1

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

    ### Read and concatenate vertical profiles from all plots ###
    tls_summary_files = get_files_list(export_folder + '/PAD_Summary', '.csv')
    i = 0
    for file in tls_summary_files:
        df_tls = pd.read_csv(file)
        df_tls.insert(0, 'plot_id', os.path.basename(file).strip('.csv'))
        if i == 0:
            df_all = df_tls
            i += 1
        else:
            df_all = pd.concat([df_all, df_tls])

    ### Format dataframe containing conventional biomass estimates ###
    # Add column for height bin as integer
    biomass['height_bin'] = (biomass['height_m'] / cell_size).round().astype(int)
    df_all['height_bin'] = (df_all['height'] / cell_size).round().astype(int)
    biomass = biomass.drop(columns='height_m')
    ### Merge TLS plant area density profiles with conventional bulk density profiles
    df_all = df_all.merge(biomass, on=['plot_id', 'height_bin'], how='outer')
    # Add column for vegetation type
    df_all['veg_type'] = [get_class(string) for string in df_all['plot_id']]

    ### Fill empty cells with 0, remove height bins where TLS and conventional biomass are both ~0
    df_all = df_all.fillna(0)
    df_all = df_all[~((df_all[feature] < .005) & (df_all['TOTAL'] < .005))]

    ### Remove plots that are unexplained outliers
    # outliers = ['T0523061403']
    outliers = []
    df_all = df_all[~(df_all['plot_id'].isin(outliers))]

    # %%#####################################################################################################################
    ### Train and Test Surface Fuel Models ###

    # Summarize conventional surface fuel load estimates by plot
    surface_biomass_plot = surface_biomass.pivot_table(index='PLOT_NAME',
                                                       values=['LOAD_LITTER', 'LOAD_DOWN_WOODY', 'LOAD_STANDING',
                                                               'LOAD_TOTAL_SURFACE'], aggfunc='mean')

    # Combine litter and downed woody debris
    surface_biomass_plot['LOAD_DOWNED'] = surface_biomass_plot['LOAD_LITTER'] + surface_biomass_plot['LOAD_DOWN_WOODY']

    # Summarize TLS data by plot (height .1m to top of canopy)
    plot_canopy = df_all[df_all['height'] >= .1].pivot_table(index='plot_id', aggfunc={'veg_type': 'first', 'foliage': 'sum', 'pad': 'sum'})
    # Calculate plant area index (sum of plant area density profile * height bin size)
    plot_canopy['pai'] = plot_canopy['pad'] * cell_size

    # Summarize TLS data by plot (height .1m to 1m
    plot_surface = df_all[(df_all['height'] >= .1) & (df_all['height'] < 1)].pivot_table(index='plot_id',
                                                                                         values=['foliage', 'pad','occluded'],
                                                                                         aggfunc='mean')
    # Merge conventional surface fuel estimates, TLS canopy metrics, TLS surface metrics
    surface_biomass_plot = surface_biomass_plot.join(plot_canopy)
    surface_biomass_plot = surface_biomass_plot.join(plot_surface, lsuffix='_canopy', rsuffix='_surface')
    surface_biomass_plot = surface_biomass_plot.dropna()

    # Model downed woody debris as a function of canopy plant area index by vegetation type
    lm_downed = smf.ols('LOAD_DOWNED ~ pai:veg_type', surface_biomass_plot).fit()
    surface_biomass_plot['LOAD_DOWNED_PRED'] = lm_downed.fittedvalues

    # Model standing (live) surface fuel load as a function of near-surface plant area density
    lm_standing = smf.ols('LOAD_STANDING ~ pad_surface', surface_biomass_plot).fit()
    print(lm_standing.summary())
    surface_biomass_plot['LOAD_STANDING_PRED'] = lm_standing.fittedvalues

    print('All veg types:')
    surface_biomass_plot_f = surface_biomass_plot
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'],surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    print('CHAP type:')
    surface_biomass_plot_f = surface_biomass_plot[surface_biomass_plot['veg_type'] == 'CHAP']
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'], surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    print('PJO type:')
    surface_biomass_plot_f = surface_biomass_plot[surface_biomass_plot['veg_type'] == 'PJO']
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'], surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    print('PPO Type:')
    surface_biomass_plot_f = surface_biomass_plot[surface_biomass_plot['veg_type'] == 'PPO']
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'], surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    # Produce surface fuel regression figures
    f, [ax1, ax2] = plt.subplots(ncols=2, constrained_layout=True, figsize=[7, 4])
    sns.scatterplot(surface_biomass_plot, x='pai', y='LOAD_DOWNED', hue='veg_type', palette=['orange', 'green', 'blue'],
                    ax=ax1)
    sns.scatterplot(surface_biomass_plot, x='pad_surface', y='LOAD_STANDING', hue='veg_type',
                    palette=['orange', 'green', 'blue'], ax=ax2)
    ax1.axline((0, lm_downed.params.iloc[0]), slope=lm_downed.params.iloc[1], color='orange', linestyle='--')
    ax1.axline((0, lm_downed.params.iloc[0]), slope=lm_downed.params.iloc[2], color='green', linestyle='--')
    ax1.axline((0, lm_downed.params.iloc[0]), slope=lm_downed.params.iloc[3], color='blue', linestyle='--')
    ax2.axline((0, lm_standing.params.iloc[0]), slope=lm_standing.params.iloc[1], color='black', linestyle='--')
    f.tight_layout(pad=.5, w_pad=2.5)
    ax1.legend(loc='lower right',title='Veg Type')
    ax2.legend(loc='lower right',title='Veg Type')
    ax1.set_xlabel('LAI ($m^2$/$m^2$)')
    ax1.set_ylabel('Downed Surface Fuel Load (kg/$m^2$)')
    ax1.text(.05,.93,'$R^2$ = ' + str(lm_downed.rsquared.round(2)),transform=ax1.transAxes)
    ax1.text(.05,.88,'RMSE = ' + str(((lm_downed.resid ** 2).mean() ** .5).round(2))+ ' kg/$m^2$',transform=ax1.transAxes)
    ax2.set_xlabel('Surface LAD ($m^2$/$m^3$)')
    ax2.set_ylabel('Standing Surface Fuel Load (kg/$m^2$)')
    ax2.text(.05,.93,'$R^2$ = ' + str(lm_standing.rsquared.round(2)),transform=ax2.transAxes)
    ax2.text(.05,.88,'RMSE = ' + str(((lm_standing.resid ** 2).mean() ** .5).round(2)) + ' kg/$m^2$',transform=ax2.transAxes)
    plt.savefig(export_folder + '\\SurfaceFuelPAD.pdf')
    plt.savefig(export_folder + '\\SurfaceFuelPAD.png')
    plt.show()

    #%%#####################################################################################################################
    ### MODEL CANOPY FUELS ###
    # Remove surface fuel and outliers
    df_all = df_all[df_all['height'] >= 1]
    outliers = ['T1423071101','T0523080101']
    df_all = df_all[~(df_all['plot_id'].isin(outliers))]
    df_all = calculate_species_proportions(df_all,biomass_classes,'cbd_total')
    df_all['biomassPred'] = 0.0

    # Train model
    results = []
    if by_veg_type:
        # Calculate effective leaf mass per area separately for each vegetation type
        for veg_type in df_all['veg_type'].unique():
            # Filter to this vegetation type
            df_class = df_all[df_all['veg_type'] == veg_type]
            # Get list of plot names
            plots = pd.DataFrame(df_class['plot_id'].unique())
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
                train_data = df_class[(df_class['plot_id'].isin(train_plots))]
                test_data = df_class[(df_class['plot_id'].isin(test_plots))]
                # Train model on training set
                model_fitter = BulkDensityProfileModelFitter(train_data, biomass_classes, 'pad', 'cbd_total',
                                                             'height', 'plot_id', 'veg_type',
                                                             profile_smoothing_factor, 1)
                model_fitter.fit_mass_ratio_bayesian(prior_mean, prior_std, sigma_residuals, sigma_intercept, fit_intercept,
                                                     two_stage_fit)
                models = model_fitter.to_models()
                # Get predictions for test set
                model = models[veg_type]
                test_data.loc[:,'biomassPred'] = model.predict(test_data,'height', 'pad', 'plot_id')
                results.append(test_data)

            # Train and save final model based on all training data
            model_fitter = BulkDensityProfileModelFitter(df_class, biomass_classes, 'pad', 'cbd_total',
                                                         'height', 'plot_id', 'veg_type',
                                                         profile_smoothing_factor, 1)
            model_fitter.fit_mass_ratio_bayesian(prior_mean, prior_std, sigma_residuals, sigma_intercept, fit_intercept,
                                                 two_stage_fit)
            model = model_fitter.to_models()[veg_type]
            model.to_file(veg_type+'.model')
            print(f'Effective Leaf Area Density {veg_type} Model:')
            print(model_fitter.mass_ratio_dict)
            print()
    else:
        # Calculate effective leaf mass per area combining data from all vegetation types
        df_class = df_all
        # Get list of plot names
        plots = pd.DataFrame(df_class['plot_id'].unique())
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
            train_data = df_class[(df_class['plot_id'].isin(train_plots))]
            test_data = df_class[(df_class['plot_id'].isin(test_plots))]
            # Train model on training set
            model_fitter = BulkDensityProfileModelFitter(train_data, biomass_classes,'pad','cbd_total',
                                                  'height','plot_id','veg_type',
                                                  profile_smoothing_factor, 1)
            model_fitter.fit_mass_ratio_bayesian(prior_mean,prior_std,sigma_residuals,sigma_intercept,fit_intercept,two_stage_fit)
            models = model_fitter.to_models()
            # Get predictions for test set
            for veg_type in test_data['veg_type'].unique():
                model = models[veg_type]
                test_data.loc[test_data['veg_type']==veg_type,'biomassPred'] = model.predict(test_data[test_data['veg_type'] == veg_type],
                                                                                'height','pad','plot_id')
            results.append(test_data)
        # Train and save final model based on all training data
        model_fitter = BulkDensityProfileModelFitter(df_class, biomass_classes, 'pad', 'cbd_total',
                                                     'height', 'plot_id', 'veg_type',
                                                     profile_smoothing_factor, 1)
        model_fitter.fit_mass_ratio_bayesian(prior_mean, prior_std, sigma_residuals, sigma_intercept, fit_intercept,
                                             two_stage_fit)
        models = model_fitter.to_models()
        for veg_type in models:
            models[veg_type].to_file(veg_type+'.model')
        print('Effective Leaf Area Density Combined Model:')
        print(model_fitter.mass_ratio_dict)
        print()

    # Format dataframe with predictions
    results = pd.concat(results)
    results = results.fillna(0)
    results['biomass'] = results['cbd_total'] # Conventional estimate of canopy bulk density at a height bin, all species combined
    plots = results['plot_id'].unique()

    # For each plot, smooth results and write to file
    for plot in plots:
        plot_filter = results['plot_id'] == plot
        # Smooth observed data as it gets smoothed within model fitting process
        results.loc[plot_filter,'biomass'] = smooth(results.loc[plot_filter,'biomass'], profile_smoothing_factor)
        # Write results to csv files (separately for each plot)
        results.loc[plot_filter,:].to_csv(export_folder + '\\' + 'Final_Profile' + '\\' + plot + '.csv')

    # Aggregate canopy bulk density along profile
    results_plot = results.pivot_table(index='plot_id', aggfunc={'biomass': ['mean', 'max', 'sum'],
                                                                 'biomassPred': ['mean', 'max', 'sum'],
                                                                 'occluded': 'mean'}).reset_index()

    # Calculate total canopy fuel load (integral of vertical profile)
    results_plot['biomass', 'sum'] *= cell_size
    results_plot['biomassPred', 'sum'] *= cell_size

    results_plot['veg_type'] = [get_class(string) for string in results_plot['plot_id']]

    # Print accuracy results

    print("Canopy fuel load (all veg types)")
    score_cfl = score(results_plot['biomass', 'sum'], results_plot['biomassPred', 'sum'])

    print("Max CBD (all veg types)")
    score_cbd = score(results_plot['biomass', 'max'], results_plot['biomassPred', 'max'])


    print("Canopy fuel load (CHAP only)")
    results_plot_filter = results_plot[results_plot['veg_type']=='CHAP']
    score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'])
    print("Max CBD (CHAP only)")
    score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'])

    print("Canopy fuel load (PJO only)")
    results_plot_filter = results_plot[results_plot['veg_type']=='PJO']
    score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'])
    print("Max CBD (PJO only)")
    score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'])

    print("Canopy fuel load (PPO only)")
    results_plot_filter = results_plot[results_plot['veg_type']=='PPO']
    score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'])
    print("Max CBD (PPO only)")
    score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'])

    print("Canopy fuel load (CHAP + PJO only)")
    results_plot_filter = results_plot[results_plot['veg_type']!='PPO']
    score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'])
    print("Max CBD (CHAP + PJO only)")
    score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'])

    print("Canopy fuel load (CHAP + PPO only)")
    results_plot_filter = results_plot[results_plot['veg_type']!='PJO']
    score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'])
    print("Max CBD (CHAP + PPO only)")
    score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'])

    print("Canopy fuel load (PJO + PPO only)")
    results_plot_filter = results_plot[results_plot['veg_type']!='CHAP']
    score(results_plot_filter['biomass', 'sum'], results_plot_filter['biomassPred', 'sum'])
    print("Max CBD (PJO + PPO only)")
    score(results_plot_filter['biomass', 'max'], results_plot_filter['biomassPred', 'max'])


    # Produce regression accuracy plots for total canopy fuel load and max canopy bulk density
    f, [ax1, ax2] = plt.subplots(ncols=2, constrained_layout=True, figsize=[7, 4])
    sns.scatterplot(x=results_plot['biomass', 'sum'], y=results_plot['biomassPred', 'sum'],
                    hue=results_plot['veg_type'], palette=['orange', 'green', 'blue'], ax=ax1)
    sns.scatterplot(x=results_plot['biomass', 'max'], y=results_plot['biomassPred', 'max'],
                    hue=results_plot['veg_type'], palette=['orange', 'green', 'blue'], ax=ax2)
    ax1.axline([0, 0], slope=1)
    ax2.axline([0, 0], slope=1)
    ax1.set_ylim(ax1.get_xlim())
    ax2.set_ylim(ax2.get_xlim())
    f.tight_layout(pad=.5, w_pad=2.5)
    ax1.legend(loc='lower right',title='Veg Type')
    ax2.legend(loc='lower right',title='Veg Type')
    ax1.set_xlabel('Conventional Estimate Canopy Fuel Load (kg/$m^2$)')
    ax1.set_ylabel('Lidar Estimate Canopy Fuel Load (kg/$m^2$)')
    ax1.text(.05,.93,'$R^2$ = ' + str(round(score_cfl['r2'], 2)),transform=ax1.transAxes)
    ax1.text(.05,.88,'RMSE = ' + str(round(score_cfl['rmse'], 2)) + ' kg/$m^2$',transform=ax1.transAxes)
    ax2.set_xlabel('Conventional Estimate $CBD_{max}$ (kg/$m^3$)')
    ax2.set_ylabel('Lidar Estimate $CBD_{max}$ (kg/$m^3$)')
    ax2.text(.05,.93,'$R^2$ = ' + str(round(score_cbd['r2'], 2)),transform=ax2.transAxes)
    ax2.text(.05,.88,'RMSE = ' + str(round(score_cbd['rmse'], 2)) + ' kg/$m^3$',transform=ax2.transAxes)

    plt.savefig(export_folder + '_'.join(['Results', feature, str(cell_size), str(max_occlusion), str(profile_smoothing_factor)]) + '.pdf')
    plt.savefig(export_folder + '_'.join(
        ['Results', feature, str(cell_size), str(max_occlusion), str(profile_smoothing_factor)]) + '.png')

    plt.show()

    if bootstrap_confidence_intervals:
        nsamples = 5000
        # Get list of plot names
        plots = pd.DataFrame(df_all['plot_id'].unique()).to_numpy().flatten()
        coef_all = []
        for sample_ind in range(nsamples):
            resampled_df = []
            plot_indices = np.random.randint(0,plots.shape[0],plots.shape[0])
            for plot_i in plot_indices:
                resampled_df.append(df_class[df_all['plot_id']==plots[plot_i]])
            resampled_df = pd.concat(resampled_df).reset_index(drop=True)

            # Train model
            if by_veg_type:
                # Calculate effective leaf mass per area separately for each vegetation type
                for veg_type in resampled_df['veg_type'].unique():
                    # Filter to this vegetation type
                    model_fitter = BulkDensityProfileModelFitter(resampled_df[resampled_df['veg_type'==veg_type]],
                                                                 biomass_classes, 'pad', 'cbd_total',
                                                                 'height', 'plot_id', 'veg_type',
                                                                 profile_smoothing_factor, 1)
                    model_fitter.fit_mass_ratio_bayesian(prior_mean, prior_std, sigma_residuals, sigma_intercept,
                                                         fit_intercept,
                                                         two_stage_fit)
                    coef = model_fitter.mass_ratio_dict
                    coef['veg_type'] = veg_type
                    coef_all.append(coef)

            else:
                # Calculate effective leaf mass per area combining data from all vegetation types
                # Train and save final model based on all training data
                model_fitter = BulkDensityProfileModelFitter(resampled_df, biomass_classes, 'pad', 'cbd_total',
                                                             'height', 'plot_id', 'veg_type',
                                                             profile_smoothing_factor, 1)
                model_fitter.fit_mass_ratio_bayesian(prior_mean, prior_std, sigma_residuals, sigma_intercept,
                                                     fit_intercept,
                                                     two_stage_fit)
                coef = model_fitter.mass_ratio_dict
                coef['veg_type'] = 'combined'
                coef_all.append(coef)
        results = pd.DataFrame(coef_all)


        def quantile_05(series):
            return series.quantile(0.05)
        def quantile_95(series):
            return series.quantile(0.95)

        if by_veg_type:
            results_summary = results.pivot_table(aggfunc=['median','std', quantile_05, quantile_95], index='veg_type')
            print(results_summary)
        else:
            results_summary = results.agg(['median', 'std', quantile_05, quantile_95])
            print(results_summary)

    # %%#####################################################################################################################
    if generate_test_figure:
        import polars as pl

        plotname = 'T1423071101'
        summary_filepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
        basename = summary_filepath.name
        dempath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['DEM', basename]))
        pointspath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['Points', basename]))
        pts = pl.read_csv(pointspath)
        dem_pts = pl.read_csv(dempath)
        results_filter = results[(results['plot_id'] == plotname) & (results['height'] > 0)]
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

        plots = results['plot_id'].unique()
        np.random.shuffle(plots)
        for plotname in plots:
            summary_filepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
            basename = summary_filepath.name
            dempath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['DEM', basename]))
            pointspath = Path('/'.join(list(summary_filepath.parts[:-2]) + ['Points', basename]))
            pts = pl.read_csv(pointspath)
            dem_pts = pl.read_csv(dempath)
            results_filter = results[(results['plot_id'] == plotname) & (results['height'] > 0)]
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
            max_x_lim = ax2.get_xlim()[1]
            if max_x_lim <= .06:
                ax2.xaxis.set_major_locator(plt.MultipleLocator(.01))
            elif max_x_lim <= .12:
                ax2.xaxis.set_major_locator(plt.MultipleLocator(.02))
            else:
                ax2.xaxis.set_major_locator(plt.MultipleLocator(.05))
            ax2.legend()
            ax1.set_title(plotname)
            ax1.set_ylabel('Height (m)')
            ax1.set_xlabel('Easting (m)')
            ax2.set_xlabel('Canopy Bulk Density (kg/m^3)')
            plt.savefig(export_folder + plotname + '.pdf')
            plt.show()


if __name__ == '__main__':
    freeze_support()
    # set_start_method('spawn')
    main()