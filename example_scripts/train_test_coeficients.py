# Train and test a canopy bulk density profile model for single-scan lidar using methods from Tenny et al 2025
# Requires addtional libraries `conda install pymc scikit-learn seaborn`
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
from multiprocessing import Process, freeze_support, set_start_method

import voxelmon
from voxelmon import TLS_PTX, BulkDensityProfileModelFitter, get_files_list, directory_to_pandas, plot_side_view, calculate_species_proportions, smooth

########################################################################################################################

def main():
    ### Setup Run Parameters ###
    cell_size = .1 # Voxel side length
    max_occlusion = .8 # Threshold to determine when to classify a voxel as occluded
    export_folder = 'D:/DataWork/TontoUpdatedResultsSingle/' # Root folder for results
    biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\AZ_CanProf_Output.csv') # Conventional estimates of canopy bulk density by species group by height bin by plot. Produced by BuildCanopyProfile.R
    biomass_ht_col = 'HEIGHT_M'
    biomass_plot_col = 'PLOT_ID'
    biomass_classes = ['PINE','JUNIPER','ARPU','OT','OS']
    lma = [0.333, .49, 0.28, 0.141, .141]
    surface_biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\SubplotBiomassEstimates.csv') # Estimated surface fuels by fuel component by subplot by plot

    profile_smoothing_factor = 0 # Value used as input to voxelmon.utils.smooth() when smoothing profiles
    feature = 'PAD' # Can be 'PAD' to train model with plant area density or 'FOLIAGE' to train model with foliage volume
    prior_mean = np.ones(len(lma))
    prior_std = .05 # .05 represents a strong prior
    sigma_residuals = 1
    sigma_intercept = .005
    fit_intercept = False
    two_stage_fit = True
    leave_one_out = False # True: Use leave-one-out cross validation. False: Use k-fold cross validation
    nfolds = 10 # Number of folds used in k-fold cross validation. Set to 1 to train with all data (can't test).
    variance_stats = True # Generate stats for effects of terrain, occlusion, height


    #%%#####################################################################################################################
    ### Define Helper Functions ###
    def get_class(plot_name):
        # Determine vegetation type from plot name
        code = int(plot_name[1:3])
        if code<5:
            return 'DES'
        elif code == 5:
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
        mean_error = (y_pred - y_obs).mean()

        m = y_obs.mean()
        rrmse = rmse / m

        mape = np.abs((y_obs - y_pred) / y_obs).mean()
        wmape = np.abs((y_obs - y_pred)).sum() / y_obs.sum()
        wmpe = (y_obs - y_pred).sum() / y_obs.sum()

        if print_output:
            print('R^2: ', r2)
            print('RMSE: ', rmse)
            print('RRMSE: ', rrmse)  # relative rmse
            print('MeanError: ',mean_error)
            #print('MAE: ', mae)
            print('MeanObs: ', m)  # mean of observed values
            #print('MAPE: ', mape)  # mean absolute percent error
            #print('WMAPE: ', wmape)  # mean absolute percent error weighted by y_obs
            #print('WMPE: ', wmpe)  # mean percent error weighted by y_obs
            print()
        return {'r2': r2, 'rmse': rmse, 'rrmse': rrmse}


    ### Read and concatenate vertical profiles from all plots ###
    df_all = directory_to_pandas(Path(export_folder) / 'PAD_Profile', filename_col='PLT_CN')

    ### Format dataframe containing conventional biomass estimates ###
    # Add column for height bin as integer
    biomass.columns = [col.upper() for col in biomass.columns]
    biomass['HEIGHT_BIN'] = (biomass[biomass_ht_col] / cell_size).round().astype(int)
    biomass['CBD_TOTAL'] = biomass[biomass_classes].sum(1)
    biomass = biomass.drop(columns=biomass_ht_col)
    ### Merge TLS plant area density profiles with conventional bulk density profiles
    df_all = df_all.merge(biomass, left_on=['PLT_CN', 'HEIGHT_BIN'], right_on=[biomass_plot_col,'HEIGHT_BIN'], how='outer')
    df_all = df_all[pd.notna(df_all['PLT_CN'])]
    # Add column for vegetation type
    df_all['VEG_TYPE'] = [get_class(string) for string in df_all['PLT_CN']]
    df_all = df_all[df_all['VEG_TYPE']!='DES'] # Remove this veg type where we didn't record trees

    ### Fill empty cells with 0, remove height bins where TLS and conventional biomass are both ~0
    df_all = df_all.fillna(0)
    df_all = df_all[~((df_all[feature] < .005) & (df_all['CBD_TOTAL'] < .005))]

    ### Remove plots that are unexplained outliers
    #outliers = ['T0523061403']
    outliers = []
    df_all = df_all[~(df_all['PLT_CN'].isin(outliers))]

    #%%#####################################################################################################################
    ### Train and Test Surface Fuel Models ###

    # Summarize conventional surface fuel load estimates by plot
    surface_biomass.rename(columns={'PLOT_NAME':'PLT_CN'},inplace=True,errors='ignore')
    surface_biomass_plot = surface_biomass.pivot_table(index='PLT_CN',
                                                       values=['LOAD_LITTER', 'LOAD_DOWN_WOODY', 'LOAD_STANDING',
                                                            'LOAD_TOTAL_SURFACE'], aggfunc='mean')

    # Combine litter and downed woody debris
    surface_biomass_plot['LOAD_DOWNED'] = surface_biomass_plot['LOAD_LITTER'] + surface_biomass_plot['LOAD_DOWN_WOODY']

    # Summarize TLS data by plot (height .3m to top of canopy)
    plot_canopy = df_all[df_all['HT'] >= .3].pivot_table(index='PLT_CN', aggfunc={'VEG_TYPE': 'first', 'FOLIAGE': 'sum', 'PAD': 'sum'})
    # Calculate plant area index (sum of plant area density profile * height bin size)
    plot_canopy['PAI'] = plot_canopy['PAD'] * cell_size

    # Summarize TLS data by plot (height .2m to 1m
    plot_surface = df_all[(df_all['HT'] >= .2) & (df_all['HT'] <= 1)].pivot_table(index='PLT_CN',
                                                                                         values=['FOLIAGE', 'PAD','OCCLUDED'],
                                                                                         aggfunc='mean')
    plot_surface.rename(columns = {'OCCLUDED':'OCCLUDED_SURFACE'},inplace=True)
    # Merge conventional surface fuel estimates, TLS canopy metrics, TLS surface metrics
    surface_biomass_plot = surface_biomass_plot.join(plot_canopy)
    surface_biomass_plot = surface_biomass_plot.join(plot_surface, lsuffix='_CANOPY', rsuffix='_SURFACE')
    surface_biomass_plot = surface_biomass_plot.dropna()

    # Model downed woody debris as a function of canopy plant area index by vegetation type
    lm_downed = smf.ols('LOAD_DOWNED ~ PAI:VEG_TYPE', surface_biomass_plot).fit()
    print(lm_downed.summary())
    #print('RMSE = ', (lm_downed.resid ** 2).mean() ** .5)
    surface_biomass_plot['LOAD_DOWNED_PRED'] = lm_downed.fittedvalues

    # Model standing (live) surface fuel load as a function of near-surface plant area density
    lm_standing = smf.ols('LOAD_STANDING ~ PAD_SURFACE', surface_biomass_plot).fit()
    print(lm_standing.summary())
    #print('RMSE = ', (lm_standing.resid ** 2).mean() ** .5)
    surface_biomass_plot['LOAD_STANDING_PRED'] = lm_standing.fittedvalues

    print('All veg types:')
    surface_biomass_plot_f = surface_biomass_plot
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'],surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    print('CHAP type:')
    surface_biomass_plot_f = surface_biomass_plot[surface_biomass_plot['VEG_TYPE'] == 'CHAP']
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'], surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    print('PJO type:')
    surface_biomass_plot_f = surface_biomass_plot[surface_biomass_plot['VEG_TYPE'] == 'PJO']
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'], surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    print('PPO Type:')
    surface_biomass_plot_f = surface_biomass_plot[surface_biomass_plot['VEG_TYPE'] == 'PPO']
    print('Surface fuel load downed:')
    score(surface_biomass_plot_f['LOAD_DOWNED'], surface_biomass_plot_f['LOAD_DOWNED_PRED'])
    print('Surface fuel load standing:')
    score(surface_biomass_plot_f['LOAD_STANDING'], surface_biomass_plot_f['LOAD_STANDING_PRED'])

    # Produce surface fuel regression figures
    f, [ax1, ax2] = plt.subplots(ncols=2, constrained_layout=True, figsize=[7, 4])
    sns.scatterplot(surface_biomass_plot, x='PAI', y='LOAD_DOWNED', hue='VEG_TYPE', palette=['orange', 'green', 'blue'],
                    ax=ax1)
    sns.scatterplot(surface_biomass_plot, x='PAD_SURFACE', y='LOAD_STANDING', hue='VEG_TYPE',
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
    plt.savefig(export_folder + '\\SurfaceFuelPAD.png')
    plt.show()

    #%%#####################################################################################################################
    ### MODEL CANOPY FUELS ###
    # Remove surface fuel and outliers
    df_all = df_all[df_all['HT'] >= 1]
    outliers = ['T1423071101']
    #outliers = []
    df_all = df_all[~(df_all['PLT_CN'].isin(outliers))]
    df_all = calculate_species_proportions(df_all,biomass_classes,'CBD_TOTAL')
    df_all['CBD_PRED'] = 0.0

    # Train model
    results = []
    # Get list of plot names
    plots = pd.DataFrame(df_all['PLT_CN'].unique())
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
        train_data = df_all[(df_all['PLT_CN'].isin(train_plots))]
        test_data = df_all[(df_all['PLT_CN'].isin(test_plots))]
        # Train model on training set
        model_fitter = BulkDensityProfileModelFitter(train_data, biomass_classes,feature,'CBD_TOTAL',
                                              'HT','PLT_CN','VEG_TYPE',
                                              profile_smoothing_factor, 1)
        model_fitter.fit_mass_ratio_bayesian(lma, prior_mean,prior_std,sigma_residuals,sigma_intercept,fit_intercept,two_stage_fit)
        #model_fitter.fit_mass_ratio_ols(lma, fit_intercept, two_stage_fit)

        models = model_fitter.to_models()
        # Get predictions for test set
        for veg_type in test_data['VEG_TYPE'].unique():
            model = models[veg_type]
            test_data.loc[test_data['VEG_TYPE']==veg_type,'CBD_PRED'] = model.predict(test_data[test_data['VEG_TYPE'] == veg_type],
                                                                            'HT',feature,'PLT_CN')
        results.append(test_data)
    # Train and save final model based on all training data
    model_fitter = BulkDensityProfileModelFitter(df_all, biomass_classes, feature, 'CBD_TOTAL',
                                                 'HT', 'PLT_CN', 'VEG_TYPE',
                                                 profile_smoothing_factor, 1)
    model_fitter.fit_mass_ratio_bayesian(lma, prior_mean, prior_std, sigma_residuals, sigma_intercept, fit_intercept,
                                         two_stage_fit)
    model_fitter.to_files(export_folder)
    models = model_fitter.to_models()
    for veg_type in models:
        models[veg_type].to_file(export_folder + '\\' + veg_type + '.model')
    print('Combined Model Lidar Coef:')
    print(model_fitter.lidar_coef_dict)
    print('Combined Model Final Effective Mass Ratio:')
    print(model_fitter.mass_ratio_dict)
    print()

    # Format dataframe with predictions
    results = pd.concat(results)
    results = results.fillna(0)

    # Aggregate canopy bulk density along profile
    summary_conv = voxelmon.summarize_profiles(results,min_height=1,cbd_col='CBD_TOTAL',height_col='HT',
                                               pad_col='PAD',plot_id_col='PLT_CN')
    summary_pred = voxelmon.summarize_profiles(results,min_height=1, cbd_col='CBD_PRED', height_col='HT',
                                               pad_col='PAD', plot_id_col='PLT_CN')

    results_plot = summary_conv.merge(summary_pred,on='PLT_CN',suffixes=('','_PRED'))

    results_plot['VEG_TYPE'] = [get_class(string) for string in results_plot['PLT_CN']]

    # Print accuracy results

    print("Canopy fuel load (all veg types)")
    score_cfl = score(results_plot['CFL'], results_plot['CFL_PRED'], refit=True)

    print("Max CBD (all veg types)")
    score_cbd = score(results_plot['CBD'], results_plot['CBD_PRED'], refit=True)


    print("Canopy fuel load (CHAP only)")
    results_plot_filter = results_plot[results_plot['VEG_TYPE']=='CHAP']
    score(results_plot_filter['CFL'], results_plot_filter['CFL_PRED'], refit=True)
    print("Max CBD (CHAP only)")
    score(results_plot_filter['CBD'], results_plot_filter['CBD_PRED'], refit=True)

    print("Canopy fuel load (PJO only)")
    results_plot_filter = results_plot[results_plot['VEG_TYPE']=='PJO']
    score(results_plot_filter['CFL'], results_plot_filter['CFL_PRED'], refit=True)
    print("Max CBD (PJO only)")
    score(results_plot_filter['CBD'], results_plot_filter['CBD_PRED'], refit=True)

    print("Canopy fuel load (PPO only)")
    results_plot_filter = results_plot[results_plot['VEG_TYPE']=='PPO']
    score(results_plot_filter['CFL'], results_plot_filter['CFL_PRED'], refit=True)
    print("Max CBD (PPO only)")
    score(results_plot_filter['CBD'], results_plot_filter['CBD_PRED'], refit=True)

    print("Canopy fuel load (CHAP + PJO only)")
    results_plot_filter = results_plot[results_plot['VEG_TYPE']!='PPO']
    score(results_plot_filter['CFL'], results_plot_filter['CFL_PRED'], refit=True)
    print("Max CBD (CHAP + PJO only)")
    score(results_plot_filter['CBD'], results_plot_filter['CBD_PRED'], refit=True)

    print("Canopy fuel load (CHAP + PPO only)")
    results_plot_filter = results_plot[results_plot['VEG_TYPE']!='PJO']
    score(results_plot_filter['CFL'], results_plot_filter['CFL_PRED'], refit=True)
    print("Max CBD (CHAP + PPO only)")
    score(results_plot_filter['CBD'], results_plot_filter['CBD_PRED'], refit=True)

    print("Canopy fuel load (PJO + PPO only)")
    results_plot_filter = results_plot[results_plot['VEG_TYPE']!='CHAP']
    score(results_plot_filter['CFL'], results_plot_filter['CFL_PRED'], refit=True)
    print("Max CBD (PJO + PPO only)")
    score(results_plot_filter['CBD'], results_plot_filter['CBD_PRED'], refit=True)


    # Produce regression accuracy plots for total canopy fuel load and max canopy bulk density
    f, [ax1, ax2] = plt.subplots(ncols=2, constrained_layout=True, figsize=[7, 4])
    sns.scatterplot(x=results_plot['CFL'], y=results_plot['CFL_PRED'],
                    hue=results_plot['VEG_TYPE'], palette=['orange', 'green', 'blue'], ax=ax1)
    sns.scatterplot(x=results_plot['CBD'], y=results_plot['CBD_PRED'],
                    hue=results_plot['VEG_TYPE'], palette=['orange', 'green', 'blue'], ax=ax2)
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

    plt.savefig(export_folder + '_'.join(
        ['Results', feature, str(cell_size), str(max_occlusion), str(profile_smoothing_factor)]) + '.png')

    plt.show()


    #%%#####################################################################################################################
    # Look for patterns in the residuals
    if variance_stats:
        from sklearn.preprocessing import robust_scale
        from pygam import LinearGAM, s

        # Monkey patch to fix issue in pygam
        import scipy.sparse
        def to_array(self):
            return self.toarray()
        scipy.sparse.spmatrix.A = property(to_array)

        def plot_bias_var_partial_dependence(df,col_pred,col_obs,col_variance_predictors):
            df = df.copy()
            df['resid'] = df[col_pred] - df[col_obs]
            df['resid_sq'] = df['resid'] ** 2
            variance_predictors = df[col_variance_predictors]

            # Fit a GAM model
            gam_bias = LinearGAM().fit(variance_predictors, df['resid'])
            gam_variance = LinearGAM().fit(variance_predictors, df['resid_sq'])
            # Extract variance explained by each smooth term
            for i, term in enumerate(gam_bias.terms):
                if term.isintercept:
                    continue

                XX = gam_bias.generate_X_grid(term=i)
                pdep, confi = gam_bias.partial_dependence(term=i, X=XX, width=0.95)

                f, ax = plt.subplots(1, 1)
                ax.scatter(variance_predictors.iloc[:, i], df['resid'], color='grey', s=2)
                ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], color='black')
                ax.plot(XX[:, term.feature], pdep)
                ax.plot(XX[:, term.feature], confi, c='r', ls='--')
                ax.set_xlabel(variance_predictors.columns[i])
                ax.set_ylabel('Residual')
                plt.title(f"Partial Dependence of Bias in {col_pred}")
                plt.show()

            for i, term in enumerate(gam_variance.terms):
                if term.isintercept:
                    continue

                XX = gam_variance.generate_X_grid(term=i)
                pdep, confi = gam_variance.partial_dependence(term=i, X=XX, width=0.95)

                f, ax = plt.subplots(1, 1)
                ax.scatter(variance_predictors.iloc[:, i], df['resid_sq'], color='grey', s=.5)
                ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], color='black')
                ax.plot(XX[:, term.feature], pdep)
                ax.plot(XX[:, term.feature], confi, c='r', ls='--')
                ax.set_xlabel(variance_predictors.columns[i])
                ax.set_ylabel('Squared Residual')
                plt.title(f"Partial Dependence of Variance in {col_pred}")
                plt.show()


        terrain_stats = directory_to_pandas(r'D:\DataWork\TontoUpdatedResultsSingle\Plot_Summary')
        results = results.merge(terrain_stats,on = 'PLT_CN')
        results['resid'] = results['CBD_PRED'] - results['CBD_TOTAL']
        results['resid_sq'] = results['resid']**2
        # Remove extreme values (more than 3 std from mean)
        variance_predictors_names = ['HT', 'OCCLUDED', 'CBD_TOTAL','TERRAIN_SLOPE','TERRAIN_CONCAVITY','TERRAIN_ROUGHNESS']
        for col in variance_predictors_names:
            std = results[col].std()
            mean = results[col].mean()
            results = results[abs(results[col]-mean) <= 3*std]
        std_resid_sq = results['resid_sq'].std()
        results = results[results['resid_sq'] <= 3*std_resid_sq]

        plot_bias_var_partial_dependence(results,'CBD_PRED','CBD_TOTAL',variance_predictors_names)


        # Repeat for plot canopy metrics
        results_plot.columns = ['_'.join(col).strip() if col[1]!='' else col[0] for col in results_plot.columns.values]
        results_plot = results_plot.merge(terrain_stats,on = 'PLT_CN')
        variance_predictors_names = ['OCCLUDED_MEAN', 'CFL', 'TERRAIN_SLOPE', 'TERRAIN_CONCAVITY', 'TERRAIN_ROUGHNESS']

        plot_bias_var_partial_dependence(results_plot,'CFL_PRED','CFL',variance_predictors_names)
        plot_bias_var_partial_dependence(results_plot,'CBD_PRED','CBD',variance_predictors_names)

        # Repeat for plot surface metrics
        results_plot = results_plot.join(surface_biomass_plot,on = 'PLT_CN',rsuffix='_SURFACE',how='inner')
        plot_bias_var_partial_dependence(results_plot, 'LOAD_DOWNED_PRED', 'LOAD_DOWNED', variance_predictors_names)
        variance_predictors_names = ['OCCLUDED_SURFACE', 'CFL', 'TERRAIN_SLOPE', 'TERRAIN_CONCAVITY','TERRAIN_ROUGHNESS']
        plot_bias_var_partial_dependence(results_plot, 'LOAD_STANDING_PRED', 'LOAD_STANDING', variance_predictors_names)


if __name__ == '__main__':
    freeze_support()
    #set_start_method('spawn')
    main()