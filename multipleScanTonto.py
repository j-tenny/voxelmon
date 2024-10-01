import pandas as pd
import time
import numpy as np
from pathlib import Path
import pickle
from voxelmon import Grid,Pulses,PtxBlk360G1,FoliageProfileModel,get_files_list,plot_side_view
import statsmodels.formula.api as smf
import seaborn as sns

cellSize = .1
gridRadius = 12 # Distance from grid center to edge
plotRadius = 11.3
maxGridHeight = 30 # Height of grid above coordinate [0,0,0]
maxOcclusion = .8

inputFolder = '../TontoNF/TLS/PTX/AllScans/'
keyword = 'Med Density 1'
exportFolder = 'D:/DataWork/pypadResultsMulti/'
surfaceBiomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\SubplotBiomassEstimates.csv')
biomass = pd.read_csv('C:\\Users\\john1\\OneDrive - Northern Arizona University\\Work\\TontoNF\\AZ_CanProf_Output.csv')
biomassClasses = list(biomass.columns[2:-1])
#biomassClasses = ['PIPO','PINYON','JUNIPER','OAKS','ARPU','OTHER']
sigma = 1.2
feature = 'pad'
process = False
byVegType = False
leaveOneOut = True
generateTestFigure = False
generateFigures = False

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

def smooth(values,sigma):
    from scipy.ndimage import gaussian_filter1d
    valuesSmooth = gaussian_filter1d(values,sigma=sigma,mode='constant',cval=np.nan)
    # Fill missing values on with original, non-smoothed values
    valuesSmooth[np.isnan(valuesSmooth)] = values[np.isnan(valuesSmooth)]
    return valuesSmooth

baseExtents = [-gridRadius,-gridRadius,-gridRadius,gridRadius,gridRadius,maxGridHeight]

filesScan1 = get_files_list(inputFolder, keyword)
filesAll = get_files_list(inputFolder,'.ptx')
filesGrouped = []
for fileScan1 in filesScan1:
    plotId = get_plot_id(fileScan1)
    filesGrouped.append([file for file in filesAll if plotId in file])
start_time_all = time.time()
i=1
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
filesGrouped = filesGrouped[6:]

if process:
    if not Path(exportFolder).exists():
        os.makedirs(exportFolder)
    if not Path(exportFolder).joinpath('DEM').exists():
        os.mkdir(Path(exportFolder).joinpath('DEM'))
    if not Path(exportFolder).joinpath('Points').exists():
        os.mkdir(Path(exportFolder).joinpath('Points'))
    if not Path(exportFolder).joinpath('PAD').exists():
        os.mkdir(Path(exportFolder).joinpath('PAD'))
    if not Path(exportFolder).joinpath('PAD_Summary').exists():
        os.mkdir(Path(exportFolder).joinpath('PAD_Summary'))

    for filegroup in filesGrouped:
        start_time = time.time()
        print("Starting file ", i, " of ",len(filesGrouped))
        baseFileName = get_plot_id(filegroup[0])

        ptxGroup = [PtxBlk360G1(filegroup[0],applyTranslation=False,applyRotation=True,dropNull=False)]
        offset = -ptxGroup[0].originOriginal
        for ptxFile in filegroup[1:]:
            ptx = PtxBlk360G1(ptxFile, applyTranslation=True, applyRotation=True, dropNull=False, offset=offset)
            ptxGroup.append(ptx)

        pulsesList = []
        pulsesThinAll = []
        for ptx in ptxGroup:
            pulses = Pulses.from_point_cloud_array(ptx.xyz,ptx.origin)
            pulses_thin = pulses.crop(baseExtents).xyz
            pulsesThinAll.append(pulses_thin)
            pulsesList.append(pulses)
        pulsesThinAll = np.concatenate(pulsesThinAll)

        import polars as pl
        tmp = pl.DataFrame(pulsesThinAll)
        tmp.write_csv(os.path.join(exportFolder, 'Points/', baseFileName) + '.csv')


        #z_range = pulsesThinAll[~is_noise_ivf(pulsesThinAll),2]
        #minHeight = z_range.min()
        #maxHeight = z_range.max()
        minHeight = pulsesThinAll[:,2].min()
        maxHeight = pulsesThinAll[:, 2].max()+1
        extents = baseExtents.copy()
        extents[2] = minHeight
        extents[5] = maxHeight

        #pulsesThinAll = pulsesThinAll.thin_distance_weighted_random(.25)
        grid = Grid(extents=extents,cellSize=cellSize)

        grid.create_dem_decreasing_window(pulsesThinAll)

        grid.calculate_pulse_metrics(pulsesList[0])

        if len(pulsesList)>1:
            for pulses in pulsesList[1:]:
                grid_temp = Grid(extents=extents,cellSize=cellSize)
                grid_temp.calculate_pulse_metrics(pulses)
                grid.add_pulse_metrics(grid_temp)

        #grid.calculate_eigenvalues(pulsesThinAll.array[:,3:6])

        grid.filter_pad_noise_ivf()
        grid.gaussian_filter_PAD(sigma=.1)
        #threshold = np.quantile(grid.pad[(grid.pad>0) & (grid.occlusion<=maxOcclusion)],.9)
        threshold = 6
        grid.classify_foliage_with_PAD(maxOcclusion=maxOcclusion,minPADFoliage=.01,maxPADFoliage=threshold)
        summary = grid.summarize_by_height(clipRadius=plotRadius)



        grid.export_grid_as_csv(os.path.join(exportFolder,'PAD/',baseFileName) + '.csv')

        grid.export_dem_as_csv(os.path.join(exportFolder,'DEM/',baseFileName) + '.csv')

        summary.write_csv(os.path.join(exportFolder, 'PAD_Summary/', baseFileName) + '.csv')

        plt.plot(summary['pad'], summary['height'])
        plt.show()

        print("Finished file ", i, " of ", len(filesGrouped)," in ", round(time.time()-start_time,3)," seconds")
        i += 1

    print("Finished all files in ",round(time.time()-start_time_all)," seconds")

from scipy.ndimage import gaussian_filter1d,uniform_filter1d
import math
import statsmodels.formula.api as smf

biomass_list = []
for plot in biomass['Plot_ID'].unique():
    biomass_filter = biomass[(biomass['Plot_ID']==plot) & (biomass['Height_m']>=1)].copy()
    for col in biomassClasses:
        biomass_filter[col] = smooth(biomass_filter[col],sigma=sigma)
    biomass_filter['TOTAL'] = smooth(biomass_filter['TOTAL'],sigma=sigma)
    biomass_list.append(biomass_filter)
biomass = pd.concat(biomass_list)
biomass['CLASS'] = [get_class(string) for string in biomass['Plot_ID']]
biomass['heightBin'] = (biomass['Height_m'] / cellSize).round().astype(int)

tls_summary_files = get_files_list(exportFolder+'/PAD_Summary','.csv')
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
surfaceBiomassPlot = surfaceBiomass.pivot_table(index='PLOT_NAME',values=['LOAD_LITTER_DUFF','LOAD_DOWN_WOODY','LOAD_STANDING','LOAD_TOTAL_SURFACE'],aggfunc='mean')
surfaceBiomassPlot['LOAD_DOWNED'] = surfaceBiomassPlot['LOAD_LITTER_DUFF'] + surfaceBiomassPlot['LOAD_DOWN_WOODY']
plotCanopy = df_all[df_all['height']>=.1].pivot_table(index='Plot_ID',aggfunc={'CLASS':'first','foliage':'sum','pad':'sum'})
plotSurface = df_all[(df_all['height']>=.1) & (df_all['height']<1)].pivot_table(index='Plot_ID',values=['foliage','pad'],aggfunc='mean')
surfaceBiomassPlot = surfaceBiomassPlot.join(plotCanopy)
surfaceBiomassPlot = surfaceBiomassPlot.join(plotSurface,lsuffix='_canopy',rsuffix='_surface')
surfaceBiomassPlot = surfaceBiomassPlot.dropna()

sns.scatterplot(surfaceBiomassPlot,x='pad_canopy',y='LOAD_DOWNED',hue='CLASS')
plt.show()

lm = smf.ols('LOAD_DOWNED ~ pad_canopy:CLASS',surfaceBiomassPlot).fit()
print(lm.summary())
print('RMSE = ',(lm.resid**2).mean()**.5)

sns.scatterplot(surfaceBiomassPlot,x='pad_surface',y='LOAD_STANDING',hue='CLASS')
plt.show()

lm = smf.ols('LOAD_STANDING ~ pad_surface:CLASS',surfaceBiomassPlot).fit()
print(lm.summary())
print('RMSE = ',(lm.resid**2).mean()**.5)

df_all = df_all[df_all['height']>=1]
outliers = ['T0523061403','T1423071101']
df_all = df_all[~(df_all['Plot_ID'].isin(outliers))]

results = []
if byVegType:
    for classId in df_all['CLASS'].unique():
        df_class = df_all[df_all['CLASS']==classId]
        plots = pd.DataFrame(df_class['Plot_ID'].unique())
        if leaveOneOut:
            nfolds = len(plots)
        else:
            nfolds = 10
        kvals = np.arange(nfolds).repeat(int(math.ceil(len(plots)/nfolds)))
        np.random.shuffle(kvals)
        plots['k'] = kvals[:len(plots)]
        for k in range(nfolds):
            trainPlots = plots.loc[plots['k']!=k,0]
            testPlots = plots.loc[plots['k']==k,0]
            trainData = df_class[(df_class['Plot_ID'].isin(trainPlots))]
            testData = df_class[(df_class['Plot_ID'].isin(testPlots))]
            model = FoliageProfileModel()
            model.fit(trainData, trainData, biomassCols=biomassClasses, cellSize=cellSize, sigma=sigma, lidarValueCol=feature)
            pred = model.predict(testData,lidarValueCol=feature)
            results.append(pred)
        with open(classId+'.model','wb') as f:
            pickle.dump(model,f)
else:
    df_class = df_all
    plots = pd.DataFrame(df_class['Plot_ID'].unique())
    if leaveOneOut:
        nfolds = len(plots)
    else:
        nfolds = 3
    kvals = np.arange(nfolds).repeat(int(math.ceil(len(plots) / nfolds)))
    np.random.shuffle(kvals)
    plots['k'] = kvals[:len(plots)]
    for k in range(nfolds):
        trainPlots = plots.loc[plots['k'] != k, 0]
        testPlots = plots.loc[plots['k'] == k, 0]
        trainData = df_class[(df_class['Plot_ID'].isin(trainPlots))]
        testData = df_class[(df_class['Plot_ID'].isin(testPlots))]
        model = FoliageProfileModel()
        model.fit(lidarProfile = trainData, biomassProfile=trainData, biomassCols=biomassClasses, cellSize=cellSize, sigma=sigma,
                  lidarValueCol=feature, fitIntercept=True, twoStageFit=True)
        pred = model.predict(testData, lidarValueCol=feature)
        results.append(pred)
    with open('combined.model', 'wb') as f:
        pickle.dump(model, f)

results = pd.concat(results)
results['volTotal'] = math.pi * gridRadius**2 * cellSize
results = results.fillna(0)
results['biomass'] = results['TOTAL']
plots = results['Plot_ID'].unique()

for plot in plots:
    results_filter = results[results['Plot_ID']==plot]
    results_filter.to_csv(exportFolder + '\\' + 'Final_Profile' + '\\' + plot+'.csv')


import matplotlib.pyplot as plt

results_plot = results.pivot_table(index='Plot_ID',aggfunc={'biomass':['mean','max'],
                                                            'biomassPred':['mean','max'],
                                                            'occluded':'mean', 'volTotal':'sum'}).reset_index()

results_plot['biomass','sum'] = results_plot['biomass','mean'] * results_plot['volTotal','sum'] / (math.pi * plotRadius**2 / 10000)
results_plot['biomassPred','sum'] = results_plot['biomassPred','mean'] * results_plot['volTotal','sum'] / (math.pi * plotRadius**2 / 10000)

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
plt.savefig(exportFolder+'_'.join(['Results',feature,str(cellSize),str(maxOcclusion),str(sigma)])+'.pdf')
plt.show()

if generateTestFigure:
    import polars as pl
    plotname = 'T1423080202'
    summaryFilepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
    basename = summaryFilepath.name
    dempath = Path('/'.join(list(summaryFilepath.parts[:-2])+['DEM',basename]))
    pointspath = Path('/'.join(list(summaryFilepath.parts[:-2])+['Points',basename]))
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

if generateFigures:
    import polars as pl
    plots = results['Plot_ID'].unique()
    np.random.shuffle(plots)
    for plotname in plots:
        summaryFilepath = [Path(filename) for filename in tls_summary_files if plotname in filename][0]
        basename = summaryFilepath.name
        dempath = Path('/'.join(list(summaryFilepath.parts[:-2])+['DEM',basename]))
        pointspath = Path('/'.join(list(summaryFilepath.parts[:-2])+['Points',basename]))
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
        plt.savefig(exportFolder+plotname+'.pdf')
        plt.show()

