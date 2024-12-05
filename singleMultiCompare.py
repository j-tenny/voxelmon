import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
import voxelmon

singleScanDir = 'D:\\DataWork\\pypadResults\\PAD_Summary'
singleScanDemDir = 'D:\\DataWork\\pypadResults\\DEM'
multiScanDir = 'D:\\DataWork\\pypadResultsMulti\\PAD_Summary'
multiScanDemDir = 'D:\\DataWork\\pypadResultsMulti\\DEM'

singleScanFiles = voxelmon.get_files_list(singleScanDir,'.csv')
multiScanFiles = voxelmon.get_files_list(multiScanDir,'.csv')

outliers = ['T0523061403']

df_all = []
for i in range(len(singleScanFiles)):
    plot_name = singleScanFiles[i][-15:-4]
    df_single = pd.read_csv(singleScanFiles[i])
    df_single.insert(0,'plotID',plot_name)
    multiscan_filename = [file for file in multiScanFiles if plot_name in file][0]
    df_multi = pd.read_csv(multiscan_filename)
    dem_multi = pd.read_csv(multiScanDemDir+'\\'+plot_name+'.csv')
    if int(plot_name[1:3])==5:
        df_multi['class'] = 'CHAP'
    elif int(plot_name[1:3])<=12:
        df_multi['class'] = 'PJO'
    else:
        df_multi['class'] = 'PPO'

    df_combined = pd.merge(df_single, df_multi, on='heightBin', suffixes=('Single', 'Multi'))

    # Get terrain variables
    # Calculate horizontal distance from center
    dem_multi['HD'] = np.sqrt(dem_multi['x'] ** 2 + dem_multi['y'] ** 2)
    # Filter outside of plot radius
    dem_multi = dem_multi[dem_multi['HD'] < 11.3]
    dem_multi = dem_multi[~np.isnan(dem_multi['z'])]

    # Fit a plane using linear regression
    model = linear_model.LinearRegression().fit(dem_multi[['x', 'y']], dem_multi['z'])

    # Extract coefficients
    intercept = model.intercept_
    coef_x, coef_y = model.coef_

    # Normal vector of the plane
    normal_vector = np.array([coef_x, coef_y, -1])

    # Normalize the normal vector to get a unit vector
    normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
    if normal_unit_vector[2] < 0:
        normal_unit_vector *= -1

    # Unit vector along the Z-axis
    z_axis_vector = np.array([0, 0, 1])

    # Calculate the dot product between the two unit vectors
    dot_product = np.dot(normal_unit_vector, z_axis_vector)

    # Assign angle to data
    df_combined['terrainSlope'] = np.degrees(np.arccos(dot_product))

    # Calculate terrain shape metrics
    dem_multi['resid'] = dem_multi['z'] - model.predict(dem_multi[['x', 'y']])
    hd_half = 11.3 / 2
    sum_inner = dem_multi[dem_multi['HD'] < hd_half]['resid'].sum()
    sum_outer = dem_multi[dem_multi['HD'] >= hd_half]['resid'].sum()
    df_combined['terrainConcavity'] = sum_inner - sum_outer
    df_combined['terrainRoughness'] = np.sqrt(np.mean(dem_multi['resid'] ** 2))

    df_all.append(df_combined)

df_all = pd.concat(df_all)

df_all = df_all[~df_all['plotID'].isin(outliers)]
df_all = df_all[df_all['heightSingle']>=.1]
df_all = df_all[(df_all['foliageSingle']>0) | (df_all['foliageMulti']>0) | (df_all['padSingle']>0) | (df_all['padSingle']>0)]
df_all['totalVolume'] = 11.3**2 * 3.14159 * .1

df_all_plot = df_all.pivot_table(aggfunc = {'foliageSingle':'sum','foliageMulti':'sum','padSingle':'mean','padMulti':'mean',
                                            'occludedSingle':'mean','occludedMulti':'mean','totalVolume':'sum','terrainSlope':'first',
                                            'terrainRoughness':'first','terrainConcavity':'first','class':'first'},index='plotID')
df_all_plot['paiSingle'] = df_all_plot['padSingle'] * df_all_plot['totalVolume'] / (11.3**2 * 3.14159)
df_all_plot['paiMulti'] = df_all_plot['padMulti'] * df_all_plot['totalVolume'] / (11.3**2 * 3.14159)

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

print('Per bin with refit')


score(df_all['foliageMulti'],df_all['foliageSingle'])

sns.scatterplot(df_all,x='foliageSingle',y='foliageMulti',hue='class')
plt.axline((0,0),(1,1),color='red')
plt.show()

print('Per plot with refit')

score(df_all_plot['foliageMulti'],df_all_plot['foliageSingle'])

sns.scatterplot(df_all_plot,x='foliageSingle',y='foliageMulti',hue='class')
plt.axline((0,0),(1,1),color='red')
plt.show()

print('Per bin with refit')
score(df_all['padMulti'],df_all['padSingle'],refit=True)
lm = smf.ols('padMulti~padSingle',df_all).fit()
print(lm.profile())

f,[ax1,ax2] = plt.subplots(ncols=2, figsize = [7,4])

sns.scatterplot(df_all,x='padSingle',y='padMulti',hue='class',palette=['orange','green','blue'],ax=ax1)
ax1.axline((0,lm.params['Intercept']),slope=lm.params['padSingle'],color='black')
ax1.axline((0,0),(1,1),color='black',linestyle='dashed')
ax1.set_ylim(ax1.get_xlim())

ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax1.set_xlabel('Single-Scan LAD ($m^2$/$m^3$)')
ax1.set_ylabel('Multiple-Scan LAD ($m^2$/$m^3$)')
ax1.text(.05,.93,'$R^2$ = ' + str(lm.rsquared.round(2)),transform=ax1.transAxes)
ax1.text(.05,.88,'RMSE = ' + str(((lm.resid ** 2).mean() ** .5).round(2))+ ' $m^2$/$m^2$',transform=ax1.transAxes)

df_all['resid'] = lm.resid
df_all['sq_resid'] = lm.resid ** 2

print('Per plot with refit')

score(df_all_plot['paiMulti'],df_all_plot['paiSingle'],refit=True)


lm2 = smf.ols('paiMulti~paiSingle',df_all_plot).fit()
print(lm2.profile())
sns.scatterplot(df_all_plot,x='paiSingle',y='paiMulti',hue='class',palette=['orange','green','blue'],ax=ax2)
ax2.axline((0,lm2.params['Intercept']),slope=lm2.params['paiSingle'],color='black')
ax2.axline((0,0),(1,1),color='black',linestyle='dashed')
ax2.set_xlim(ax2.get_ylim())

ax2.set_xlabel('Single-Scan LAI ($m^2$/$m^2$)')
ax2.set_ylabel('Multiple-Scan LAI ($m^2$/$m^2$)')
ax2.text(.05,.93,'$R^2$ = ' + str(lm2.rsquared.round(2)),transform=ax2.transAxes)
ax2.text(.05,.88,'RMSE = ' + str(((lm2.resid ** 2).mean() ** .5).round(2)) + ' $m^2$/$m^2$',transform=ax2.transAxes)

f.tight_layout(pad=.5, w_pad=2.5)
plt.savefig('D:/DataWork/pypadResults/single-multi_pad.pdf')
plt.show()

sns.scatterplot(df_all[['class','occludedSingle','heightSingle']].groupby(['class','heightSingle']).mean(),x='occludedSingle',y='heightSingle',hue='class')
plt.show()
df_all_plot[['occludedMulti','occludedSingle']].mean()
df_plot_summary = df_all_plot.groupby('class').mean()

bins_corr = df_all[['resid','sq_resid','occludedSingle','occludedMulti','heightSingle','terrainSlope','terrainConcavity','terrainRoughness']].corr()
sns.heatmap(bins_corr)
plt.show()