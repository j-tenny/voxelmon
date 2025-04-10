import voxelmon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read tables and make necessary adjustments
treelist = pd.read_csv('D:/DataWork/BurnPro3D/03_tree.csv')
# Replace species codes where there is no valid allometry
treelist['tree_sp'] = treelist['tree_sp'].replace({'ACGL':'ACMA3'})

plots = pd.read_csv('D:/DataWork/BurnPro3D/01_plot_identification.csv')
sp_ref = pd.read_csv('C:/Users/john1/OneDrive - Northern Arizona University/Work/PLANTS_LMA_Database/FIA_SPECIES.csv')
sp_ref['SPCD'] = sp_ref['SPCD'].astype(int)
sp_ref['W_SPGRPCD'] = sp_ref['W_SPGRPCD'].astype(int)
sp_ref = sp_ref[['SPCD','W_SPGRPCD','SPECIES_SYMBOL']].drop_duplicates()
lma_spcd = pd.read_csv('C:/Users/john1/OneDrive - Northern Arizona University/Work/PLANTS_LMA_Database/LMA_FIA_SPCD.csv')
lma_spcd = lma_spcd.rename(columns={'LMA (kg/m2)':'LMA'})
lma_spgrpcd = pd.read_csv('C:/Users/john1/OneDrive - Northern Arizona University/Work/PLANTS_LMA_Database/LMA_FIA_SPGRP.csv')
lma_spgrpcd = lma_spgrpcd.rename(columns={'mean':'LMA'})

treelist = treelist.merge(plots[['inventory_id','plot_blk','inventory_plot_diam','site_name']], on='inventory_id')
treelist = treelist.merge(sp_ref, left_on='tree_sp', right_on='SPECIES_SYMBOL', how='left', validate='many_to_one')

missing_codes = treelist.loc[treelist['W_SPGRPCD'].isna(),'tree_sp'].unique()
if len(missing_codes) > 0:
    raise KeyError('Codes in treelist not found in sp_ref: ' + str(missing_codes))
# Calculate crown ratio from canopy base height and total height
treelist['CR'] = 100*(1-(treelist['tree_htlcb'] / treelist['tree_ht']))
print(f'Number of recs where CR is nan: {treelist['CR'].isna().sum()} of {treelist.shape[0]}')
treelist.loc[treelist['CR'].isna(),'CR'] = treelist['CR'].mean()
# Calculate trees per hectare represented by each record. Note, column is labeled diam, but is actually radius.
treelist['TPA_UNADJ'] = 1 / (treelist['inventory_plot_diam'] ** 2 * np.pi / 10000)
# Edit column names to match FIA format
treelist = treelist.rename(columns={'tree_dbh':'DIA','tree_ht':'HT','W_SPGRPCD':'SPGRPCD','plot_blk':'PLT_CN'})
treelist = treelist[['PLT_CN','SPCD','SPGRPCD','DIA','HT','CR','TPA_UNADJ','site_name']]
# Get leaf biomass and leaf area per tree
treelist = voxelmon.estimate_foliage_from_treelist(treelist,lma_spcd,lma_spgrpcd)
# Divide trees into height bins
ht_interval=.2
# Write csv containing profiles organized by species
profiles = voxelmon.profiles_from_treelist(treelist, ht_interval=ht_interval)
profiles.to_csv('D:/DataWork/BurnPro3D/field_profiles.csv',index=False)
# Plot profiles not organized by species
profiles = voxelmon.profiles_from_treelist(treelist, ht_interval=ht_interval, by_species=False)
plot_ids = profiles['PLT_CN'].unique()
profile = profiles.loc[profiles['PLT_CN']==plot_ids[0],:]
plt.plot(profile['CBD'],profile['HT'])
plt.plot(profile['LAD'],profile['HT'])
plt.plot(profile['LMA'],profile['HT'])
plt.legend(['CBD','LAD','LMA'])
plt.show()
# Make summaries by stratum
treelist['DIA_SQ'] = treelist['DIA']**2
plot_summary = treelist.pivot_table(index=['site_name','PLT_CN'],
                                    aggfunc={'DIA_SQ':'mean','LMA':'mean','LEAF_AREA':'sum','DRYBIO_FOLIAGE':'sum','TPA_UNADJ':'sum'}).reset_index()
plot_summary['QMD'] = plot_summary['DIA_SQ']**.5
plot_summary.pivot_table(index='site_name',values=['QMD','DRYBIO_FOLIAGE','LMA','LEAF_AREA','TPA_UNADJ'])

