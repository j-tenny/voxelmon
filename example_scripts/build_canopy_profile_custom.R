# Produce canopy bulk density profiles from conventional tree measurements with custom allometric equations
library(data.table)
library(readxl)
library(ggplot2)

group_species = TRUE
plot_area = .04
binSize = .1

field_path = '../TontoNF/FieldData.xlsx'

fieldTree = as.data.table(read_excel(field_path,'TREE'))[!is.na(SPECIES)]
fieldShrub = as.data.table(read_excel(field_path,'SHRUB'))[!is.na(SPECIES)]
species = as.data.table(read_excel(field_path,'SPECIES'))[!is.na(SPECIES)]

fieldTree$STEM_COUNT = 1
fieldShrub$DRC_MEASURE = fieldShrub$QM_DIAMETER
fieldShrub$DBH_EQ = 0
fieldShrub$DBH_EQ = exp(-0.35031+1.03991*log(fieldShrub$DRC_MEASURE))
fieldShrub[HEIGHT<=1.7 | DBH_EQ<.5,DBH_EQ:=.5]

field = rbind(fieldTree,fieldShrub,use.names=TRUE,fill=TRUE)
field = merge(field,species[,c('SPECIES','EQUATION','SPECIES_GROUP')],by='SPECIES',all.x=T,all.y=F)
field$BASE_HT[is.na(field$BASE_HT)] = 0
field = field[!is.na(field$HEIGHT),]
field$CR = as.integer(round(100 * (1 - field$BASE_HT / field$HEIGHT)))
field$CR[field$CR<10] = 10
#field$TREE_NUM[is.na(field$TREE_NUM)] = seq(1,sum(is.na(field$TREE_NUM)))

field[is.na(DRC_EQ) & DRC_MEASURE>0,DRC_EQ:=DRC_MEASURE]

field[is.na(SNAG_CLASS),SNAG_CLASS:='L']
#field = field[SNAG_CLASS=='L']

field[EQUATION=='McClaran',FOLIAR_BIOMASS:=STEM_COUNT * (exp((-4.88+1.67*log(DRC_EQ))*1.02))]
field[EQUATION=='Clary',FOLIAR_BIOMASS:= STEM_COUNT * ((0.24322 * (DRC_EQ*10)^1.66)/1000)]
field[EQUATION=='Grier_juniper',FOLIAR_BIOMASS:=STEM_COUNT * 10^(-1.737+1.382*log10(DRC_EQ))]
field[EQUATION=='Grier_pinyon',FOLIAR_BIOMASS:=STEM_COUNT * 10^(-.946+1.565*log10(DRC_EQ))]
field[EQUATION=='Hughes_manzanita',FOLIAR_BIOMASS:=STEM_COUNT * (exp(-1.923+2.054*log(DRC_EQ*10))/1000)]
field[EQUATION=='Hughes_snowbrush',FOLIAR_BIOMASS:=STEM_COUNT * (exp(-1.37+1.77*log(DRC_EQ*10))/1000)]
#field[EQUATION=='Westfall_122_310',FOLIAR_BIOMASS:=STEM_COUNT * ((.9455*(DBH_EQ/2.54)^2.77591*(HEIGHT*3.2808)^-0.82285)/2.205)] # Westfall et al NSVB
field[EQUATION=='Westfall_122_310',FOLIAR_BIOMASS:=STEM_COUNT * 1.0672 * exp(-4.1317+log(DBH_EQ)*2.0159)] # Keyes et al 2005
field[EQUATION=='Westfall_901',FOLIAR_BIOMASS:=STEM_COUNT * (1.369*DBH_EQ^2.3054*HEIGHT^-0.7064)]
field[EQUATION=='Succulent',FOLIAR_BIOMASS:=STEM_COUNT*.25] #Might want to replace with a real equation

field[SNAG_CLASS=='D',FOLIAR_BIOMASS:=0]
field[SNAG_CLASS=='U',FOLIAR_BIOMASS:=FOLIAR_BIOMASS*.5]
field[SNAG_CLASS=='S',FOLIAR_BIOMASS:=FOLIAR_BIOMASS*.25]

#field$BASAL_AREA = (field$DIAMETER/2.54)^2 * .005454
#field[,.(tpa = .N/binSize,basal=sum(BASAL_AREA)/binSize),by=PLOT_NAME]

options(scipen = 5)
species_summary = field[,.(basal = sum(BASAL_AREA,na.rm = T)),by=SPECIES]
species_summary[,pct := 100 * basal / sum(basal)]
species_summary[order(basal),]

if (group_species){
  field$SPECIES = field$SPECIES_GROUP
}

field_summary = field[SNAG_CLASS=='L',.(BASAL = sum(STEM_COUNT * (DBH_EQ*.01)^2,na.rm=T)/plot_area),by=.(PLOT_NAME,SPECIES)]
fwrite(field_summary,'../TontoNF/field_summary_output.csv')



i = 0
colname_template = c('plot_id','height_m',unique(field$SPECIES))
for (plotid in unique(field$PLOT_NAME)){
  field_slice = field[PLOT_NAME==plotid,]
  heights = seq(0,max(field_slice$HEIGHT),binSize)
  output = as.data.frame(matrix(0,ncol=length(colname_template),nrow=length(heights)))
  names(output) = colname_template
  output$plot_id = plotid
  output$height_m = heights
  for (row_index in seq(0,nrow(field_slice))){
    species = field_slice[row_index,SPECIES]
    base_ht = field_slice[row_index,BASE_HT]
    ht = field_slice[row_index,HEIGHT]
    bulk_density = field_slice[row_index,(FOLIAR_BIOMASS/(pi*11.3^2*(HEIGHT-BASE_HT)))]
    output[(output$height_m>=base_ht) & (output$height_m < ht),species] = output[(output$height_m>=base_ht) & (output$height_m < ht),species]+bulk_density
  }
  for (col in colname_template[-1:-2]){
    output[,col] = zoo::rollmean(output[,col],k=4,na.pad = T,align='center')
  }
  if (i==0){
    output_all = output
  }
  else{
    output_all = rbind(output_all,output)
  }
  i = i+1
}


output_all[is.na(output_all)]=0

ggplot(melt(rev(as.data.table(output_all[,-2])),id.vars = 'plot_id',value.name = 'biomass'),aes(plot_id,biomass,fill=variable))+
  geom_col()


ggplot(melt(rev(as.data.table(output_all[,-2])),id.vars = 'plot_id',value.name = 'biomass'),aes(plot_id,biomass,fill=variable))+
  geom_col()

output_all$TOTAL = apply(output_all[,3:ncol(output_all)],1,sum)

fwrite(output_all,'../TontoNF/AZ_CanProf_Output.csv')

