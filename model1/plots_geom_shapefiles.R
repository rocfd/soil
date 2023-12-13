library(tidyverse)
library(rgdal)
library(sf)

#load data
shapefile <- st_read("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/Data/provided_soil_data_LUCAS-SOIL-2018-data-report-readme-v2/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018 .shp")

shapefile_train<- shapefile %>% filter(POINTID %in% train_meta_data$POINTID)
train_meta_data<- read.delim("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/meta_data_with_diversity.csv", sep = ",", row.names = 1)

train_meta_data <- train_meta_data[match(shapefile_train$POINTID, train_meta_data$POINTID), ]
shapefile_train$Elevation<- train_meta_data$Elevation
shapefile_train$Electrical_conductivity<- train_meta_data$Electrical_conductivity
shapefile_train$P<- train_meta_data$P
shapefile_train$N<- train_meta_data$N
shapefile_train$clean_LU<- train_meta_data$clean_LU
shapefile_train$bacterial_diversity<- train_meta_data$bacterial_diversity
shapefile_train$fungi_diversity<- train_meta_data$fungi_diversity
shapefile_train$euk_diversity<- train_meta_data$euk_diversity
shapefile_train$pH<- train_meta_data$pH
shapefile_train$Organic_carbon<- train_meta_data$Organic_carbon
plot(shapefile_train)

geom_regions<- st_read("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/Data/regional_info_matched_to_nature_paper_eea_v_3035_1_mio_biogeo-regions_p_2016_v01_r00/BiogeoRegions2016.shp")


#make plots
plot1<- ggplot() + 
  geom_sf(data =geom_regions, aes( colour = short_name)) +
  geom_sf(data = shapefile_train,aes( colour = clean_LU, shape = clean_LU))  + 
  scale_colour_manual(values = c("U111"="gold","U120"="darkgreen","U100"="#7C4700","U400"="#537d90")) +
 theme_classic()
# plot1

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_LU.pdf", width = 8, height = 8)
plot1
dev.off()

plot2<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = as.numeric( P), shape = clean_LU))  + 
   scale_colour_gradient(low="blue", high = "yellow")+
  theme_classic()
# plot2

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_Phosphorus.pdf", width = 8, height = 8)
plot2
dev.off()


plot3<- ggplot() + 
   geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = N, shape = clean_LU))  + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic()
# plot3

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_Nitrogen.pdf", width = 8, height = 8)
plot3
dev.off()


plot4<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = Elevation, shape = clean_LU))  + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic()
# plot4

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_Elevation.pdf", width = 8, height = 8)
plot4
dev.off()


plot5<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = Electrical_conductivity, shape = clean_LU) ) + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic() 
# plot5

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_EC.pdf", width = 8, height = 8)
plot5
dev.off()



plot6<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = bacterial_diversity, shape = clean_LU) ) + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic() 
# plot5

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_bac_div.pdf", width = 8, height = 8)
plot6
dev.off()

plot7<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = fungi_diversity, shape = clean_LU) ) + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic() 
# plot5

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_fung_div.pdf", width = 8, height = 8)
plot7
dev.off()

plot8<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = euk_diversity, shape = clean_LU) ) + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic() 
# plot5

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_euk_div.pdf", width = 8, height = 8)
plot8
dev.off()


plot9<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = pH, shape = clean_LU) ) + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic() 
# plot5

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_pH.pdf", width = 8, height = 8)
plot9
dev.off()


plot10<- ggplot() + geom_sf(data =geom_regions,  colour = "grey") +
  geom_sf(data = shapefile_train,aes( colour = Organic_carbon, shape = clean_LU) ) + 
  scale_colour_gradient(low="blue", high = "yellow")+ theme_classic() 
# plot5

pdf("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/Geo_OC.pdf", width = 8, height = 8)
plot10
dev.off()

diversity<- read.delim("/Users/martina7/Dropbox (The Francis Crick)/AI4LifeScienceHackathon/analysis/meta_data_with_diversity.csv", sep = ",", row.names = 1)


