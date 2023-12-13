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


