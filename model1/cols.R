### SHARED COLUMNS AND MAP FEATURES

### libs
library(dplyr)


# load data ---------------------------------------------------------------
glimpse(meta_luc09)
glimpse(luc15)
glimpse(luc18)


# df comparison -----------------------------------------------------------
luc1815_cols
luc1809_cols
luc1509_cols


# export columns ----------------------------------------------------------
cols18 <- as.data.frame(colnames(luc18))
cols15 <- as.data.frame(colnames(luc15))
cols09 <- as.data.frame(colnames(meta_luc09))

dir.create("cols")
write.csv(cols18, "cols/cols18.csv", row.names = F, quote = F)
write.csv(cols15, "cols/cols15.csv", row.names = F, quote = F)
write.csv(cols09, "cols/cols09.csv", row.names = F, quote = F)

head(meta_luc09$CEC)
head(luc18$EC)


# shared cols -------------------------------------------------------------
sel_luc18 <- select(luc18, c(
  "CaCO3",
  "EC",
  "Elev",
  "K",
  "LC",
  "LU1_recoded",
  "N",
  "NUTS_0",
  "NUTS_1",
  "NUTS_2",
  "NUTS_3",
  "OC",
  "P",
  "pH_CaCl2",
  "pH_H2O",
  "POINTID"
))

sel_luc15 <- select(luc15,c(
  "CaCO3",
  "EC",
  "Elevation",
  "K",
  "LC1",
  "LU1_recoded",
  "N",
  "NUTS_0",
  "NUTS_1",
  "NUTS_2",
  "NUTS_3",
  "OC",
  "P",
  "pH.CaCl2.",
  "pH.H2O.",
  "Point_ID",
))

sel_luc09 <- select(meta_luc09, c(
  "CaCO3",
  "CEC",
  "Elevation.m.",
  "K",
  "LC",
  "LU1_recoded",
  "N",
  "nuts0",
  "nuts1",
  "nuts2",
  "nuts3",
  "OC",
  "P_x",
  "pH_in_CaCl",
  "pH_in_H2O",
  "POINT_ID"
))


# recode columns ----------------------------------------------------------
shared_dict_18 <- list(
  "CaCO3" ="CaCO3",
  "EC" ="EC",
  "Elev" ="Elevation",
  "K" ="K",
  "LC" ="LC",
  "LU1_recoded" ="LU1_recoded",
  "N" ="N",
  "NUTS_0" ="NUTS_0",
  "NUTS_1" ="NUTS_1",
  "NUTS_2" ="NUTS_2",
  "NUTS_3" ="NUTS_3",
  "OC" ="OC",
  "P" ="P",
  "pH_CaCl2" ="pH_CaCl2",
  "pH_H2O" ="pH_H2O",
  "POINTID" ="POINTID"
)


shared_dict_15 <- list(
  "CaCO3" = "CaCO3",
  "EC" = "EC",
  "Elevation" = "Elevation",
  "K" = "K",
  "LC1" = "LC",
  "LU1_recoded" = "LU1_recoded",
  "N" = "N",
  "NUTS_0" = "NUTS_0",
  "NUTS_1" = "NUTS_1",
  "NUTS_2" = "NUTS_2",
  "NUTS_3" = "NUTS_3",
  "OC" = "OC",
  "P" = "P",
  "pH.CaCl2." = "pH_CaCl2",
  "pH.H2O." = "pH_H2O",
  "Point_ID" = "POINTID"
)

shared_dict_09 <- list(
  "CaCO3" = "CaCO3",
  "CEC" = "EC",
  "Elevation.m." = "Elevation",
  "K" = "K",
  "LC" = "LC",
  "LU1_recoded" = "LU1_recoded",
  "N" = "N",
  "nuts0" = "NUTS_0",
  "nuts1" = "NUTS_1",
  "nuts2" = "NUTS_2",
  "nuts3" = "NUTS_3",
  "OC" = "OC",
  "P_x" = "P",
  "pH_in_CaCl" = "pH_CaCl2",
  "pH_in_H2O" = "pH_H2O",
  "POINT_ID" =  "POINTID"
)


# map shared cols ---------------------------------------------------------

# Use purrr::set_names to apply the mapping to column names
sha_luc18 <- sel_luc18 %>%
  set_names(map_chr(names(.), ~ ifelse(.x %in% names(shared_dict_18), shared_dict_18[[.x]], as.character(.x))))

sha_luc15 <- sel_luc15 %>%
  set_names(map_chr(names(.), ~ ifelse(.x %in% names(shared_dict_15), shared_dict_15[[.x]], as.character(.x))))

sha_luc09 <- sel_luc09 %>%
  set_names(map_chr(names(.), ~ ifelse(.x %in% names(shared_dict_09), shared_dict_09[[.x]], as.character(.x))))



# filter LU_recoded na ----------------------------------------------------
fn_luc15 <- filter(sha_luc15, !is.na(LU1_recoded))
fn_luc09 <- filter(sha_luc09, !is.na(LU1_recoded))

# export 15 and 09 --------------------------------------------------------
write.csv(fn_luc15, "z_datasets/luc15_fn.csv", quote = F, row.names = F)
write.csv(fn_luc09, "z_datasets/luc09_fn.csv", quote = F, row.names = F)
