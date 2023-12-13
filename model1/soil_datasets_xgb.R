### PREDIG XGBOOST ENSEMBLE
### TRAIN TEST ENSEMBLE SCALED

### LIBS
library(dplyr)
library(xgboost)
library(stringr)


# LOAD DATASETS -----------------------------------------------------------
l18 <- read.csv("../20231122_eda/z_datasets/luc18_fn.csv")
l15 <- read.csv("../20231122_eda/z_datasets/luc15_fn.csv")
l09 <- read.csv("../20231122_eda/z_datasets/luc09_fn.csv")
glimpse(l18)
glimpse(l15)
glimpse(l09)

table(l18$LU1_recoded, useNA = "always")
table(l15$LU1_recoded)
table(l09$LU1_recoded)

# lu as factor ------------------------------------------------------------
l18$LU1_recoded <- as.factor(l18$LU1_recoded)
l15$LU1_recoded <- as.factor(l15$LU1_recoded)
l09$LU1_recoded <- as.factor(l09$LU1_recoded)

table(l18$LU1_recoded)
table(l15$LU1_recoded)
table(l09$LU1_recoded)


# SELECT NUMERIC ----------------------------------------------------------
# recode P ----------------------------------------------------------------
l18$P <- replace(l18$P, l18$P == "< LOD", NA)
l18$P <- replace(l18$P, l18$P == "<0.0", NA)

l15$P <- replace(l15$P, l15$P == "< LOD", NA)
l15$P <- replace(l15$P, l15$P == "<0.0", NA)

l09$P <- replace(l09$P, l09$P == "< LOD", NA)
l09$P <- replace(l09$P, l09$P == "<0.0", NA)

table(l18$LU1_recoded)
table(l15$LU1_recoded)
table(l09$LU1_recoded)

# numerize characters -----------------------------------------------------
table(is.na(l18$CaCO3))
table(is.na(l18$K))
table(is.na(l18$N))
table(is.na(l18$OC))
table(is.na(l18$P))

l18$CaCO3 <- as.numeric(l18$CaCO3)
l18$K <- as.numeric(l18$K)
l18$N <- as.numeric(l18$N)
l18$OC <- as.numeric(l18$OC)
l18$P <- as.numeric(l18$P)


l15$CaCO3 <- as.numeric(l15$CaCO3)
l15$K <- as.numeric(l15$K)
l15$N <- as.numeric(l15$N)
l15$OC <- as.numeric(l15$OC)
l15$P <- as.numeric(l15$P)

l09$CaCO3 <- as.numeric(l09$CaCO3)
l09$K <- as.numeric(l09$K)
l09$N <- as.numeric(l09$N)
l09$OC <- as.numeric(l09$OC)
l09$P <- as.numeric(l09$P)

glimpse(l18)
# select numerical --------------------------------------------------------
num_l18 <- l18 %>% select_if(is.numeric)
num_l15 <- l15 %>% select_if(is.numeric)
num_l09 <- l09 %>% select_if(is.numeric)


# model features (num + label) --------------------------------------------
mod_l18 <- cbind(l18$LU1_recoded, num_l18)
mod_l15 <- cbind(l15$LU1_recoded, num_l15)
mod_l09 <- cbind(l09$LU1_recoded, num_l09)

mod_l18 <- rename(mod_l18, LU_recoded = "l18$LU1_recoded")
mod_l15 <- rename(mod_l15, LU_recoded = "l15$LU1_recoded")
mod_l09 <- rename(mod_l09, LU_recoded = "l09$LU1_recoded")


# discard ids -------------------------------------------------------------
mod_l18 <- select(mod_l18, -c("POINTID"))
mod_l15 <- select(mod_l15, -c("POINTID"))
mod_l09 <- select(mod_l09, -c("POINTID"))

table(is.na(mod_l18$LU_recoded))

# remove LU NAs -----------------------------------------------------------
mod_l18 <- filter(mod_l18, !is.na(LU_recoded))
mod_l15 <- filter(mod_l15, !is.na(LU_recoded))
mod_l09 <- filter(mod_l09, !is.na(LU_recoded))
table(is.na(mod_l18$LU_recoded))

dict_18 <- list(
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

glimpse(mod_l18)
glimpse(mod_l15)
glimpse(mod_l09)

# LABEL ---------------------------------------------------------

mod_l18$LU_label <- mod_l18$LU_recoded
mod_l15$LU_label <- mod_l15$LU_recoded
mod_l09$LU_label <- mod_l09$LU_recoded

table(mod_l18$LU_recoded, useNA = "always")

mod_l18$LU_label <- recode(mod_l18$LU_label,
"U100"  = 0,
"U111" = 1,
"U120" = 2,
"U400" = 3)

mod_l15$LU_label <- recode(mod_l15$LU_label,
                           "U100"  = 0,
                           "U111" = 1,
                           "U120" = 2,
                           "U400" = 3)

mod_l09$LU_label <- recode(mod_l09$LU_label,
                           "U100"  = 0,
                           "U111" = 1,
                           "U120" = 2,
                           "U400" = 3)
table(mod_l09$LU_label)
glimpse(mod_l09)

# export model datasets ---------------------------------------------------
write.csv(mod_l18, "feats/lucas18_model_features.csv", quote = T, row.names = F)
write.csv(mod_l15, "feats/lucas15_model_features.csv", quote = T, row.names = F)
write.csv(mod_l09, "feats/lucas09_model_features.csv", quote = T, row.names = F)

