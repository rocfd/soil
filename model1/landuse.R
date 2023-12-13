### LAND USE CATEGORIES OVER TIME

### LIBS
library(dplyr)
library(ggplot2)
library(esquisse)
library(purrr)


# LOAD DATA ---------------------------------------------------------------
colnames(luc18)
colnames(luc15)
colnames(luc09)


# CATEGORIES METADATA -----------------------------------------------------
U100 <- "U100"
U111 <- "U111"
U120 <- "U120"
U400 <- "U400"


# CAT 2009 ----------------------------------------------------------------
table(meta_luc09$LU1)
lu1_luc09 <- as.data.frame(table(meta_luc09$LU1))
colnames(lu1_luc09) <- c("LU1", "Frequency")
write.csv(lu1_luc09, "landuse/luc09_lu1.csv", quote = F, row.names = F)

# Define the mapping dictionary
mapping_dict_09 <- list(
  "MISSING" = NA,
  "U113" = NA,
  "U130" = NA,
  "U140" = NA,
  "U150" = NA,
  "U111" = "U111",
  "U112" = "U100",
  "U120" = "U120",
  "U210" = NA, 
  "U221" = NA, 
  "U226" = NA,
  "U311" = NA, 
  "U312" = NA, 
  "U313" = NA, 
  "U315" = NA, 
  "U316" = NA, 
  "U317" = NA, 
  "U318" = NA, 
  "U321" = NA, 
  "U322" = NA, 
  "U330" = NA, 
  "U340" = NA, 
  "U350" = NA, 
  "U361" = NA, 
  "U362" = NA, 
  "U363" = NA, 
  "U364" = NA, 
  "U370" = NA,
  "U400" = "U400", 
  "U410" = "U400",
  "U420" = "U400"
)


meta_luc09 <- meta_luc09 %>%
  mutate(LU1_recoded = map_chr(LU1, ~ ifelse(.x %in% names(mapping_dict_09), mapping_dict_09[[.x]], as.character(.x))))

table(meta_luc09$LU1_recoded)


# LUCAS 15 ----------------------------------------------------------------

table(luc15$LU1)

# Define the mapping dictionary
mapping_dict_15 <- list(
  "MISSING" = NA,
  "U113" = NA,
  "U130" = NA,
  "U140" = NA,
  "U150" = NA,
  "U111" = "U111",
  "U112" = "U100",
  "U120" = "U120",
  "U210" = NA, 
  "U221" = NA, 
  "U226" = NA,
  "U311" = NA, 
  "U312" = NA, 
  "U313" = NA, 
  "U315" = NA, 
  "U316" = NA, 
  "U317" = NA, 
  "U318" = NA, 
  "U319" = NA, 
  "U321" = NA, 
  "U322" = NA, 
  "U330" = NA, 
  "U340" = NA, 
  "U341" = NA, 
  "U342" = NA, 
  "U350" = NA, 
  "U361" = NA, 
  "U362" = NA, 
  "U363" = NA, 
  "U364" = NA, 
  "U370" = NA,
  "U400" = "U400", 
  "U410" = "U400",
  "U411" = NA,
  "U413" = NA,
  "U414" = NA,
  "U415" = "U400",
  "U420" = "U400"
)

luc15 <- luc15 %>%
  mutate(LU1_recoded = map_chr(LU1, ~ ifelse(.x %in% names(mapping_dict_15), mapping_dict_15[[.x]], as.character(.x))))

table(luc18$LU1_recoded)

# LUCAS18 -----------------------------------------------------------------

table(luc18$LU)

# Define the mapping dictionary
mapping_dict_18 <- list(
  "MISSING" = NA,
  "U113" = NA,
  "U130" = NA,
  "U140" = NA,
  "U150" = NA,
  "U111" = "U111",
  "U112" = "U100",
  "U120" = "U120",
  "U210" = NA, 
  "U221" = NA, 
  "U226" = NA,
  "U311" = NA, 
  "U312" = NA, 
  "U313" = NA, 
  "U315" = NA, 
  "U316" = NA, 
  "U317" = NA, 
  "U318" = NA, 
  "U319" = NA, 
  "U321" = NA, 
  "U322" = NA, 
  "U330" = NA, 
  "U340" = NA, 
  "U341" = NA, 
  "U342" = NA, 
  "U350" = NA, 
  "U361" = NA, 
  "U362" = NA, 
  "U363" = NA, 
  "U364" = NA, 
  "U370" = NA,
  "U400" = "U400", 
  "U411" = NA,
  "U413" = NA,
  "U414" = NA,
  "U415" = "U400",
  "U420" = "U400"
)

luc18 <- luc18 %>%
  mutate(LU1_recoded = map_chr(LU, ~ ifelse(.x %in% names(mapping_dict_18), mapping_dict_18[[.x]], as.character(.x))))

table(luc18$LU1_recoded)


# EXPORT RECODED ----------------------------------------------------------
dir.create("datasets/lu")
write.csv(meta_luc09, "z_datasets/lu/luc09_lu.csv", row.names = F, quote = F)
write.csv(luc15, "z_datasets/lu/luc15_lu.csv", row.names = F, quote = F)
write.csv(luc18, "z_datasets/lu/luc18_lu.csv", row.names = F, quote = F)

table(luc15$LU1_recoded)
table(luc18$LU1_recoded)
table(meta_luc09$LU1_recoded)
