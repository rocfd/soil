### SOIL EDA


# LIBRARIES ---------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(gt)
library(gtExtras)
library(esquisse)
library(patchwork)
library(ggiraph)
library(webshot2)
library(GGally)
library(kableExtra)


# LOAD DATA ---------------------------------------------------------------
luc18 <- read.csv("datasets/luc18.csv")
org_luc18 <- read.csv("datasets/luc18_org.csv")
ero_luc18 <- read.csv("datasets/luc18_erosion.csv")
dens_luc18 <- read.csv("datasets/luc18_density.csv")

luc15 <- read.csv("datasets/luc15.csv")
luc09 <- read.table("datasets/luc09.csv", sep = "|", header = T)
View(luc09)

meta_luc09 <- read.table("datasets/luc09_meta.csv", sep = "|", header = T)
View(meta_luc09)

luc_all <- read.delim("datasets/luc_all_metadata.txt", sep = "\t", header = T)
glimpse(luc_all)

glimpse(luc18)

# FEATURE EXPLORATION -----------------------------------------------------

colnames(luc18)
colnames(org_luc18)
colnames(ero_luc18)
colnames(dens_luc18)
colnames(luc15)
colnames(luc09)

dir.create("tables")
luc1815_cols <- compare_dataframes(luc18, luc15)
luc1815_gt <- create_col_table(luc1815_cols, "luc18", "luc15")
export_table(luc1815_gt, "tables/luc1815_cols")

luc1809_cols <- compare_dataframes(luc18, luc09)
luc1809_gt <- create_col_table(luc1809_cols, "luc18", "luc09")
export_table(luc1809_gt, "tables/luc1809_cols")

luc1509_cols <- compare_dataframes(luc15, luc09)
luc1509_gt <- create_col_table(luc1509_cols, "luc15", "luc09")
export_table(luc1509_gt, "tables/luc1509_cols")

luc18all_cols <- compare_dataframes(luc18, luc_all)
luc18all_gt <- create_col_table(luc18all_cols, "luc18", "luc_all")
export_table(luc18all_gt, "tables/luc18all_cols")


# SUMMARY PLOTS -----------------------------------------------------------

plt18 <- gtExtras::gt_plt_summary(luc18)
plt18

plt18_org <- gtExtras::gt_plt_summary(org_luc18)
plt18_org

plt18_ero <- gtExtras::gt_plt_summary(ero_luc18)
plt18_ero

plt18_ero <- gtExtras::gt_plt_summary(ero_luc18)
plt18_ero

plt15 <- gtExtras::gt_plt_summary(luc15)
plt15

plt09 <- gtExtras::gt_plt_summary(luc09)
plt09
class(plt09)

pltall <- gtExtras::gt_plt_summary(luc_all)


gtsave(data = plt18, filename = "z_plots/sum_luc18.html")
gtsave(data = plt15, filename = "z_plots/sum_luc15.html")
gtsave(data = plt09, filename = "z_plots/sum_luc09.html")
gtsave_extra(data = pltall, filename = "plots/sum_luc_all.png")


# factor analysis ---------------------------------------------------------

luc_all_columns_df <- extract_factors(luc_all)
luc_all_dict <- create_factor_dictionary(luc_all_columns_df)
luc_all_df <- factor_dict_to_dataframe(luc_all_dict)
View(luc_all_df)
print_gt_table(luc_all_df)

glimpse(luc_all)



# correlation analysis ----------------------------------------------------

library(GGally)

luc18_cor_matrix <- cor(luc_all_num, method = "spearman")

luc_all_num <- luc_all %>% select_if(is.numeric)
glimpse(luc_all_num)

pdf(bg = "white", width = 20, height = 10, "luc_num_correlations.pdf")
ggpairs(
  luc_all_num,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2018 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()
