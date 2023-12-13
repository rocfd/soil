### KRUSKALL
library(ggpubr)
library(matrixTests)
library(gt)


# data --------------------------------------------------------------------
glimpse(fn_luc18)
glimpse(fn_luc15)
glimpse(fn_luc09)

table(fn_luc15$LU1_recoded)

# lu as factor ------------------------------------------------------------
fn_luc18$LU1_recoded <- as.factor(fn_luc18$LU1_recoded)
fn_luc15$LU1_recoded <- as.factor(fn_luc15$LU1_recoded)
fn_luc09$LU1_recoded <- as.factor(fn_luc09$LU1_recoded)


# kruskall ----------------------------------------------------------------
krus18 <- as.data.frame(col_kruskalwallis(g = fn_luc18$LU1_recoded, x = num_luc18))
krus15 <- as.data.frame(col_kruskalwallis(g = fn_luc15$LU1_recoded, x = num_luc15))
krus09 <- as.data.frame(col_kruskalwallis(g = fn_luc09$LU1_recoded, x = num_luc09))

krus18 <- cbind(rownames(krus18), krus18)
krus15 <- cbind(rownames(krus15), krus15)
krus09 <- cbind(rownames(krus09), krus09)

colnames(krus18) <- c("Feature",
                      "obs.tot",
                      "obs.groups",
                      "df",
                      "statistic",
                      "pvalue" )
colnames(krus15) <- c("Feature",
                      "obs.tot",
                      "obs.groups",
                      "df",
                      "statistic",
                      "pvalue" )
colnames(krus09) <- c("Feature",
                      "obs.tot",
                      "obs.groups",
                      "df",
                      "statistic",
                      "pvalue" )

# gt tables ---------------------------------------------------------------
krus18_gt <- gt(krus18) |> 
  tab_header("LUCAS2018 - Kruskall Wallis")
krus15_gt <- gt(krus15) |> 
  tab_header("LUCAS2015 - Kruskall Wallis")
krus09_gt <- gt(krus09) |> 
  tab_header("LUCAS2009 - Kruskall Wallis")

export_table(krus18_gt, "lucas18_kruskall")
export_table(krus15_gt, "lucas15_kruskall")
export_table(krus09_gt, "lucas09_kruskall")
