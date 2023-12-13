### CORRELATIONS
library(GGally)

# data --------------------------------------------------------------------
glimpse(fn_luc18)
glimpse(fn_luc15)
glimpse(fn_luc09)

table(fn_luc15$LU1_recoded)

# pointid as factor -------------------------------------------------------
fn_luc18$POINTID <- as.factor(as.character(fn_luc18$POINTID))
fn_luc15$POINTID <- as.factor(as.character(fn_luc15$POINTID))
fn_luc09$POINTID <- as.factor(as.character(fn_luc09$POINTID))


# recode P ----------------------------------------------------------------
fn_luc18$P <- replace(fn_luc18$P, fn_luc18$P == "< LOD", NA)
fn_luc18$P <- replace(fn_luc18$P, fn_luc18$P == "<0.0", NA)
table(fn_luc18$P, useNA = "always")



# numerize characters -----------------------------------------------------
fn_luc18$CaCO3 <- as.numeric(fn_luc18$CaCO3)
fn_luc18$K <- as.numeric(fn_luc18$K)
fn_luc18$N <- as.numeric(fn_luc18$N)
fn_luc18$P <- as.numeric(fn_luc18$P)
fn_luc18$OC <- as.numeric(fn_luc18$OC)

fn_luc15$CaCO3 <- as.numeric(fn_luc15$CaCO3)
fn_luc15$K <- as.numeric(fn_luc15$K)
fn_luc15$N <- as.numeric(fn_luc15$N)
fn_luc15$OC <- as.numeric(fn_luc15$OC)

fn_luc09$CaCO3 <- as.numeric(fn_luc09$CaCO3)
fn_luc09$K <- as.numeric(fn_luc09$K)
fn_luc09$N <- as.numeric(fn_luc09$N)
fn_luc09$OC <- as.numeric(fn_luc09$OC)

# select numerical --------------------------------------------------------
num_luc18 <- fn_luc18 %>% select_if(is.numeric)
num_luc15 <- fn_luc15 %>% select_if(is.numeric)
num_luc09 <- fn_luc09 %>% select_if(is.numeric)

# correlation analysis ----------------------------------------------------
pdf(bg = "white", width = 20, height = 10, "z_plots/luc18_correlation.pdf")
ggpairs(
  num_luc18,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2018 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()

pdf(bg = "white", width = 20, height = 10, "z_plots/luc15_correlation.pdf")
ggpairs(
  num_luc15,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2015 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()

pdf(bg = "white", width = 20, height = 10, "z_plots/luc09_correlation.pdf")
ggpairs(
  num_luc09,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2009 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()


# correlation analysis ----------------------------------------------------
png(bg = "white", width = 2000, height = 1000, res = 100, "z_plots/luc18_correlation.png")
ggpairs(
  num_luc18,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2018 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()

png(bg = "white", width = 2000, height = 1000, res = 100, "z_plots/luc15_correlation.png")
ggpairs(
  num_luc15,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2015 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()

png(bg = "white", width = 2000, height = 1000, res = 100, "z_plots/luc09_correlation.png")
ggpairs(
  num_luc09,
  upper = list(continuous = wrap("cor", method = "spearman", size = 2.5, hjust=0.7)),
  lower = list(continuous = wrap("points", alpha = 0.3,    size=0.1), 
               combo = wrap("dot", alpha = 0.4,            size=0.2) ),
  title = "LUCAS2009 (Spearman Corr: *** p-value < 0.001)"
)
dev.off()
