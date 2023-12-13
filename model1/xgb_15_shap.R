### luc15 EXPLAINABILITY

### luc15 ORIGINAL

### XGBOOST FEAT CONTRIBUTION X BIO

### LIBRARIES
library(dplyr)
library(ggplot2)
library(esquisse)
library(ggthemes)
library(xgboost)
#library(SHAPforxgboost)
library(shapviz)
library(gapminder)

### LOAD MODEL
lucas15 <- xgb.load("models/lucas_xgb_15.model")
class(lucas15)


### LOAD test DF
train_df_luc15 <- read.csv("datasets/luc15_train.csv")
glimpse(train_df_luc15)

preds_luc15 <- select(train_df_luc15, -c("LU_label"))
class(preds_luc15)
glimpse(preds_luc15)
colnames(preds_luc15)


# shap --------------------------------------------------------------------
library(shapviz)
shap_values_15 <- shapviz(lucas15, X_pred = as.matrix(preds_luc15))
names(shap_values_15) <- c("U100", "U111", "U120", "U400")

# U100 --------------------------------------------------------------------
png("shap/luc15/lucas15_shap_imp_U100.png", width = 1000, height = 1000, res = 200)
imp15_u100 <- sv_importance(shap_values_15$U100, kind = "bee") +
  labs(title = "LUCAS 2015 - U100")
imp15_u100
dev.off()


shap_inter_15 <- shapviz(lucas15, X_pred = as.matrix(preds_luc15), interactions = T)
names(shap_inter_15) <- c("U100", "U111", "U120", "U400")

png("shap/luc15/lucas15_shap_inter_U100.png", width = 1500, height = 1000, res = 200)
int15 <- sv_interaction(shap_inter_15$U100, kind = "bee") +
  labs(title = "LUCAS 2015 - U100")
int15
dev.off()


# U111 --------------------------------------------------------------------
png("shap/luc15/lucas15_shap_imp_U111.png", width = 1000, height = 1000, res = 200)
imp15 <- sv_importance(shap_values_15$U111, kind = "bee") +
  labs(title = "LUCAS 2015 - U111")
imp15
dev.off()

png("shap/luc15/lucas15_shap_inter_U111.png", width = 1500, height = 1000, res = 200)
int15 <- sv_interaction(shap_inter_15$U111, kind = "bee") +
  labs(title = "LUCAS 2015 - U111")
int15
dev.off()

# U120 --------------------------------------------------------------------
png("shap/luc15/lucas15_shap_imp_U120.png", width = 1000, height = 1000, res = 200)
imp15 <- sv_importance(shap_values_15$U120, kind = "bee") +
  labs(title = "LUCAS 2015 - U120")
imp15
dev.off()

png("shap/luc15/lucas15_shap_inter_U120.png", width = 1500, height = 1000, res = 200)
int15 <- sv_interaction(shap_inter_15$U120, kind = "bee") +
  labs(title = "LUCAS 2015 - U120")
int15
dev.off()


# U400 --------------------------------------------------------------------
png("shap/luc15/lucas15_shap_imp_U400.png", width = 1000, height = 1000, res = 200)
imp15 <- sv_importance(shap_values_15$U400, kind = "bee") +
  labs(title = "LUCAS 2015 - U400")
imp15
dev.off()

png("shap/luc15/lucas15_shap_inter_U400.png", width = 1500, height = 1000, res = 200)
int15 <- sv_interaction(shap_inter_15$U400, kind = "bee") +
  labs(title = "LUCAS 2015 - U400")
int15
dev.off()



# BARPLOTS ----------------------------------------------------------------

png("shap/luc15/lucas15_shap_imp_U100_bar.png", width = 1000, height = 1000, res = 200)
imp15_u100 <- sv_importance(shap_values_15$U100) +
  labs(title = "LUCAS 2015 - U100")
imp15_u100
dev.off()

png("shap/luc15/lucas15_shap_imp_U111_bar.png", width = 1000, height = 1000, res = 200)
imp15_U111 <- sv_importance(shap_values_15$U111) +
  labs(title = "LUCAS 2015 - U111")
imp15_U111
dev.off()

png("shap/luc15/lucas15_shap_imp_U120_bar.png", width = 1000, height = 1000, res = 200)
imp15_U120 <- sv_importance(shap_values_15$U120) +
  labs(title = "LUCAS 2015 - U120")
imp15_U120
dev.off()

png("shap/luc15/lucas15_shap_imp_U400_bar.png", width = 1000, height = 1000, res = 200)
imp15_U400 <- sv_importance(shap_values_15$U400) +
  labs(title = "LUCAS 2015 - U400")
imp15_U400
dev.off()

