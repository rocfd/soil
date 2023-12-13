### luc09 EXPLAINABILITY

### luc09 ORIGINAL

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
lucas09 <- xgb.load("models/lucas_xgb_09.model")
class(lucas09)


### LOAD test DF
train_df_luc09 <- read.csv("datasets/luc09_train.csv")
glimpse(train_df_luc09)

preds_luc09 <- select(train_df_luc09, -c("LU_label"))
class(preds_luc09)
glimpse(preds_luc09)
colnames(preds_luc09)


# shap --------------------------------------------------------------------
library(shapviz)
shap_values_09 <- shapviz(lucas09, X_pred = as.matrix(preds_luc09))
names(shap_values_09) <- c("U100", "U111", "U120", "U400")

# U100 --------------------------------------------------------------------
png("shap/luc09/lucas09_shap_imp_U100.png", width = 1000, height = 1000, res = 200)
imp09_u100 <- sv_importance(shap_values_09$U100, kind = "bee") +
  labs(title = "LUCAS 2009 - U100")
imp09_u100
dev.off()


shap_inter_09 <- shapviz(lucas09, X_pred = as.matrix(preds_luc09), interactions = T)
names(shap_inter_09) <- c("U100", "U111", "U120", "U400")

png("shap/luc09/lucas09_shap_inter_U100.png", width = 0900, height = 1000, res = 200)
int09 <- sv_interaction(shap_inter_09$U100, kind = "bee") +
  labs(title = "LUCAS 2009 - U100")
int09
dev.off()


# U111 --------------------------------------------------------------------
png("shap/luc09/lucas09_shap_imp_U111.png", width = 1000, height = 1000, res = 200)
imp09 <- sv_importance(shap_values_09$U111, kind = "bee") +
  labs(title = "LUCAS 2009 - U111")
imp09
dev.off()

png("shap/luc09/lucas09_shap_inter_U111.png", width = 0900, height = 1000, res = 200)
int09 <- sv_interaction(shap_inter_09$U111, kind = "bee") +
  labs(title = "LUCAS 2009 - U111")
int09
dev.off()

# U120 --------------------------------------------------------------------
png("shap/luc09/lucas09_shap_imp_U120.png", width = 1000, height = 1000, res = 200)
imp09 <- sv_importance(shap_values_09$U120, kind = "bee") +
  labs(title = "LUCAS 2009 - U120")
imp09
dev.off()

png("shap/luc09/lucas09_shap_inter_U120.png", width = 0900, height = 1000, res = 200)
int09 <- sv_interaction(shap_inter_09$U120, kind = "bee") +
  labs(title = "LUCAS 2009 - U120")
int09
dev.off()


# U400 --------------------------------------------------------------------
png("shap/luc09/lucas09_shap_imp_U400.png", width = 1000, height = 1000, res = 200)
imp09 <- sv_importance(shap_values_09$U400, kind = "bee") +
  labs(title = "LUCAS 2009 - U400")
imp09
dev.off()

png("shap/luc09/lucas09_shap_inter_U400.png", width = 0900, height = 1000, res = 200)
int09 <- sv_interaction(shap_inter_09$U400, kind = "bee") +
  labs(title = "LUCAS 2009 - U400")
int09
dev.off()



# BARPLOTS ----------------------------------------------------------------

png("shap/luc09/lucas09_shap_imp_U100_bar.png", width = 1000, height = 1000, res = 200)
imp09_u100 <- sv_importance(shap_values_09$U100) +
  labs(title = "LUCAS 2009 - U100")
imp09_u100
dev.off()

png("shap/luc09/lucas09_shap_imp_U111_bar.png", width = 1000, height = 1000, res = 200)
imp09_U111 <- sv_importance(shap_values_09$U111) +
  labs(title = "LUCAS 2009 - U111")
imp09_U111
dev.off()

png("shap/luc09/lucas09_shap_imp_U120_bar.png", width = 1000, height = 1000, res = 200)
imp09_U120 <- sv_importance(shap_values_09$U120) +
  labs(title = "LUCAS 2009 - U120")
imp09_U120
dev.off()

png("shap/luc09/lucas09_shap_imp_U400_bar.png", width = 1000, height = 1000, res = 200)
imp09_U400 <- sv_importance(shap_values_09$U400) +
  labs(title = "LUCAS 2009 - U400")
imp09_U400
dev.off()

