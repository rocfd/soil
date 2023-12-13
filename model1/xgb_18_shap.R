### luc18 EXPLAINABILITY

### luc18 ORIGINAL

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
lucas18 <- xgb.load("models/lucas_xgb_18.model")
class(lucas18)


### LOAD test DF
train_df_luc18 <- read.csv("datasets/luc18_train.csv")
glimpse(train_df_luc18)

preds_luc18 <- select(train_df_luc18, -c("LU_label"))
class(preds_luc18)
glimpse(preds_luc18)
colnames(preds_luc18)


# shap --------------------------------------------------------------------

library(shapviz)

params <- list(objective = "multi:softprob", num_class = 3, learning_rate = 0.2)
X_pred <- data.matrix(iris[, -5])
dtrain <- xgboost::xgb.DMatrix(X_pred, label = as.integer(iris[, 5]) - 1)
fit <- xgboost::xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = 100
)

shap_values_18 <- shapviz(lucas18, X_pred = as.matrix(preds_luc18))
names(shap_values_18)

names(shap_values_18) <- c("U100", "U111", "U120", "U400")
shap_values_18

glimpse(shap_values_18)




# U100 --------------------------------------------------------------------
png("shap/lucas18_shap_imp_U100.png", width = 1000, height = 1000, res = 200)
imp18_u100 <- sv_importance(shap_values_18$U100, kind = "bee") +
  labs(title = "LUCAS 2018 - U100")
imp18_u100
dev.off()


shap_inter_18 <- shapviz(lucas18, X_pred = as.matrix(preds_luc18), interactions = T)
names(shap_inter_18) <- c("U100", "U111", "U120", "U400")

png("shap/lucas18_shap_inter_U100.png", width = 1500, height = 1000, res = 200)
int18 <- sv_interaction(shap_inter_18$U100, kind = "bee") +
  labs(title = "LUCAS 2018 - U100")
int18
dev.off()


# U111 --------------------------------------------------------------------
png("shap/lucas18_shap_imp_U111.png", width = 1000, height = 1000, res = 200)
imp18 <- sv_importance(shap_values_18$U111, kind = "bee") +
  labs(title = "LUCAS 2018 - U111")
imp18
dev.off()

png("shap/lucas18_shap_inter_U111.png", width = 1500, height = 1000, res = 200)
int18 <- sv_interaction(shap_inter_18$U111, kind = "bee") +
  labs(title = "LUCAS 2018 - U111")
int18
dev.off()

# U120 --------------------------------------------------------------------
png("shap/lucas18_shap_imp_U120.png", width = 1000, height = 1000, res = 200)
imp18 <- sv_importance(shap_values_18$U120, kind = "bee") +
  labs(title = "LUCAS 2018 - U120")
imp18
dev.off()

png("shap/lucas18_shap_inter_U120.png", width = 1500, height = 1000, res = 200)
int18 <- sv_interaction(shap_inter_18$U120, kind = "bee") +
  labs(title = "LUCAS 2018 - U120")
int18
dev.off()


# U400 --------------------------------------------------------------------
png("shap/lucas18_shap_imp_U400.png", width = 1000, height = 1000, res = 200)
imp18 <- sv_importance(shap_values_18$U400, kind = "bee") +
  labs(title = "LUCAS 2018 - U400")
imp18
dev.off()

png("shap/lucas18_shap_inter_U400.png", width = 1500, height = 1000, res = 200)
int18 <- sv_interaction(shap_inter_18$U400, kind = "bee") +
  labs(title = "LUCAS 2018 - U400")
int18
dev.off()



# BARPLOTS ----------------------------------------------------------------

png("shap/lucas18_shap_imp_U100_bar.png", width = 1000, height = 1000, res = 200)
imp18_u100 <- sv_importance(shap_values_18$U100) +
  labs(title = "LUCAS 2018 - U100")
imp18_u100
dev.off()

png("shap/lucas18_shap_imp_U111_bar.png", width = 1000, height = 1000, res = 200)
imp18_U111 <- sv_importance(shap_values_18$U111) +
  labs(title = "LUCAS 2018 - U111")
imp18_U111
dev.off()

png("shap/lucas18_shap_imp_U120_bar.png", width = 1000, height = 1000, res = 200)
imp18_U120 <- sv_importance(shap_values_18$U120) +
  labs(title = "LUCAS 2018 - U120")
imp18_U120
dev.off()

png("shap/lucas18_shap_imp_U400_bar.png", width = 1000, height = 1000, res = 200)
imp18_U400 <- sv_importance(shap_values_18$U400) +
  labs(title = "LUCAS 2018 - U400")
imp18_U400
dev.off()

