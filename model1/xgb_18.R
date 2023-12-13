### SOIL XGBOOST TIME-MODELS
### ALL OPTS

### LIBS
library(dplyr)
library(xgboost)
library(stringr)
library(caret)


# FEATURE DATASETS --------------------------------------------------------
glimpse(mod_l18)
glimpse(mod_l15)
glimpse(mod_l09)

table(mod_l18$LU_label, useNA ="always")
table(mod_l15$LU_label, useNA ="always")
table(mod_l18$LU_label, useNA ="always")

# mod_l18$LU_recoded <- as.numeric(mod_l18$LU_recoded)
# mod_l15$LU_recoded <- as.numeric(mod_l15$LU_recoded)
# mod_l09$LU_recoded <- as.numeric(mod_l09$LU_recoded)
# 
# table(mod_l18$LU_recoded, useNA = "always")
# 
# mod_l18 <- filter(mod_l18, !is.na(LU_recoded))
# mod_l15 <- filter(mod_l15, !is.na(LU_recoded))
# mod_l09 <- filter(mod_l09, !is.na(LU_recoded))

mat_18 <- select(mod_l18, -c("LU_recoded"))
mat_15 <- select(mod_l15, -c("LU_recoded"))
mat_09 <- select(mod_l09, -c("LU_recoded"))


# Split the data into training and testing sets
set.seed(1234)
trainIndex_18 <- createDataPartition(mat_18$LU_label, p = 0.8, 
                                     list = FALSE, 
                                     times = 1)
train_data_18 <- mat_18[trainIndex_18, ]
test_data_18 <- mat_18[-trainIndex_18, ]

dir.create("datasets")
write.csv(train_data_18, "datasets/luc18_train.csv", quote = T, row.names = F)
write.csv(test_data_18, "datasets/luc18_train.csv", quote = T, row.names = F)

glimpse(train_data_18)

mat_train18_x <- as.matrix(select(train_data_18, -c("LU_label")))
mat_test18_x  <- as.matrix(select(test_data_18, -c("LU_label")))

mat_train18_y <- as.matrix(train_data_18$LU_label)
mat_test18_y  <- as.matrix(test_data_18$LU_label)

class(mat_train18_y) # numeric
class(mat_test18_y) # numeric

any(is.na(mat_train18_y))

# GRID TRAIN - HYPERPARAMETER GRID CREATION ---------------------------------------------------
### create hyperparameter grid
depth_range <- seq(2, 10, 1)
eta_range <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

xgb_l18_hyper_grid <- expand.grid(max_depth = depth_range,
                                         eta = eta_range)

l18_xgb_train_auc <- NULL
l18_xgb_test_auc <- NULL

train_data_18$LU_label <- as.factor(as.character(train_data_18$LU_label))

table(train_data_18$LU_label)


# XGB tunning -------------------------------------------------------------

### COMPUTE TIME
xgb_l18_time_start <- Sys.time()

### PERFORM 5CVGRIDS
for (j in 1:nrow(xgb_l18_hyper_grid)) {
  set.seed(1234)
  l18_m_xgb_untuned <- xgb.cv(
    data = mat_train18_x,
    label = mat_train18_y,
    nrounds = 500,
    objective = "multi:softmax",
    num_class = 4,
    early_stopping_rounds = 3,
    eval_metric = "auc", 
    nfold = 5,
    max_depth = xgb_l18_hyper_grid$max_depth[j],
    eta = xgb_l18_hyper_grid$eta[j],
    lambda = 0,
    alpha = 1,
    maximize = T,
    stratified = T,
    nthread = 3,
    verbose = 1
  )
  
  l18_xgb_train_auc[j] <- l18_m_xgb_untuned$evaluation_log$train_auc_mean[l18_m_xgb_untuned$best_iteration]
  l18_xgb_test_auc[j] <- l18_m_xgb_untuned$evaluation_log$test_auc_mean[l18_m_xgb_untuned$best_iteration]
  
  cat(j, "\n")
}

### RECORD TIME
xgb_l18_time_end <- Sys.time()
xgb_l18_time_taken <- xgb_l18_time_end - xgb_l18_time_start
xgb_l18_time_taken #11.08543 hours #6cores
