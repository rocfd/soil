# GRID SEARCH PERFORMANCE -------------------------------------------------
xgb_15_train_auc <- l15_m_xgb_untuned$evaluation_log[l15_m_xgb_untuned$best_iteration]$train_auc_mean
xgb_15_train_auc # 0.9668994
xgb_15_test_auc <- l15_m_xgb_untuned$evaluation_log[l15_m_xgb_untuned$best_iteration]$test_auc_mean
xgb_15_test_auc # 0.9584834
xgb_15_best_iteration <- l15_m_xgb_untuned$best_iteration
xgb_15_best_iteration #13

### find ideal hyperparamters
xgb_l15_hyper_grid[which.max(xgb_15_test_auc), ]

# GRID SEARCH SELECTED PARAMETERS -------------------------------------
### eta_p_xgb_15
eta_p_xgb_15 = xgb_l15_hyper_grid[which.max(xgb_15_test_auc), ]$eta
eta_p_xgb_15

### max_depth_p_xgb_15
max_depth_p_xgb_15 = xgb_l15_hyper_grid[which.max(xgb_15_test_auc), ]$max_depth
max_depth_p_xgb_15

### XGB MATRICES
xgb_15_train = xgb.DMatrix(data = mat_train15_x, label = mat_train15_y)
xgb_15_test = xgb.DMatrix(data = mat_test15_x, label = mat_test15_y)

# DEFINE WATCHLIST --------------------------------------------------------
xgb_15_watchlist = list(train=xgb_15_train, test=xgb_15_test)

# FIT XGBOOST MODEL (XGB.TRAIN FUNCTION) ------------------------------
xgb_15_train_model_time_start <- Sys.time()
xgb_15_train_model = xgb.train(data = xgb_15_train, 
                               max.depth = max_depth_p_xgb_15,
                               eta = eta_p_xgb_15,
                               #scale_pos_weight = scale_pos_weight_p_xgb_15,
                               #max_delta_step = max_delta_step_p_xgb_15,
                               #label = label = m_all_mod_rand[, 1],
                               watchlist=xgb_15_watchlist, 
                               nrounds = 500,
                               early_stopping_rounds = 5, 
                               objective = "multi:softmax",
                               eval_metric = "auc",
                               num_class = 4,
                               #showsd = TRUE,
                               lambda = 0,
                               alpha = 1,
                               prediction = TRUE,
                               maximize = TRUE,
                               stratified = TRUE,
                               nthread = 1,
                               verbose = 1)

### RECORD TIME - XGBTRAIN
xgb_15_train_model_time_end <- Sys.time()
xgb_15_train_model_time_taken <- xgb_15_train_model_time_end - xgb_15_train_model_time_start
xgb_15_train_model_time_taken #0.1111641 secs #4cores

xgb_15_train_model_auc <- str_split(xgb_15_train_model$best_msg[1], ":")[[1]][2]
xgb_15_train_model_auc
xgb_15_train_model_auc <- str_sub(xgb_15_train_model_auc, start = 1L, end = -12L)
xgb_15_train_model_auc # 1.000000
xgb_15_test_model_auc <- str_split(xgb_15_train_model$best_msg[1], ":")[[1]][3]
xgb_15_test_model_auc #0.563882

# lucas MODEL TT (XGBOOST FUNCTION) -----------------------------------
xgb_15_xgboost_model_time_start <- Sys.time()
xgb_15_xgboost_model <- xgboost(data = xgb_15_train, 
                                max.depth = max_depth_p_xgb_15, 
                                eta = eta_p_xgb_15,
                                # scale_pos_weight = scale_pos_weight_p_xgb_15,
                                # max_delta_step = max_delta_step_p_xgb_15,
                                nrounds = xgb_15_train_model$best_iteration, 
                                num_class = 4,           
                                early_stopping_rounds = 5, 
                                objective = "multi:softmax",
                                #watchlist=xgb_15_watchlist,
                                eval_metric = "auc",
                                # metrics = c("auc", "auc"),
                                # showsd = TRUE,
                                lambda = 0,
                                alpha = 1,
                                prediction = TRUE,
                                maximize = TRUE,
                                stratified = TRUE,
                                nthread = 1,
                                verbose = 1)
#save_name = "lucas_spw_train.model")

### RECORD TIME - XGBOOST FUNCTION
xgb_15_xgboost_model_time_end <- Sys.time()
xgb_15_xgboost_model_time_taken <- xgb_15_xgboost_model_time_end - xgb_15_xgboost_model_time_start
xgb_15_xgboost_model_time_taken #0.09779692 secs #4coreså

xgb_15_xgboost_model_auc <- str_split(xgb_15_xgboost_model$best_msg[1], ":")[[1]][2]
#xgb_15_xgboost_model_auc
#xgb_15_xgboost_model_auc <- str_sub(xgb_15_xgboost_model_auc, start = 1L, end = -12L)
xgb_15_xgboost_model_auc # 1.000000

# xgb_15_xgboost_model_auc <- str_split(xgb_15_xgboost_model$best_msg[1], ":")[[1]][3]
# xgb_15_xgboost_model_auc #0.562607


# EXPORT MODELS -----------------------------------------------------------
# SAVE MODEL --------------------------------------------------------------
xgb.save(xgb_15_xgboost_model, fname = "models/lucas_xgb_15.model")

sink(file = "models/lucas_xgb_15_raw.json")
xgb_15_xgboost_model_raw <- xgb.save.raw(xgb_15_xgboost_model, raw_format = "ubj")
xgb_15_xgboost_model_raw
sink()

# sink(file = "xgb_ens_models/lucas_xgb_15_raw.json")
# xgb_15_xgboost_model_raw <- xgb.save.raw(xgb_15_xgboost_model, raw_format = "json")
# xgb_15_xgboost_model_raw
# sink()
