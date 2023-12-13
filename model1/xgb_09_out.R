# GRID SEARCH PERFORMANCE -------------------------------------------------
xgb_09_train_auc <- l09_m_xgb_untuned$evaluation_log[l09_m_xgb_untuned$best_iteration]$train_auc_mean
xgb_09_train_auc # 0.9668994
xgb_09_test_auc <- l09_m_xgb_untuned$evaluation_log[l09_m_xgb_untuned$best_iteration]$test_auc_mean
xgb_09_test_auc # 0.9584834
xgb_09_best_iteration <- l09_m_xgb_untuned$best_iteration
xgb_09_best_iteration #13

### find ideal hyperparamters
xgb_l09_hyper_grid[which.max(xgb_09_test_auc), ]

# GRID SEARCH SELECTED PARAMETERS -------------------------------------
### eta_p_xgb_09
eta_p_xgb_09 = xgb_l09_hyper_grid[which.max(xgb_09_test_auc), ]$eta
eta_p_xgb_09

### max_depth_p_xgb_09
max_depth_p_xgb_09 = xgb_l09_hyper_grid[which.max(xgb_09_test_auc), ]$max_depth
max_depth_p_xgb_09

### XGB MATRICES
xgb_09_train = xgb.DMatrix(data = mat_train09_x, label = mat_train09_y)
xgb_09_test = xgb.DMatrix(data = mat_test09_x, label = mat_test09_y)

# DEFINE WATCHLIST --------------------------------------------------------
xgb_09_watchlist = list(train=xgb_09_train, test=xgb_09_test)

# FIT XGBOOST MODEL (XGB.TRAIN FUNCTION) ------------------------------
xgb_09_train_model_time_start <- Sys.time()
xgb_09_train_model = xgb.train(data = xgb_09_train, 
                               max.depth = max_depth_p_xgb_09,
                               eta = eta_p_xgb_09,
                               #scale_pos_weight = scale_pos_weight_p_xgb_09,
                               #max_delta_step = max_delta_step_p_xgb_09,
                               #label = label = m_all_mod_rand[, 1],
                               watchlist=xgb_09_watchlist, 
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
xgb_09_train_model_time_end <- Sys.time()
xgb_09_train_model_time_taken <- xgb_09_train_model_time_end - xgb_09_train_model_time_start
xgb_09_train_model_time_taken #0.1111641 secs #4cores

xgb_09_train_model_auc <- str_split(xgb_09_train_model$best_msg[1], ":")[[1]][2]
xgb_09_train_model_auc
xgb_09_train_model_auc <- str_sub(xgb_09_train_model_auc, start = 1L, end = -12L)
xgb_09_train_model_auc # 1.000000
xgb_09_test_model_auc <- str_split(xgb_09_train_model$best_msg[1], ":")[[1]][3]
xgb_09_test_model_auc #0.563882

# lucas MODEL TT (XGBOOST FUNCTION) -----------------------------------
xgb_09_xgboost_model_time_start <- Sys.time()
xgb_09_xgboost_model <- xgboost(data = xgb_09_train, 
                                max.depth = max_depth_p_xgb_09, 
                                eta = eta_p_xgb_09,
                                # scale_pos_weight = scale_pos_weight_p_xgb_09,
                                # max_delta_step = max_delta_step_p_xgb_09,
                                nrounds = xgb_09_train_model$best_iteration, 
                                num_class = 4,           
                                early_stopping_rounds = 5, 
                                objective = "multi:softmax",
                                #watchlist=xgb_09_watchlist,
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
xgb_09_xgboost_model_time_end <- Sys.time()
xgb_09_xgboost_model_time_taken <- xgb_09_xgboost_model_time_end - xgb_09_xgboost_model_time_start
xgb_09_xgboost_model_time_taken #0.09779692 secs #4coreså

xgb_09_xgboost_model_auc <- str_split(xgb_09_xgboost_model$best_msg[1], ":")[[1]][2]
#xgb_09_xgboost_model_auc
#xgb_09_xgboost_model_auc <- str_sub(xgb_09_xgboost_model_auc, start = 1L, end = -12L)
xgb_09_xgboost_model_auc # 1.000000

# xgb_09_xgboost_model_auc <- str_split(xgb_09_xgboost_model$best_msg[1], ":")[[1]][3]
# xgb_09_xgboost_model_auc #0.562607


# EXPORT MODELS -----------------------------------------------------------
# SAVE MODEL --------------------------------------------------------------
xgb.save(xgb_09_xgboost_model, fname = "models/lucas_xgb_09.model")

sink(file = "models/lucas_xgb_09_raw.json")
xgb_09_xgboost_model_raw <- xgb.save.raw(xgb_09_xgboost_model, raw_format = "ubj")
xgb_09_xgboost_model_raw
sink()

# sink(file = "xgb_ens_models/lucas_xgb_09_raw.json")
# xgb_09_xgboost_model_raw <- xgb.save.raw(xgb_09_xgboost_model, raw_format = "json")
# xgb_09_xgboost_model_raw
# sink()
