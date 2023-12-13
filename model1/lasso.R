### LASSO


# libs --------------------------------------------------------------------

library(caret)
library(glmnet)
library(recipes)



# data --------------------------------------------------------------------
glimpse(fn_luc18)
glimpse(fn_luc15)
glimpse(fn_luc09)


# model features ---------------------------------------------------------
md_luc18 <- select(fn_luc18, -c("POINTID"))
md_luc15 <- select(fn_luc15, -c("POINTID"))
md_luc09 <- select(fn_luc09, -c("POINTID"))
glimpse(md_luc18)
glimpse(md_luc15)
glimpse(md_luc09)
table(md_luc18$LU1_recoded, useNA = "always")


# Split the data into training and testing sets
set.seed(123)
trainIndex_18 <- createDataPartition(md_luc18$LU1_recoded, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
train_data_18 <- md_luc18[trainIndex_18, ]
test_data_18 <- md_luc18[-trainIndex_18, ]

# Example imputation
train_impute_18 <- preProcess(train_data_18, method = c("medianImpute"))
test_impute_18 <- preProcess(test_data_18, method = c("medianImpute"))

train_data_18_inputed <- predict(train_impute_18, newdata = train_data_18)
test_data_18_inputed <-  predict(test_impute_18, newdata = test_data_18)

# Create a recipe for pre-processing (including one-hot encoding for categorical variables)
preprocess_recipe_18 <- recipe(LU1_recoded ~ ., data = train_data_18) %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Fit and apply the pre-processing recipe
preprocess_model_18 <- prep(preprocess_recipe_18)
train_data_18_processed <- bake(preprocess_model_18, new_data = train_data_18_inputed)
test_data_18_processed <- bake(preprocess_model_18, new_data = test_data_18_inputed)
View(train_data_18_processed)

dim(train_data_18_processed)
colnames(train_data_18_processed)

# Create a control object for cross-validation
ctrl_18 <- trainControl(method = "cv", number = 5)

# Train the Lasso regression model using glmnet
train_data_18_processed_x <- select(train_data_18_processed, -c("LU1_recoded"))

# Measure execution time
execution_time <- system.time({
  # Train the Lasso regression model using glmnet
  lasso_18 <- train(LU1_recoded ~ ., 
                    data = train_data_18_processed, 
                    method = "glmnet", 
                    trControl = ctrl_18,
                    maxit = 100000000,
                    thresh = 1e-7)
})[3]  # Extract the elapsed time

cat("Execution time:", execution_time, "seconds\n")

# lasso_model <- cv.glmnet(x = as.matrix(train_data_18_processed[, -c("LU1_recoded")]), 
#                          y = train_data_18_processed$LU1_recoded,
#                          alpha = 1,
#                          maxit = 100000)  # or a higher value
# 
# lasso_model <- cv.glmnet(x = as.matrix(train_data_18_processed[, -c("LU1_recoded")]), 
#                          y = train_data_18_processed$LU1_recoded,
#                          alpha = 1,
#                          thresh = 1e-7)  # or a smaller value



# lasso_18 <- cv.glmnet(x = train_data_18_processed_x,
#           y = train_data_18_processed$LU1_recoded,
#           alpha = 1,
#           standardize = FALSE)


# Make predictions on the test set
predictions <- predict(lasso_18, newdata = test_data_18_processed)

# Evaluate the model performance
mse <- mean((predictions - test_data_18_processed$LU1_recoded)^2)
rmse <- sqrt(mse)
print(paste("Mean Squared Error:", mse))
print(paste("Root Mean Squared Error:", rmse))
