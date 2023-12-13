#######################################################
# Optimise models to predict LU from feature matrices #
#    Written by Marcellus Augustine on 06/12/2023     #
#######################################################
import pickle
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, cohen_kappa_score
import sys
import os

# Define variables
location = "cs_cluster"
model_type = sys.argv[1]
n_trials = int(sys.argv[2])

# Define paths
if location == "local":
    input_df_path = "C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/Final data/"
    out_path = "C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/models"
elif location == "cs_cluster":
    input_df_path = f"/SAN/colcc/NMD_analysis/MIDAS_analysis/AI4LS/"
    out_path = f"/SAN/colcc/NMD_analysis/MIDAS_analysis/AI4LS/models"


####################################################
# Define the base models and their hyperparameters #
####################################################
def roc_auc_score_ovr(y_true, y_pred):
    roc_auc = roc_auc_score(y_score=y_pred, y_true=y_true, multi_class="ovr")
    return roc_auc


def elastic_net_objective(trial):
    hyperparams = {
        "multi_class": "multinomial",
        "solver": "saga",
        "penalty": "elasticnet",
        "C": trial.suggest_loguniform("C", 2 ** -10, 2 ** 15),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1, step=0.01)
    }
    return LogisticRegression(**hyperparams, random_state=42, max_iter=int(5e4))  # increase max_iter if it doesn't converge


def svm_objective(trial):
    # Define the search space for SVM hyperparameters
    hyperparams = {
        'probability': True,
        'C': trial.suggest_loguniform('C', 1e-3, 1e3),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    }
    return SVC(**hyperparams, random_state=42)


def xgboost_objective(trial):
    hyperparams = {
        'use_label_encoder': False,
        'objective': 'multi:softprob',
        'eval_metric': roc_auc_score_ovr,
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-9, 1),
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.1, 1),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1),
        'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1, 10),  # 1e-8, 10),
        'nthread': -1
    }
    return XGBClassifier(**hyperparams, random_state=42)


def random_forest_objective(trial):
    hyperparams = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),  # changed from 50-2000 to 50-1000
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 'auto']),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 50),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 1),
        'bootstrap': True,   # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'oob_score': trial.suggest_categorical('oob_score', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ["balanced_subsample", None]),  # , 'balanced']),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0, 1),
        'max_samples': trial.suggest_float('max_samples', 0, 0.6),  # changed from 0-1 to 0-0.6
        'n_jobs': -1
    }
    return RandomForestClassifier(**hyperparams, random_state=42)


models = dict(xgboost={'model': xgboost_objective},
              random_forest={'model': random_forest_objective},
              elastic_net={'model': elastic_net_objective},
              svm={'model': svm_objective})


models_final = {
    'xgboost': XGBClassifier,
    'random_forest': RandomForestClassifier,
    'elastic_net': LogisticRegression,
    'svm': SVC,
}

############################
# Load train and test data #
############################
x_train = pd.read_csv(f'{input_df_path}/TrainMatrix_2018.csv')
x_test = pd.read_csv(f"{input_df_path}/TestMatrix_2018.csv")


# Encode categorical variables - Sample_season, LC_simpl_2018
def one_hot_encode_variables(categs, categ_var, train, test):
    oh = OneHotEncoder(categories=categs)
    tr_oh = oh.fit_transform(train[[categ_var]])
    tr = pd.concat([train.drop(columns=[categ_var]),
                    pd.DataFrame(data=tr_oh.toarray(), columns=oh.get_feature_names_out())], axis=1)

    ts_oh = oh.transform(test[[categ_var]])
    ts = pd.concat([test.drop(columns=[categ_var]),
                    pd.DataFrame(data=ts_oh.toarray(), columns=oh.get_feature_names_out())], axis=1)
    return tr, ts


x_train, x_test = one_hot_encode_variables(categs=[list(set(x_train["Sample_season"].to_list() + x_test["Sample_season"].to_list()))],
                                           categ_var="Sample_season", train=x_train, test=x_test)

x_train, x_test = one_hot_encode_variables(categs=[list(set(x_train["LC_simpl_2018"].to_list() + x_test["LC_simpl_2018"].to_list()))],
                                           categ_var="LC_simpl_2018", train=x_train, test=x_test)

x_train, x_test = one_hot_encode_variables(categs=[list(set(x_train["Depth"].to_list() + x_test["Depth"].to_list()))],
                                           categ_var="Depth", train=x_train, test=x_test)

# one hot encoded the target variable for multiclass classification
print(list(set(x_train["clean_LU"].to_list() + x_test["clean_LU"].to_list())))
x_train["clean_LU_label"] = 0
x_train.loc[x_train["clean_LU"] == "U100", "clean_LU_label"] = 0
x_train.loc[x_train["clean_LU"] == "U111", "clean_LU_label"] = 1
x_train.loc[x_train["clean_LU"] == "U120", "clean_LU_label"] = 2
x_train.loc[x_train["clean_LU"] == "U400", "clean_LU_label"] = 3
x_train.drop(columns=["clean_LU"], inplace=True)
print(x_train["clean_LU_label"].value_counts())

x_test["clean_LU_label"] = 0
x_test.loc[x_test["clean_LU"] == "U100", "clean_LU_label"] = 0
x_test.loc[x_test["clean_LU"] == "U111", "clean_LU_label"] = 1
x_test.loc[x_test["clean_LU"] == "U120", "clean_LU_label"] = 2
x_test.loc[x_test["clean_LU"] == "U400", "clean_LU_label"] = 3
x_test.drop(columns=["clean_LU"], inplace=True)
print(x_test["clean_LU_label"].value_counts())


# Scale data - exclude the one hot encoded variables
categ_cols_enc = ['Sample_season_Autumn', 'Sample_season_Summer', 'Sample_season_Spring',
                  'LC_simpl_2018_Cropland', 'LC_simpl_2018_Woodland',  'LC_simpl_2018_Grassland',
                  'LC_simpl_2018_Bareland', 'LC_simpl_2018_Shrubland',
                  'Depth_0-10', 'Depth_0-20',
                  "clean_LU_label"]

sc = StandardScaler(with_std=True, with_mean=True)
x_train_sc = sc.fit_transform(x_train.set_index("BARCODE_ID").drop(columns=categ_cols_enc))
x_test_sc = sc.transform(x_test.set_index("BARCODE_ID").drop(columns=categ_cols_enc))
x_train = pd.concat([x_train[["BARCODE_ID"] + categ_cols_enc], pd.DataFrame(data=x_train_sc, columns=sc.get_feature_names_out())], axis=1)
x_test = pd.concat([x_test[["BARCODE_ID"] + categ_cols_enc], pd.DataFrame(data=x_test_sc, columns=sc.get_feature_names_out())], axis=1)

# Split into features and target
X = x_train.drop(columns=['clean_LU_label'])
y = x_train['clean_LU_label'].to_numpy().ravel()

Xtest = x_test.drop(columns=['clean_LU_label'])
ytest = x_test['clean_LU_label'].to_numpy().ravel()
del x_train, x_test

#############################
# Loop over the base models #
#############################
# Define the cross-validation scheme - used 5-fold 5-repeated CV for speed
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

for _ in [0]:
    name = model_type
    config = models[name]

    # Define the objective
    def base_objective(trial):
        model = config['model'](trial)
        # Define the scores list
        scores = []

        # Loop over the cross-validation folds
        for train_idx, test_idx in cv.split(X, y):
            # Split the data into train and test sets
            # X_train, y_train = X.iloc[train_idx, :], y.iloc[:, train_idx]
            # X_test, y_test = X.iloc[test_idx, :], y.iloc[:, test_idx]
            X_train, y_train = X.iloc[train_idx, :], y[train_idx]
            X_test, y_test = X.iloc[test_idx, :], y[test_idx]

            # Fit the model and make predictions
            model.fit(X_train.drop(columns=["BARCODE_ID"]), y_train)
            y_pred = model.predict_proba(X_test.drop(columns=["BARCODE_ID"]))

            # Evaluate the predictions
            score = roc_auc_score_ovr(y_true=y_test, y_pred=y_pred)
            scores.append(score)

        # Calculate the mean score
        mean_score = np.mean(scores)

        # Report the score to Optuna
        return mean_score


    # Optimize the hyperparameters if desired
    if not os.path.exists(f'{out_path}/{name}/study_{name}_LU218.pkl'):

        # Define the study
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(study_name=name, sampler=sampler, direction="maximize",
                                    storage=f"sqlite:///{out_path}/{name}/study_{name}_LU2018.db")

        study.optimize(base_objective, n_trials=n_trials)
        print(f"{name}: optimised study")

        # Save the best parameters to pkl file
        with open(f"{out_path}/{name}/study_{name}_LU218.pkl", 'wb') as f:
            pickle.dump(study, f)
        print(f"{name}: saved study to to .pkl file")

    else:
        with open(f"{out_path}/{name}/study_{name}_LU218.pkl", 'rb') as f:
            study = pickle.load(f)

    # Define the pipeline for the best parameters model with undersampling
    param_dict = study.best_params
    if name == "elastic_net":
        param_dict["penalty"] = "elasticnet"
        param_dict["solver"] = "saga"
        param_dict["max_iter"] = int(5e4)
    elif name == "xgboost":
        param_dict['use_label_encoder'] = False
        param_dict['objective'] = 'multi:softprob'
        param_dict['eval_metric'] = roc_auc_score_ovr
        param_dict['nthread'] = -1
    elif name == "random_forest":
        param_dict["n_jobs"] = -1
    elif name == "svm":
        param_dict['probability'] = True
    print(f"{name}: created param_dict to get predictions on CV folds")

    model = models_final[name](**param_dict, random_state=42)

    # Define the predictions list
    pred = pd.DataFrame()
    tr_pred = pd.DataFrame()
    hout_ts_pred = pd.DataFrame()
    all_scores = pd.DataFrame()

    # Loop over the cross-validation folds and predict with the best parameters
    counter = 0
    for train_idx, test_idx in cv.split(X, y):

        # Split the data into train and test sets
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]

        # Fit the model and make predictions
        model.fit(X_train.drop(columns=["BARCODE_ID"]), y_train)

        # TRAIN set: save predictions to file
        tr_predictions = model.predict_proba(X_train.drop(columns=["BARCODE_ID"]))
        tr_score = roc_auc_score_ovr(y_true=y_train, y_pred=tr_predictions)

        # TEST set: save the predictions for the stacked ensemble
        predictions = model.predict_proba(X_test.drop(columns=["BARCODE_ID"]))
        test_score = roc_auc_score_ovr(y_true=y_test, y_pred=predictions)

        # HELD-OUT TEST set: save predictions to file
        hout_ts_predictions = model.predict_proba(Xtest.drop(columns=["BARCODE_ID"]))
        hout_ts_score = roc_auc_score_ovr(y_true=ytest, y_pred=hout_ts_predictions)

        # Concatenate to save to file
        tmp_scores = pd.DataFrame(data={"fold_id": counter, "CV_train": tr_score,
                                        "CV_test": test_score, "held_out": hout_ts_score}, index=[0])
        all_scores = pd.concat([all_scores, tmp_scores], axis=0, ignore_index=True)
        counter += 1

    print(f"{name}: computed predictions on CV fold")

    # save scores
    all_scores.to_csv(fr"{out_path}/{name}/{name}_LU2018_ROCAUCscores_ovr.csv", index=True)

    print(f"{name}: appended predictions and saved to file")

print(f"{name}: finished")
