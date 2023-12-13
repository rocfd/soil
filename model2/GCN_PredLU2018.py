#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
# GCN model: edges are distance between sampling locations, nodes are samples
# Predict land use category as a node classification task
# Written by Marcellus Augustine on 07/12/2023
#############################################
# Standard Libraries
import pickle
import logging
import os
import sys

# Third-party Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import optuna
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix

# Scikit-learn Libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

# PyTorch Libraries
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.loader import NeighborLoader


############################################################################################################
# Filenames, Global Variables and Parameters
############################################################################################################
location = "nemo"
KNN = int(sys.argv[1])  # vary for number of nodes to sample during training: 10, 25, 50
num_trials = int(sys.argv[2])  # numer of trials for optuna (100 unless timed out)


# Define paths
if location == "local":
    INPUT_DF_PATH = "C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/Final data/"
    OPTUNA_DATABASE_PATH = f"C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/models/GCN/optuna_db/"
    MODEL_PATH = f"C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/models/GCN/"
elif location == "nemo":
    INPUT_DF_PATH = "/camp/home/augustm/working/augustm/other_projs/AI4LS/"
    OPTUNA_DATABASE_PATH = "/camp/home/augustm/working/augustm/other_projs/AI4LS/models/GCN/optuna_db/"
    MODEL_PATH = "/camp/home/augustm/working/augustm/other_projs/AI4LS/models/GCN/"

# Number of epochs
EPOCHS = int(os.getenv("EPOCHS", 100))

# Study name
STUDY_NAME = 'study_GCN_LU2018'

# Optuna file name
TRAINING_DATABASE_FILE = f'sqlite:///{OPTUNA_DATABASE_PATH}{STUDY_NAME}.db'  # SQLite database - changed path


############################################################################################################
# GCN class model - to use edge weights
############################################################################################################
class GNN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate):
        super(GNN, self).__init__()

        # Method for preprocessing/postprocessing layers
        def create_mlp_layers(n_layers, n_in, n_out):
            layers = [Sequential(Linear(n_in, n_out), ReLU(), BatchNorm1d(n_out))]
            for _ in range(n_layers - 1):
                layers.append(Sequential(Linear(n_out, n_out), ReLU(), BatchNorm1d(n_out)))
            return layers
        self.preprocess = nn.ModuleList(create_mlp_layers(num_layers_pre, n_features, n_hidden))

        # Create Graph Convolution layers
        conv_mult = [2 if i < (num_layers - 1) else 1 for i in range(num_layers)]  # all have 2 but layer has 1
        self.convs = nn.ModuleList([  # create list of layers and convert to nn.ModuleList()
            # first layer: i=0, therefore GCNConv(n_hidden * 1, n_hidden * 2)
            # second layer: i=1, therefore GCNConv(n_hidden * 2, n_hidden * 2)  # 1st index (0) into conv_mult
            # third layer: i=2, therefore GCNConv(n_hidden * 2, n_hidden * 2)  # 2nd index (1) into conv_mult
            # fourth layer: i=3, therefore GCNConv(n_hidden * 2, n_hidden * 1)  # 3rd index (2) into conv_mult
            GCNConv(n_hidden * (conv_mult[i - 1] if i > 0 else 1), n_hidden * mult) for i, mult in enumerate(conv_mult)
        ])
        self.conv_norms = nn.ModuleList([BatchNorm1d(n_hidden * mult) for mult in conv_mult])
        self.conv_relu = nn.ModuleList([ReLU() for _ in conv_mult])
        self.postprocess = nn.ModuleList(create_mlp_layers(num_layers_post, n_hidden, n_hidden))
        max_layers = max(num_layers_pre, num_layers, num_layers_post)
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout * (rate ** i)) for i in range(max_layers)])
        self.classifier = Linear(n_hidden, n_classes)

    def forward(self, x, edge_index, edge_weight):

        # Helper method for applying sequential layers and dropout
        def apply_layers_with_dropout(layers, xx):
            for i, layer in enumerate(layers):
                xx = layer(xx)
                xx = self.dropouts[i](xx)
            return xx

        x = apply_layers_with_dropout(self.preprocess, x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.conv_norms[i](x)
            x = self.conv_relu[i](x)
            x = self.dropouts[i](x)
        x = apply_layers_with_dropout(self.postprocess, x)
        x = self.classifier(x)
        return x


############################################################################################################
# Objective function for a single set of hyperparameters in Optuna
############################################################################################################
def roc_auc_score_ovr(y_true, y_pred):
    roc_auc = roc_auc_score(y_score=y_pred, y_true=y_true, multi_class="ovr")
    return roc_auc


def base_objective_single_arg(train_idx=None, test_idx=None, xx=None, yy=None, n_features=None,
                              n_classes=None, n_hidden=None, num_layers=None, num_layers_pre=None, num_layers_post=None,
                              dropout=None, lr=None, weight_decay=None,
                              nodes_train=None, batch_size=None, num_epochs=None, n_b=None,
                              Network=None, edge_weights=None, rate=None):  # I kept weights here but they are not used
    model = GNN(n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate)

    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Split the data into train and test sets
    X_folds = xx.loc[[True if i in nodes_train else False for i in xx.index], :]
    y_folds = yy[[True if i in nodes_train else False for i in xx.index]]
    X_train, y_train = X_folds.iloc[train_idx, :], y_folds[train_idx]
    X_test, y_test = X_folds.iloc[test_idx, :], y_folds[test_idx]

    # Create torch geometric data object
    Xgraph = torch.tensor(xx.values, dtype=torch.float)

    # Labels for a binary classification task
    ygraph = torch.tensor(yy, dtype=torch.float)
    data = Data(x=Xgraph, edge_index=Network, edge_attr=edge_weights, y=ygraph,
                train_mask=torch.tensor([True if i in X_train.index else False for i in xx.index], dtype=torch.bool),
                test_mask=torch.tensor([True if i in X_test.index else False for i in xx.index], dtype=torch.bool))
    train_loader = NeighborLoader(
        data,
        num_neighbors=[n_b] * num_layers,
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )
    criterion = nn.CrossEntropyLoss()  # F.cross_entropy(out, data_train.y)
    model.train()

    for epoch in range(num_epochs + 1):
        total_loss = 0

        # Train on batches
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out[batch.train_mask, :], batch.y[batch.train_mask].long())
            total_loss += loss
            loss.backward()
            optimizer.step()

    # Make predictions on test fold
    model.eval()

    # Create a NeighborLoader for the test nodes
    test_loader = NeighborLoader(data, num_neighbors=[n_b] * num_layers,  # [10, 10],
                                 batch_size=sum([True if i in X_test.index else False for i in xx.index]),
                                 input_nodes=data.test_mask)

    # Extract the single batch from the loader
    test_batch = next(iter(test_loader))

    # Evaluate the model
    with torch.no_grad():
        out = model(test_batch.x, test_batch.edge_index, test_batch.edge_attr)
    y_pred = torch.nn.Softmax(out[test_batch.test_mask]).detach().numpy()  # torch.exp(out)

    # Evaluate the predictions
    score = roc_auc_score_ovr(y_true=test_batch.y[test_batch.test_mask].detach().numpy(), y_pred=y_pred)
    # print(score)
    # Report the score to Optuna
    return score


############################################################################################################
# Optimization function
############################################################################################################
def run_optimization(n_trials, X=None, y=None, cv=None, nodes_train=None, nodes_test=None, Network=None,
                     edge_weights=None, num_epochs=None):  # kept weights but aren't used (remove if memory is problem)
    """
    Wrapper function for the objective function.
    """
    # Prepare arguments for parallel execution
    X_folds = X.loc[[True if i in nodes_train else False for i in X.index], :]
    y_folds = y[[True if i in nodes_train else False for i in X.index]]
    cv_splits = list(cv.split(X_folds, y_folds))
    args = [dict(train_idx=train_idx, test_idx=test_idx, xx=X, yy=y)
            for fold_id, (train_idx, test_idx) in enumerate(cv_splits)]

    # Run the optimization in parallel

    def objective(trial):
        """
                Objective function for Optuna optimization.
        """
        lr = trial.suggest_categorical("lr", [0.01, 0.02, 0.03])
        n_hidden = trial.suggest_categorical("n_hidden", [16, 32, 64, 128, 256, 512, 1024])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256,
                                                              512])  # larger batch sizes may lead to better results but take longer to train due to memory demands
        num_layers = trial.suggest_categorical('num_layers',
                                               [2, 3, 4])  # fixed to value that performed best in previous models
        num_layers_pre = trial.suggest_categorical('num_layers_pre',
                                                   [1, 2, 3])  # fixed to value that performed best in previous models
        num_layers_post = trial.suggest_categorical('num_layers_post',
                                                    [1, 2, 3])  # fixed to value that performed best in previous models
        # num_epochs = EPOCHS  # trial.suggest_categorical('num_epochs', [100, 200, 400]) # Optimizing the number of epochs slows the optimization down considerably
        rate = trial.suggest_categorical("rate",
                                         [0.1, 0.2, 0.3, 0.4, 0.5])  # dropout rate for the dropout layer in the model
        n_b = KNN  # necessary for the NeighborLoader
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_features = len(X.columns)
        n_classes = len(np.unique(y))
        print(n_classes)

        # Use Joblib's Parallel to parallelize the execution
        scores = Parallel(n_jobs=-1)(
            delayed(base_objective_single_arg)(**arg, n_features=n_features, n_classes=n_classes, n_hidden=n_hidden,
                                               num_layers=num_layers, num_layers_pre=num_layers_pre,
                                               num_layers_post=num_layers_post, dropout=dropout, lr=lr,
                                               weight_decay=weight_decay,
                                               nodes_train=nodes_train, batch_size=batch_size,
                                               num_epochs=num_epochs, n_b=n_b, Network=Network,
                                               edge_weights=edge_weights, rate=rate) for arg in
            # I kept weights here but they are not used (remove if memory is a problem)
            args)

        # Calculate the mean score
        mean_score = np.mean(scores)
        # print(mean_score)
        return mean_score

    if os.path.exists(f'{OPTUNA_DATABASE_PATH}{STUDY_NAME}.db'):
        study = optuna.load_study(study_name=STUDY_NAME)  # , storage=TRAINING_DATABASE_FILE)
    else:
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(study_name=STUDY_NAME, # storage=TRAINING_DATABASE_FILE,
                                    direction='maximize',
                                    load_if_exists=False, sampler=sampler)  # load_if_exists=True does job of os.path.exists(f'{DATABASE_PATH}{STUDY_NAME}.db') but caused problems in Myriad
    study.optimize(objective, n_trials=n_trials)
    return study


############################################################################################################
# Function that performs the predictions for one fold
############################################################################################################
def make_predictions(train_idx=None, test_idx=None, fold_id=None, xx=None, yy=None,
                     n_features=None, n_classes=None, n_hidden=None, num_layers=None, num_layers_pre=None,
                     num_layers_post=None, dropout=None, lr=None, weight_decay=None, nodes_train=None,
                     nodes_test=None, batch_size=None, num_epochs=None, n_b=None, Network=None, edge_weights=None,
                     rate=None):
    """
    Function that performs the predictions for each fold with the best model devised with the optimum hyperparameters
    found by Optuna. The rest of the parameters are fitted to the training fold data and the predictions are made in the
    validation fold
    """
    model = GNN(n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate)
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Split the data into train and test sets
    X_folds = xx.loc[[True if i in nodes_train else False for i in xx.index], :]
    y_folds = yy[[True if i in nodes_train else False for i in xx.index]]
    X_train, y_train = X_folds.iloc[train_idx, :], y_folds[train_idx]
    X_test, y_test = X_folds.iloc[test_idx, :], y_folds[test_idx]

    # Create torch geometric data object
    Xgraph = torch.tensor(xx.values, dtype=torch.float)

    # Edge list
    edge_index = Network

    # Labels for a binary classification task
    ygraph = torch.tensor(yy, dtype=torch.float)
    data = Data(x=Xgraph, edge_index=edge_index, edge_attr=edge_weights, y=ygraph,
                train_mask=torch.tensor([True if i in X_train.index else False for i in xx.index], dtype=torch.bool),
                test_mask=torch.tensor([True if i in X_test.index else False for i in xx.index], dtype=torch.bool),
                val_mask=torch.tensor([True if i in nodes_test else False for i in xx.index], dtype=torch.bool))
    train_loader = NeighborLoader(
        data,
        num_neighbors=[n_b] * num_layers,
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )

    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs + 1):
        total_loss = 0
        # Train on batches
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out[batch.train_mask, :], batch.y[batch.train_mask].long())
            total_loss += loss
            loss.backward()
            optimizer.step()

    ####################### save model
    filename = f"{MODEL_PATH}Best_SAGE_train_foldID_{fold_id}{STUDY_NAME}.pth"
    torch.save(model.state_dict(), filename)
    #######################

    model.eval()
    # Create a NeighborLoader for the train nodes
    train_loader = NeighborLoader(data,
                                 num_neighbors=[n_b] * num_layers,
                                 batch_size=sum([True if i in X_train.index else False for i in xx.index]),
                                 input_nodes=data.train_mask)

    # Extract the single batch from the loader
    train_batch = next(iter(train_loader))

    # Evaluate the model
    with torch.no_grad():
        out = model(train_batch.x, train_batch.edge_index, train_batch.edge_attr)
    predictions_train = torch.nn.Softmax(out[train_batch.train_mask]).detach().numpy()  # torch.exp(out)
    predictions_train = pd.DataFrame(
        dict(Gene=xx.index[train_batch.n_id[train_batch.train_mask].detach().numpy()], Prob=predictions_train,
             Target=train_batch.y[train_batch.train_mask].detach().numpy(),
             Fold_ID=np.repeat(fold_id, len(predictions_train[:, 1])),
             Set=np.repeat('Train (fold)', len(predictions_train[:, 1]))))

    # Create a NeighborLoader for the test nodes
    test_loader = NeighborLoader(data,
                                 num_neighbors=[n_b] * num_layers,
                                 batch_size=sum([True if i in X_test.index else False for i in xx.index]),
                                 input_nodes=data.test_mask)

    # Extract the single batch from the loader
    test_batch = next(iter(test_loader))
    with torch.no_grad():
        out = model(test_batch.x, test_batch.edge_index, test_batch.edge_attr)
    predictions_test = torch.nn.Softmax(out[test_batch.test_mask]).detach().numpy()  # torch.exp(out)
    predictions_test = pd.DataFrame(
        dict(Gene=xx.index[test_batch.n_id[test_batch.test_mask].detach().numpy()], Prob=predictions_test,
             Target=test_batch.y[test_batch.test_mask].detach().numpy(),
             Fold_ID=np.repeat(fold_id, len(predictions_test[:, 1])),
             Set=np.repeat('Test (fold)', len(predictions_test[:, 1]))))

    # Create a NeighborLoader for the held-out nodes
    hout_loader = NeighborLoader(data,
                                 num_neighbors=[n_b] * num_layers,
                                 batch_size=sum([True if i in nodes_test else False for i in xx.index]),
                                 input_nodes=data.val_mask)

    # Extract the single batch from the loader
    hout_batch = next(iter(hout_loader))
    with torch.no_grad():
        out = model(hout_batch.x, hout_batch.edge_index, hout_batch.edge_attr)
    predictions_ht = torch.nn.Softmax(out[hout_batch.val_mask]).detach().numpy()  # torch.exp(out)
    predictions_ht = pd.DataFrame(
        dict(Gene=xx.index[hout_batch.n_id[hout_batch.val_mask].detach().numpy()], Prob=predictions_ht,
             Target=hout_batch.y[hout_batch.val_mask].detach().numpy(),
             Fold_ID=np.repeat(fold_id, len(predictions_ht[:, 1])),
             Set=np.repeat('Held out (pred with fold model)', len(predictions_ht[:, 1]))))
    pr = pd.concat([predictions_train, predictions_test, predictions_ht], axis=0)
    return pr


def one_hot_encode_variables(categs, categ_var, train, test):
    oh = OneHotEncoder(categories=categs)
    tr_oh = oh.fit_transform(train[[categ_var]])
    tr = pd.concat([train.drop(columns=[categ_var]),
                    pd.DataFrame(data=tr_oh.toarray(), columns=oh.get_feature_names_out())], axis=1)

    ts_oh = oh.transform(test[[categ_var]])
    ts = pd.concat([test.drop(columns=[categ_var]),
                    pd.DataFrame(data=ts_oh.toarray(), columns=oh.get_feature_names_out())], axis=1)
    return tr, ts

def pairwise_distances(df, col_list):
    coords = df[col_list].values
    distances = pdist(X=coords, metric="euclidean")
    dist_mat = squareform(X=distances)
    dist_df = pd.DataFrame(dist_mat, columns=df["BARCODE_ID"].values, index=df["BARCODE_ID"].values)
    return dist_df


def reshape_dist_mat(df, sq_dist_df, dist_type):
    # Reshape
    samples = df['BARCODE_ID'].to_list()
    samples.sort()
    distance_data = []
    for i, sample1 in enumerate(samples):
        for j, sample2 in enumerate(samples):
            if i < j:
                distance_data.append([sample1, sample2, sq_dist_df.iloc[i, j]])

    dist_df_long = pd.DataFrame(distance_data, columns=['source_name', 'target_name', f'{dist_type}'])
    return dist_df_long


def convert_adjacency_to_edge_index(adj_mat: object) -> object:
    """Converts adjacency matrix to edge index format for PyTorch."""
    # Convert adjacency matrix to coo format
    coo = coo_matrix(adj_mat)
    # Get source and target indices - rows and col coords for non-0 elements in the original dense weight matrix (Wl)
    source_nodes = torch.tensor(coo.row, dtype=torch.long)  # construct tensor of dtype torch.long = torch.int64
    target_nodes = torch.tensor(coo.col, dtype=torch.long)
    # torch.stack: concatenate the source and target nodes to create a 2D tensor (2 x sum(source, target))
    # torch.tensor: create a 1D tensor of len==ncol of above 2D tensor containing the edge weights
    return torch.stack([source_nodes, target_nodes], dim=0), torch.tensor(coo.data, dtype=torch.float)


def main():
    try:
        ############################
        # Load train and test data #
        ############################
        x_train = pd.read_csv(f'{INPUT_DF_PATH}/TrainMatrix_2018.csv')
        x_test = pd.read_csv(f"{INPUT_DF_PATH}/TestMatrix_2018.csv")

        # Encode categorical variables - Sample_season, LC_simpl_2018
        x_train, x_test = one_hot_encode_variables(
            categs=[list(set(x_train["Sample_season"].to_list() + x_test["Sample_season"].to_list()))],
            categ_var="Sample_season", train=x_train, test=x_test)

        x_train, x_test = one_hot_encode_variables(
            categs=[list(set(x_train["LC_simpl_2018"].to_list() + x_test["LC_simpl_2018"].to_list()))],
            categ_var="LC_simpl_2018", train=x_train, test=x_test)

        x_train, x_test = one_hot_encode_variables(
            categs=[list(set(x_train["Depth"].to_list() + x_test["Depth"].to_list()))],
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
                          'LC_simpl_2018_Cropland', 'LC_simpl_2018_Woodland', 'LC_simpl_2018_Grassland',
                          'LC_simpl_2018_Bareland', 'LC_simpl_2018_Shrubland',
                          'Depth_0-10', 'Depth_0-20',
                          "clean_LU_label"]
        # 'clean_LU_U120', 'clean_LU_U111', 'clean_LU_U100', 'clean_LU_U400']
        sc = StandardScaler(with_std=True, with_mean=True)
        x_train_sc = sc.fit_transform(x_train.set_index("BARCODE_ID").drop(columns=categ_cols_enc))
        x_test_sc = sc.transform(x_test.set_index("BARCODE_ID").drop(columns=categ_cols_enc))
        x_train = pd.concat([x_train[["BARCODE_ID"] + categ_cols_enc],
                             pd.DataFrame(data=x_train_sc, columns=sc.get_feature_names_out())], axis=1)
        x_test = pd.concat(
            [x_test[["BARCODE_ID"] + categ_cols_enc], pd.DataFrame(data=x_test_sc, columns=sc.get_feature_names_out())],
            axis=1)

        # Split into features and target
        X = x_train.copy(deep=True)

        Xtest = x_test.copy(deep=True)
        del x_train, x_test

        #######################################################################################################
        # Compute pairwise euclidean distance and then reformat into edge list
        #######################################################################################################
        # Read in data
        train = pd.read_table(f"{INPUT_DF_PATH}Meta_data_all_samps_coordinates_train_set.txt", sep="\t")
        test = pd.read_table(f"{INPUT_DF_PATH}Meta_data_all_samps_coordinates_test_set.txt", sep="\t")

        all_coords = pd.concat([train, test], axis=0)
        spat_dist_sq = pairwise_distances(df=all_coords, col_list=["X_coordinate", "Y_coordinate"])

        edge_index, edge_weights = convert_adjacency_to_edge_index(spat_dist_sq)

        # Subset list of train and held-out test nodes to common nodes
        Nodes_train = X["BARCODE_ID"].values
        Nodes_ht = Xtest["BARCODE_ID"].values

        # Create unified feature matrix to hold all data
        X_all_scaled = pd.concat([X, Xtest], axis=0)
        y_all = X_all_scaled["clean_LU_label"].values
        X_all_scaled.drop(columns=["clean_LU_label"], inplace=True)
        X_all_scaled.set_index("BARCODE_ID", inplace=True)

        del X, Xtest  #, node_idx_source, node_idx_target

        ############################################################################################################
        # Define the cross-validation scheme
        ############################################################################################################
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

        ############################################################################################################
        # Run optimization
        ############################################################################################################
        study = run_optimization(n_trials=num_trials, X=X_all_scaled, y=y_all, cv=cv, nodes_train=Nodes_train,
                                 nodes_test=Nodes_ht, Network=edge_index, edge_weights=edge_weights,
                                 num_epochs=EPOCHS)  # I kept weights here but they are not used (remove if memory is a problem)

        with open(f'{OPTUNA_DATABASE_PATH}{STUDY_NAME}.pkl', 'wb') as f:
            pickle.dump(study, f)

        #############################################################################################################
        # Select the best parameters
        #############################################################################################################
        lr = study.best_params['lr']
        n_hidden = study.best_params['n_hidden']
        dropout = study.best_params['dropout']
        weight_decay = study.best_params['weight_decay']
        num_layers = study.best_params['num_layers']
        num_layers_pre = study.best_params['num_layers_pre']
        num_layers_post = study.best_params['num_layers_post']
        batch_size = study.best_params['batch_size']
        num_epochs = EPOCHS  # study.best_params['num_epochs']
        rate = study.best_params['rate']
        n_b = KNN

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_features = len(X_all_scaled.columns)
        n_classes = len(np.unique(y_all))

        #############################################################################################################
        # Predictions with fold models
        #############################################################################################################
        X_folds = X_all_scaled.loc[[True if i in Nodes_train else False for i in X_all_scaled.index], :]
        y_folds = y_all[[True if i in Nodes_train else False for i in X_all_scaled.index]]

        args = [dict(train_idx=train_idx, test_idx=test_idx, fold_id=fold_id, xx=X_all_scaled, yy=y_all,
                     n_features=n_features, n_classes=n_classes, n_hidden=n_hidden, num_layers=num_layers,
                     num_layers_pre=num_layers_pre, num_layers_post=num_layers_post, dropout=dropout, lr=lr,
                     weight_decay=weight_decay, nodes_train=Nodes_train,
                     nodes_test=Nodes_ht, batch_size=batch_size, num_epochs=num_epochs, n_b=n_b, Network=edge_index,
                     edge_weights=edge_weights, rate=rate) for
                # I kept weights here but they are not used (remove if memory is a problem)
                fold_id, (train_idx, test_idx) in enumerate(cv.split(X_folds, y_folds))]
        results = Parallel(n_jobs=-1)(delayed(make_predictions)(**arg) for arg in args)
        pred = pd.concat(results, axis=0)

        #############################################################################################################
        # Save predictions
        #############################################################################################################
        filename = f"{MODEL_PATH}/predictions/Predictions_Best_SAGE_WithCVModels_{STUDY_NAME}.csv"
        pred.to_csv(filename, index=True)

    except Exception as e:
        print(f"Error in main: {e}")
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
    print("done script")
