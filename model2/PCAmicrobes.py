#################################################
# Compute PCAs from microbial diversity indices #
#  Written by Marcellus Augustine on 06/12/2023 #
#################################################
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Define path
path_stem = r"C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/Final data/"

# read in files
train = pd.read_table(f"{path_stem}Meta_data_all_samps_coordinates_train_set.txt", sep="\t")
test = pd.read_table(f"{path_stem}Meta_data_all_samps_coordinates_test_set.txt", sep="\t")

bac = pd.read_table(f"{path_stem}Bacteria_tax_table_genus_final.txt", sep="\t").set_index("genus").transpose().\
    reset_index().rename(columns={"index": "BARCODE_ID"})

fung = pd.read_table(f"{path_stem}Fungi_tax_table_genus.txt", sep="\t").set_index("genus").transpose().\
    reset_index().rename(columns={"index": "BARCODE_ID"})

euk = pd.read_table(f"{path_stem}Eukaryotes_tax_table_genus.txt", sep="\t").set_index("genus").transpose().\
    reset_index().rename(columns={"index": "BARCODE_ID"})


# Scale data
def scale_data(df, train_ids):
    scaler = StandardScaler(with_std=True, with_mean=True)
    scaler.fit(df.loc[df["BARCODE_ID"].isin(train_ids)].drop(columns=["BARCODE_ID"]))
    df_sc = scaler.transform(df.drop(columns=["BARCODE_ID"]))
    df_sc = pd.DataFrame(data=df_sc, index=df["BARCODE_ID"].values, columns=scaler.get_feature_names_out()).\
        reset_index().rename(columns={"index": "BARCODE_ID"})
    return df_sc


train_samps = train["BARCODE_ID"].to_list()
bac_sc = scale_data(df=bac, train_ids=train_samps)
fung_sc = scale_data(df=fung, train_ids=train_samps)
euk_sc = scale_data(df=euk, train_ids=train_samps)


# Compute PCA on train set and project test set onto these PCs
def compute_pca(train_set, test_set, microbe_str):
    pca = PCA(random_state=42)
    train_pcs = pca.fit_transform(train_set)
    train_pcs = pd.DataFrame(train_pcs, columns=[f"{microbe_str}_PC{i}" for i in range(pca.n_components_)], index=train_set.index)
    test_pcs = pd.DataFrame(pca.transform(test_set),
                            columns=[f"{microbe_str}_PC{i}" for i in range(pca.n_components_)], index=test_set.index)
    var_exp = pd.DataFrame(pca.explained_variance_ratio_, columns=["pct_var_exp"])
    return train_pcs, test_pcs, var_exp


bac_tr, bac_ts, bac_var_exp = compute_pca(train_set=bac_sc.loc[bac_sc["BARCODE_ID"].isin(train_samps)].set_index("BARCODE_ID"),
                                          test_set=bac_sc.loc[~bac_sc["BARCODE_ID"].isin(train_samps)].set_index("BARCODE_ID"),
                                          microbe_str="bac")

fung_tr, fung_ts, fung_var_exp = compute_pca(train_set=fung_sc.loc[fung_sc["BARCODE_ID"].isin(train_samps)].set_index("BARCODE_ID"),
                                             test_set=fung_sc.loc[~fung_sc["BARCODE_ID"].isin(train_samps)].set_index("BARCODE_ID"),
                                             microbe_str="fung")

euk_tr, euk_ts, euk_var_exp = compute_pca(train_set=euk_sc.loc[euk_sc["BARCODE_ID"].isin(train_samps)].set_index("BARCODE_ID"),
                                          test_set=euk_sc.loc[~euk_sc["BARCODE_ID"].isin(train_samps)].set_index("BARCODE_ID"),
                                          microbe_str="euk")


# Plot elbow to pick threshold for number of PCs
def plot_elbow(df, title_str):
    df = df.reset_index()  # reset index to use as x-axis value (number of PCs)
    sns.lineplot(data=df, x="index", y="pct_var_exp")
    plt.xlabel("PC")
    plt.ylabel("% variance explained")
    plt.title(f"{title_str}")


plot_elbow(df=bac_var_exp, title_str="Bacterial diversity indices")  # 75 PCs
plot_elbow(df=fung_var_exp, title_str="Fungal diversity indices")  # 75 PCs
plot_elbow(df=euk_var_exp, title_str="Eukaryotes diversity indices")  # 75

train_mat = train.merge(right=bac_tr.iloc[:, 0:75].reset_index().rename(columns={"index": "BARCODE_ID"}),
                        on="BARCODE_ID", how="inner")
train_mat = train_mat.merge(right=fung_tr.iloc[:, 0:75].reset_index().rename(columns={"index": "BARCODE_ID"}),
                            on="BARCODE_ID", how="inner")
train_mat = train_mat.merge(right=euk_tr.iloc[:, 0:75].reset_index().rename(columns={"index": "BARCODE_ID"}),
                            on="BARCODE_ID", how="inner")

test_mat = test.merge(right=bac_ts.iloc[:, 0:75].reset_index().rename(columns={"index": "BARCODE_ID"}),
                      on="BARCODE_ID", how="inner")
test_mat = test_mat.merge(right=fung_ts.iloc[:, 0:75].reset_index().rename(columns={"index": "BARCODE_ID"}),
                          on="BARCODE_ID", how="inner")
test_mat = test_mat.merge(right=euk_ts.iloc[:, 0:75].reset_index().rename(columns={"index": "BARCODE_ID"}),
                          on="BARCODE_ID", how="inner")

# one-hot encode: Sample_season, LC_simpl_2018

cols_to_drop = ["geometry", "X_coordinate", "Y_coordinate", "Sample_date", "LU1_2018", "Bioregions_chr", "LU1_2015",
                "LC_num", "LC1_2018", "POINTID"]

train_mat.drop(columns=cols_to_drop).to_csv(f"{path_stem}TrainMatrix_2018.csv", index=False)
test_mat.drop(columns=cols_to_drop).to_csv(f"{path_stem}TestMatrix_2018.csv", index=False)

