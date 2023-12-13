## LUC18 EXTRACT MICROBIAL SAMPLES


# data --------------------------------------------------------------------
meta_train <- read.delim("z_datasets/Meta_data_all_samps_coordinates_train_set.txt", sep = "\t", header = T)
meta_test  <- read.delim("z_datasets/Meta_data_all_samps_coordinates_test_set.txt", sep = "\t", header = T)
glimpse(meta_train)
glimpse(meta_test)


# extract point and barcode id --------------------------------------------
meta_train_id <- select(meta_train, c("POINTID", "BARCODE_ID"))
meta_test_id <- select(meta_test, c("POINTID", "BARCODE_ID"))
meta_id <- rbind(meta_train_id, meta_test_id)
dim(meta_id)


# remove microbial ids from luc18 -----------------------------------------
fn_luc18 <- filter(sha_luc18, !POINTID %in% meta_id$POINTID)
dim(fn_luc18)
dim(sel_luc18)


# remove NA at lu_recode ---------------------------------------------------
fn_luc18 <- filter(fn_luc18, !is.na(LU1_recoded))

# export fn luc18 ---------------------------------------------------------
write.csv(fn_luc18, "z_datasets/luc18_fn.csv", quote = F, row.names = F)

