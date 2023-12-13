library(ggpubr)
library(dplyr)



path_stem = "C:/Users/Marce/Documents/MBPhD/OtherStuff/AI4LS/models/"


models = c("elastic_net", "svm", "random_forest", "xgboost")

i=1
roc = data.frame()
for (i in 1:length(models)) {
  
  curr_model = models[i]
  
  curr_roc = read.csv(header = T, stringsAsFactors = F, 
                      file = paste0(path_stem, curr_model, "/", curr_model, "_LU2018_ROCAUCscores_ovr.csv"))
  head(curr_roc)
  
  curr_roc$model = curr_model
  roc = rbind(roc, curr_roc)
  
}


roc_melt = reshape2::melt(roc, id.vars = c("X", "fold_id", "model"))
head(roc_melt)

roc_melt$variable = factor(roc_melt$variable, levels = c("CV_train", "CV_test", "held_out"))
roc_melt$model = factor(roc_melt$model, levels = models)

ggplot(data = roc_melt, mapping = aes(x = model, y = value, color = model)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0.5, 1)) + 
  ggtitle("Model performances") + ylab("ROC-AUC") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1), text = element_text(size = 16),
        plot.title = element_text(face = "bold", hjust = 0.5), legend.position = "none") +
  facet_wrap(~variable)

