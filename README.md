****#IUCNN installation ***

install.packages("devtools")
library(usethis)
library(devtools)
install.packages(IUCNN)
library(IUCNN)

**#Python library installation using reticulate from within R**

install.packages("reticulate")
library("reticulate")
reticulate::py_install("tensorflow", pip = TRUE)

#Loading data
## Training occurrences
choose.files()
train_occ <- read.table("C:\\Users\\samuelK\\Desktop\\occ.txt",header=TRUE)
attach(train_occ)
names(train_occ)

## Training labels: Detailed classification
Choose.files()
train_lab <- read.table("C:\\Users\\samuelK\\Desktop\\Lab.txt",header=TRUE)

## Training labels: Binary classification labels
Choose.files()
train_lab2 <- read.table("C:\\Users\\samuelK\\Desktop\\broadlabels.txt",header=TRUE)


# Features generation
## Geographic features
geo_feat <-iucnn_prepare_features(train_occ, type="geographic",impute_features=FALSE)

## Climate features
climate_feat <-iucnn_prepare_features(train_occ, type="climate",impute_features=FALSE)

## Geographic + climate
allfeat <-iucnn_prepare_features(train_occ,type=c("geographic","climate"),impute_features=FALSE)

#Getting Labels
##For the 5-class

class_labfive <- iucnn_prepare_labels(train_lab,geo_feat,level="detail")

##For the 2-class

class_labtw <- iucnn_prepare_labels(train_lab,geo_feat,level="broad")

#Model training
#DETAILED CLASSIFICATION:5-CLASS
## Model checks

modelcheck_1 <- iucnn_train_model(x=geo_feat,lab=class_labfive, path_to_output= "iucnn_model_1")
modelcheck_2 <- iucnn_train_model(x=climate_feat, lab=class_labfive, path_to_output= "iucnn_model_2")
modelcheck_3 <- iucnn_train_model(x=allfeat, lab=class_labfive, path_to_output= "iucnn_model_3")

#Geographic_features

geography <- iucnn_modeltest(
  x=geo_feat,
  lab=class_labfive,
  logfile = "model_testing_logfile.txt",
  model_outpath = "modeltest",
  mode = "nn-class",
  cv_fold = 1,
  test_fraction = 0.2,
  n_layers = c("30","40_20","50_30_10"),
  dropout_rate = c(0, 0.1),
  use_bias = TRUE,
  balance_classes = FALSE,
  seed = 1234,
  act_f = "relu",
  act_f_out = "auto",
  max_epochs = 172,
  patience = 80,
  mc_dropout = TRUE,
  mc_dropout_reps = 100,
  randomize_instances = TRUE,
  rescale_features = FALSE,
  init_logfile = TRUE,
  recycle_settings = FALSE
  init_logfile = TRUE)

#Best model selection

geobestmodel <- iucnn_best_model(x=geography, criterion = "val_acc", require_dropout = FALSE)
summary(geobestmodel)
plot(geobestmodel)


#Climate_features

climate <- iucnn_modeltest(
  x=climate_feat,
  lab=class_labfive,
  logfile = "model_testing_logfile.txt1",
  model_outpath = "modeltest1",
  mode = "nn-class",
  cv_fold = 1,
  test_fraction = 0.2,
  n_layers = c("30","40_20","50_30_10"),
  dropout_rate = c(0, 0.1),
  use_bias = TRUE,
  balance_classes = FALSE,
  seed = 1234,
  act_f = "relu",
  act_f_out = "auto",
  max_epochs = 450,
  patience = 100,
  mc_dropout = TRUE,
  mc_dropout_reps = 100,
  randomize_instances = TRUE,
  rescale_features = FALSE,
  init_logfile = TRUE,
  recycle_settings = FALSE
)

climatebestM <- iucnn_best_model(x=climate, criterion = "val_acc", require_dropout = FALSE)
summary(climatebestM)
plot(climatebestM)


#Geographic + climate

combinedfeat <- iucnn_modeltest(
  x=allfeat,
  lab=class_labfive,
  logfile = "model_testing_logfile.txt2",
  model_outpath = "modeltest3",
  mode = "nn-class",
  cv_fold = 1,
  test_fraction = 0.2,
  n_layers = c("30","40_20","50_30_10"),
  dropout_rate = c(0, 0.1),
  use_bias = TRUE,
  balance_classes = FALSE,
  seed = 1234,
  act_f = "relu",
  act_f_out = "auto",
  max_epochs = 75,
  patience = 50,
  mc_dropout = TRUE,
  mc_dropout_reps = 100,
  randomize_instances = TRUE,
  rescale_features = FALSE,
  init_logfile = TRUE,
  recycle_settings = FALSE
)

allfeatbestM <- iucnn_best_model(x=combinedfeat, criterion = "val_acc", require_dropout = FALSE)
summary(allfeatbestM)
plot(allfeatbestM)


#BINARY CLASSIFICATION 
## Model checks

modelcheck_4 <- iucnn_train_model(x=geo_feat,lab=class_labtw, path_to_output= "iucnn_model_4")
modelcheck_5 <- iucnn_train_model(x=climate_feat, lab=class_labtw, path_to_output= "iucnn_model_5")
modelcheck_6 <- iucnn_train_model(x=allfeat, lab=class_labtw, path_to_output= "iucnn_model_6")

#Geographic_features
geography2 <- iucnn_modeltest(
  x=geo_feat,
  lab=class_labtw,
  logfile = "model_testing_logfile.txt3",
  model_outpath = "modeltest4",
  mode = "nn-class",
  cv_fold = 1,
  test_fraction = 0.2,
  n_layers = c("30","40_20","50_30_10"),
  dropout_rate = c(0, 0.1),
  use_bias = TRUE,
  balance_classes = FALSE,
  seed = 1234,
  act_f = "relu",
  act_f_out = "auto",
  max_epochs = 188,
  patience = 80,
  mc_dropout = TRUE,
  mc_dropout_reps = 100,
  randomize_instances = TRUE,
  rescale_features = FALSE,
  init_logfile = TRUE,
  recycle_settings = FALSE
)

geo2class <- iucnn_best_model(x=geography2, criterion = "val_acc", require_dropout = FALSE)
summary(geo2class) 
plot(geo2class)


### Climate_features

climate2 <- iucnn_modeltest(
  x=climate_feat,
  lab=class_labtw,
  logfile = "model_testing_logfile.txt4",
  model_outpath = "modeltest5",
  mode = "nn-class",
  cv_fold = 1,
  test_fraction = 0.2,
  n_layers = c("30","40_20","50_30_10"),
  dropout_rate = c(0, 0.1),
  use_bias = TRUE,
  balance_classes = FALSE,
  seed = 1234,
  act_f = "relu",
  act_f_out = "auto",
  max_epochs = 188,
  patience = 100,
  mc_dropout = TRUE,
  mc_dropout_reps = 100,
  randomize_instances = TRUE,
  rescale_features = FALSE,
  init_logfile = TRUE,
  recycle_settings = FALSE
)

climclass2 <- iucnn_best_model(x=climate2, criterion = "val_acc", require_dropout = FALSE)
summary(climclass2)
plot(climclass2)


###Geo+climate

combinedfeat2 <- iucnn_modeltest(
  x=allfeat,
  lab=class_labtw,
  logfile = "model_testing_logfile.txt5",
  model_outpath = "modeltest6",
  mode = "nn-class",
  cv_fold = 1,
  test_fraction = 0.2,
  n_layers = c("30","40_20","50_30_10"),
  dropout_rate = c(0, 0.1),
  use_bias = TRUE,
  balance_classes = FALSE,
  seed = 123,
  act_f = "relu",
  act_f_out = "auto",
  max_epochs = 125,
  patience = 80,
  mc_dropout = TRUE,
  mc_dropout_reps = 100,
  randomize_instances = TRUE,
  rescale_features = FALSE,
  init_logfile = TRUE,
  recycle_settings = FALSE
)

all2 <- iucnn_best_model(x=combinedfeat2, criterion = "val_acc", require_dropout = FALSE)
summary(all2)
plot(all2)


## PREDICTIONS
# Features generation for species whose red listing categories need to be estimated 
# Getting features for NE species
## Detailed classification

#Features for NE species
choose.files()
pred_occNE<-read.table("C:\\Users\\samuelK\\Desktop\\prediction_occNE.txt",header=TRUE)
attach(pred_occNE2)
names(pred_occNE2)
NEfeatures <- iucnn_prepare_features(pred_occNE,type=c("geographic"),impute_features = FALSE)


#Getting features for DD species
Choose.files()
pred_occDD <- read.table("C:\\Users\\samuelK\\Desktop\\prediction_DD11.txt",header=TRUE)
DDfeatures <-iucnn_prepare_features(pred_occDD,type=c("geographic"),impute_features=FALSE)


## Binary classification 

#Features for NE species
choose.files()
pred_occNE2<-read.table("C:\\Users\\LUC MAS\\samuelK\\prediction_occNE.txt",header=TRUE)
attach(pred_occNE2)
names(pred_occNE2)
NEfeatures2 <- iucnn_prepare_features(pred_occNE2,type=c("geographic","climate"),impute_features = FALSE)


#Getting features for DD species
Choose.files()
pred_occDD2 <- read.table("C:\\Users\\samuelK\\Desktop\\prediction_DD11.txt",header=TRUE)

DDfeatures2 <-iucnn_prepare_features(pred_occDD2,type=c("geographic","climate"),impute_features=FALSE)


#Determination of the conservation status with the best-fitting modelS at two predictions level
#Detailed
#NE species predictions

m_production <- iucnn_train_model(geo_feat,class_labfive,production_model = geobestmodel,overwrite = TRUE)
conservationstatusNE <- iucnn_predict_status(NEfeatures,m_production) 
plot(conservationstatusNE)  

#DD species predictions

conservationstatusDD <- iucnn_predict_status(DDfeatures,m_production)
plot(conservationstatusDD)


#Binary 
#NE species prediction

m_production1 <- iucnn_train_model(allfeat,class_labtw,production_model = all2,overwrite = TRUE)
conservationstatusNE2 <- iucnn_predict_status(NEfeatures2,m_production1) 
plot(conservationstatusNE2)  

#DD species prediction

conservationstatusDD2 <- iucnn_predict_status(DDfeatures2,model=m_production1)
plot(conservationstatusDD2)

# Export results
library(writexl)
write_xlsx(geography,"C:\\Users\\samuelK\\Desktop\\xxx.xlsx")

