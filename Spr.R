# Open the folder
path = readClipboard()
setwd("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Bosnia_Landsat")
getwd() # for checking
.libPaths("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Bosnia_Landsat")

sessionInfo()
# load packages
library(xgboost)
library(rgdal)        # spatial data processing
library(raster)       # raster processing
library(plyr)         # data manipulation 
library(dplyr)        # data manipulation 
library(RStoolbox)    # plotting spatial data 
library(RColorBrewer) # color
library(ggplot2)      # plotting
library(sp)           # spatial data
library(caret)        # machine laerning
library(doParallel)   # Parallel processing
library(doSNOW)
library(e1071)
library(GGally)
library(klaR)


# Import training and testing data ----
list.files( pattern = "csv$", full.names = TRUE)
original =  read.csv("./Stat_Removed.csv", header = T,stringsAsFactors = FALSE)
original = (na.omit(original))
original = data.frame(original)  # to remove the unwelcomed attributes


#Step 2: Data Cleaning
names(original) 
original = original[c("Av_Lev_Spr","B2Sp","B3Sp","B4Sp","B5Sp", 
                       "B6Sp","BUSp" , "LSTSp", "NDMISp",
                       "NDVISp","SAVISp","Slope")] # remove unnecessary variables
names(original)
original$Av_Lev_Aut=as.factor(original$Av_Lev_Spr)

#Step 3: Data Normalization
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

names(original)
original.n = as.data.frame(lapply(original[,2:12], normalize)) # Keep the "Av_Lev_Win" variables since it the response variable that needs to be predicted.                     

##### to predict which variable would be the best one for splitting the Decision Tree, plot a graph that represents the split for each of the 13 variables, ####

#Creating seperate dataframe for '"LevelAve" features which is our target.
number.perfect.splits = apply(X=original.n, MARGIN = 2, FUN = function(col){
  t = table(original$Av_Lev_Aut,col)
  sum(t == 0)})

# Descending order of perfect splits
order = order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits = number.perfect.splits[order]

# Plot graph
par(mar=c(10,2,2,2))
barplot(number.perfect.splits,main="Number of perfect splits vs feature",xlab="",
        ylab="Feature",las=3,col="wheat") # BU, NDVI, SAVI and NDMI are the best classifiers

#Step 4: Data Splicing
set.seed(123)
data.d = sample(1:nrow(original),size=nrow(original)*0.70,replace = FALSE) #random selection of 65% data.
train.data = original.n[data.d,] # 70% training data
test.data = original.n[-data.d,] # remaining 30% test data

#Creating seperate dataframe for "Av_Lev_Win" features which is our target.
train.data_labels = original[data.d,1]
test.data_labels =original[-data.d,1]

train.data$PM=train.data_labels


#Tunning prameters
myControl <- trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=3,
                          returnResamp='all', 
                          allowParallel=TRUE)


tune_grid <- expand.grid(nrounds = 200,           # the max number of iterations INCREASE THE PROCESSING TIME COST
                         max_depth = 6,            # depth of a tree EFFECTIVE OPTIMIZATION
                         eta = 0.3,               # control the learning rate
                         gamma = 0,             # minimum loss reduction required
                         colsample_bytree = 1,  # subsample ratio of columns when constructing each tree
                         min_child_weight = 1,     # minimum sum of instance weight (hessian) needed in a child 
                         subsample = 1)          # subsample ratio of the training instance

# Step 5 modeling
set.seed(849)
fit.xgb_train= train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                        NDVISp+SAVISp+Slope, 
                      data=train.data,
                      method = "xgbTree",
                      metric= "Accuracy",
                      preProc = c("center", "scale"), 
                      trControl = myControl,
                      tuneGrid = tune_grid,
                      tuneLength = 10)

fit.xgb_train$resample
fit.xgb_train$modelInfo
X.xgb = varImp(fit.xgb_train)
plot(X.xgb)


#Confusion Matrix - train data
p1<-predict(fit.xgb_train, test.data, type = "raw")
confusionMatrix(p1, as.factor(test.data_labels))  # using more deep tree, the accuracy linearly increases! 

######## Hyperparameter----

tune_grid2 = expand.grid(nrounds = c(200,210),           # the max number of iterations INCREASE THE PROCESSING TIME COST
                          max_depth = c(6,18,22),            # depth of a tree EFFECTIVE OPTIMIZATION
                          eta = c(0.3),               # control the learning rate
                          gamma = c(0.01),             # minimum loss reduction required
                          colsample_bytree = c(0.75),  # subsample ratio of columns when constructing each tree
                          min_child_weight = c(0),     # minimum sum of instance weight (hessian) needed in a child 
                          subsample = c(0.5))           # subsample ratio of the training instance

set.seed(849)
fit.xgb_train2= train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                         NDVISp+SAVISp+Slope,  
                       data=train.data,
                       method = "xgbTree",
                       metric= "Accuracy",
                       preProc = c("center", "scale"), 
                       trControl = myControl,
                       tuneGrid = tune_grid2,
                       tuneLength = 10)
plot(fit.xgb_train2)
summaryRes=fit.xgb_train2$results # nrounds was fixed = 210
head(summaryRes)
summary(summaryRes)
head(summaryRes[order(summaryRes$Accuracy, decreasing = TRUE),],n=6)  # sort max to min for first 5 values based on Accuracy
# Plot
pairs(summaryRes[,c(-7,-9:-11)])
# Save it
write.csv(fit.xgb_train2$results,file = "fit.xgb_train_hyper.csv")#, sep = "",row.names = T)

# Re-run using recomended settings of expand.grid
tune_grid3 = expand.grid(nrounds = c(210),           # the max number of iterations INCREASE THE PROCESSING TIME COST
                          max_depth = c(18),           # depth of a tree EFFECTIVE OPTIMIZATION
                          eta = c(0.3),               # control the learning rate
                          gamma = c(0.1),            # minimum loss reduction required
                          colsample_bytree = c(0.75),  # subsample ratio of columns when constructing each tree
                          min_child_weight = c(0),     # minimum sum of instance weight (hessian) needed in a child 
                          subsample = c(0.5))            # subsample ratio of the training instance

set.seed(849)
fit.xgb_train3= train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                         NDVISp+SAVISp+Slope, 
                       data=train.data,
                       method = "xgbTree",
                       metric= "Accuracy",
                       preProc = c("center", "scale"), 
                       trControl = myControl,
                       tuneGrid = tune_grid3,
                       tuneLength = 10,
                       importance = TRUE)


fit.xgb_train3$results
X.xgb = varImp(fit.xgb_train3)
plot(X.xgb)
plot(fit.xgb_train3)

# Extract variable importance from the trained model
var_imp = varImp(fit.xgb_train3, scale = TRUE)

# Create a bar chart of variable importance
ggplot(var_imp, aes(x = reorder(Var2, Importance), y = Importance)) + 
  geom_bar(stat = "identity") + 
  xlab("Variable") + ylab("Importance") + 
  ggtitle("Variable Importance from XGBoost Model Spring") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Confusion Matrix - train data
p2_xgb_train3=predict(fit.xgb_train3, test.data, type = "raw")
confusionMatrix(p2_xgb_train3, as.factor(test.data_labels)) 

# Plot ROC curves
#install.packages("pROC")
library(pROC)

# the model is used to predict the test data. However, you should ask for type="prob" here
predictions_XGB_Sp = as.data.frame(predict(fit.xgb_train3, test.data, type = "prob"))

# predict class and then attach test class
predictions_XGB_Sp$predict = names(predictions_XGB_Sp)[1:2][apply(predictions_XGB_Sp[,1:2], 1, which.max)]
predictions_XGB_Sp$observed = test.data_labels
head(predictions_XGB_Sp)

# ROC PLOT
roc.Moderate = roc(ifelse(predictions_XGB_Sp$observed== "Moderate", "Moderate", "non-Moderate"), as.numeric(predictions_XGB_Sp$Moderate))
roc.UHealSen = roc(ifelse(predictions_XGB_Sp$observed=="UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_XGB_Sp$UHealSen))
roc.XGB_Sp = roc(ifelse(predictions_XGB_Sp$observed== "Moderate", "Moderate", "non-Moderate"), as.numeric(predictions_XGB_Sp$Moderate))

plot(roc.UHeal, col = "#FF9900", main="XGBoost best tune prediction ROC plot using testing data", xlim=c(0.64,0.1))
lines(roc.UHealSen, col = "red")

plot(roc.XGB_Sp)

#Train xgbTree model USING aLL dependent data

original.n$PM=original$Av_Lev_Aut
set.seed(849)
fit.xgbAll= train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                     NDVISp+SAVISp+Slope,
                   data=original.n,
                   method = "xgbTree",
                   metric= "Accuracy",
                   preProc = c("center", "scale"), 
                   trControl = myControl,
                   tuneGrid = tune_grid3,
                   tuneLength = 10,
                   importance = TRUE)

X.xgbAll = varImp(fit.xgbAll)
plot(X.xgbAll, main="varImportance XB All tunned")



# 6  Produce prediction map using Raster data --------------
#Produce prediction map using Trained model results and Raster layers data

# Load the Raster data
list.files( "C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images",pattern = "tif$", full.names = TRUE)
B2Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B2_Sp.tif" )
names(B2Sp) = "B2Sp"
B3Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B3_Sp.tif" )
names(B3Sp) = "B3Sp"
B4Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B4_Sp.tif")
names(B4Sp) = "B4Sp"
B5Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B5_Sp.tif")
names(B5Sp) = "B5Sp"
B6Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B6_Sp.tif")
names(B6Sp) = "B6Sp"
BUSp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/BU_Sp.tif")
names(BUSp) = "BUSp"
DEM = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/DEM.tif")
LSTSp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/LST_Sp.tif")
names(LSTSp) = "LSTSp"
NDBISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDBI_Sp.tif")
names(NDBISp) = "NDBISp"
NDMISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDMI_Sp.tif")
names(NDMISp) = "NDMISp"
NDVISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDVI_Sp.tif")
names(NDVISp) = "NDVISp"
distoroad = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/RdDen.tif")
SAVISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/SAVI_Sp.tif")
names(SAVISp) = "SAVISp"
Slope = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Slope.tif")
names(Slope) = "Slope"
Study_area =shapefile("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/CASESTUDY/BosniaP.shp")
#dis_to_road <- mask(crop(distoroad, Study_area), Study_area)


# stack multiple raster files

Rasters= stack(B2Sp,B3Sp,B4Sp,B5Sp,B6Sp,BUSp,LSTSp,NDMISp,
                NDVISp,SAVISp,Slope)
plot(Rasters$LSTSp)
names(Rasters)



# 6-1-1 Convert rasters to dataframe with Long-Lat -----------------------

#Convert raster to dataframe with Long-Lat
Rasters.df = as.data.frame(Rasters, xy = TRUE, na.rm = TRUE)
head(Rasters.df,1)
#Rasters.df=Rasters.df[,c(-6)] #

# Now:Prediction using imported Rasters
Rasters.df_N = Rasters.df[,c(-1,-2)] # remove x, y

# Data Normalization
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
# Keep the "LevelAve" variables since ita????s the response variable that needs to be predicted.
names(original)
Rasters.df_N_Nor = as.data.frame(lapply(Rasters.df_N, normalize))    
str(Rasters.df_N_Nor)


#Convert Dataframe back to raster with Long-Lat


# PRODUCE PROBABILITY MAP
p3_xgb=as.data.frame(predict(fit.xgbAll, Rasters.df_N_Nor, type = "prob"))
summary(p3_xgb)
Rasters.df$Levels_Moderate = p3_xgb$Moderate
Rasters.df$Levels_UHealSen = p3_xgb$UHealSen

x=SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_UHealSen = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHealSen")])
proj4string(r_ave_UHealSen)=CRS(projection(NDVISp))

r_ave_Moderate = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_Moderate")])
proj4string(r_ave_Moderate)=CRS(projection(NDVISp))

# Plot Maps
spplot(r_ave_Moderate, main="Moderate XGB")
writeRaster(r_ave_Moderate,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_Moderate_XGB_Spr.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_UHealSen, main="UHealSen XGB")
writeRaster(r_ave_UHealSen,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHealSen_XGB_Spr.tif", format="GTiff", overwrite=TRUE) 


# PRODUCE CLASSIFICATION MAP
#Prediction at grid location
p3_xgb_Spr=as.data.frame(predict(fit.xgbAll, Rasters.df_N_Nor, type = "raw"))
summary(p3_xgb_Spr)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_ave=p3_xgb_Spr$`predict(fit.xgbAll, Rasters.df_N_Nor, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID<-read.csv("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images//Levels_key.csv", header = T)

# Join ID
grid.new_XGB=join(Rasters.df, ID, by="Levels_ave", type="inner") 
# Omit missing values
grid.new_XGB.na=na.omit(grid.new_XGB)    
head(grid.new_XGB.na, n=2)

#Convert to raster
x=SpatialPointsDataFrame(as.data.frame(grid.new_XGB.na)[, c("x", "y")], data = grid.new_XGB.na)
r_ave_XGB <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# borrow the projection from Raster data
proj4string(r_ave_XGB)=CRS(projection(SAVISp)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
spplot(r_ave_XGB)
writeRaster(r_ave_XGB,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Classification_Map XGB Spring.tif", format="GTiff", overwrite=TRUE) 

###### Naive Bayes Method ##############
# Step 1 Run the modelling
# Using default prameters
myControl = trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=3,
                          returnResamp='all', 
                          allowParallel=TRUE)


#Train Na?ve Bayes model
set.seed(849)
fit.nb_def = train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                      NDVISp+SAVISp+Slope,  
                    data=train.data,
                    method = "nb",
                    metric= "Accuracy",
                    preProc = c("center", "scale"),
                    trControl = myControl)
fit.nb_def$resample 
X.nb  = varImp(fit.nb_def)
plot(X.nb )

# Plot graph
plot(X.nb,main="varImportance NB" )
# 3. Close the file
dev.off()

p1=predict(fit.nb_def, test.data, type = "raw")
confusionMatrix(p1, test.data_labels)   # using more deep tree, the accuracy linearly increases! 


#Step 2: Tuning parameters: 

tune_gridNaive = expand.grid(fL= c(0,0.5,1.0) ,             # (Laplace Correction)
                              usekernel= T ,                     #(Distribution Type)
                              adjust= c(0,0.5,1.0)              #(Bandwidth Adjustment)
)

#Train Na?ve Bayes model
#We will use the train() function of the caret package with the "method" parameter "nb" wrapped from the e1071 package.
set.seed(849)
fit.nb = train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                 NDVISp+SAVISp+Slope, 
               data=original.n,
               method = "nb",
               tuneGrid=tune_gridNaive,
               metric= "Accuracy",
               preProc = c("center", "scale"), 
               trControl = myControl,
               importance = TRUE)
plot(fit.nb, main="NB with different K values For Spring")
fit.nb$results 
summaryRes=fit.nb$results #
head(summaryRes)
summary(summaryRes)
head(summaryRes[order(summaryRes$Accuracy, decreasing = TRUE),],n=6)  # sort max to min for first 5 values based on Accuracy


## using best tunned hyperparameters
tune_gridNaive2 = expand.grid(fL= c(0) ,             # (Laplace Correction)
                              usekernel= TRUE,                     #(Distribution Type)
                              adjust= c(1)              #(Bandwidth Adjustment)
)

#Train Na?ve Bayes model
#We will use the train() function of the caret package with the "method" parameter "nb" wrapped from the e1071 package.
set.seed(1)
fit.nb2 = train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                   NDVISp+SAVISp+Slope,  
                 data=original.n,
                 method = "nb",
                 tuneGrid=tune_gridNaive2,
                 metric= "Accuracy",
                 preProc = c("center", "scale"), 
                 trControl = myControl,
                 importance = TRUE)
fit.nb2$results 
fit.nb2 = na.omit(fit.nb2)
# Extract variable importance from the trained model
var_impq = partial(fit.nb2, pred.var = "B2Sp")
# Create a bar chart of variable importance
ggplot(var_imp, aes(x = reorder(Var2, Importance), y = Importance)) + 
  geom_bar(stat = "identity") + 
  xlab("Variable") + ylab("Importance") + 
  ggtitle("Variable Importance from NB Model Autumn") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Confusion Matrix - train data
p2_nb=predict(fit.nb2, test.data, type = "raw")
confusionMatrix(p2_nb, as.factor(test.data_labels))  # using more deep tree, the accuracy linearly increases! 

# the model is used to predict the test data. However, you should ask for type="prob" here
predictions_NB_Sp = as.data.frame(predict(fit.nb2, test.data, type = "prob"))

# predict class and then attach test class
predictions_NB_Sp$predict = names(predictions_NB_Sp)[1:2][apply(predictions_NB_Sp[,1:2], 1, which.max)]
predictions_NB_Sp$observed = test.data_labels
head(predictions_NB_Sp)


# 1 ROC curve, 
roc.UHealSen = roc(ifelse(predictions_NB_Sp$observed== "UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_NB_Sp$UHealSen))
roc.Moderate = roc(ifelse(predictions_NB_Sp$observed=="Moderate", "Moderate", "non-Moderate"), as.numeric(predictions_NB_Sp$Moderate))
roc.NB_Sp = roc(ifelse(predictions_NB_Sp$observed== "UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_NB_Sp$UHealSen))

plot(roc.NB_Sp)

tune_gridNaive2 = expand.grid(fL= c(0) ,             # (Laplace Correction)
                              usekernel= TRUE,                     #(Distribution Type)
                              adjust= c(1)              #(Bandwidth Adjustment)
)
original.n$PM=original$Av_Lev_Spr
set.seed(849)
fit.nbAll= train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                    NDVISp+SAVISp+Slope, 
                  data=original.n,
                  method = "nb",
                  metric= "Accuracy",
                  preProc = c("center", "scale"), 
                  trControl = myControl,
                  tuneGrid = tune_gridNaive2,
                  tuneLength = 10,
                  importance = TRUE)

X.nbAll = varImp(fit.nbAll)
plot(X.nbAll, main="varImportance NB Spring tunned")
fit.nbAll$results

# 6  Produce prediction map using Raster data --------------
#Produce prediction map using Trained model results and Raster layers data

# Load the Raster data
list.files( "C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images",pattern = "tif$", full.names = TRUE)
B2Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B2_Sp.tif" )
names(B2Sp) = "B2Sp"
B3Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B3_Sp.tif" )
names(B3Sp) = "B3Sp"
B4Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B4_Sp.tif")
names(B4Sp) = "B4Sp"
B5Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B5_Sp.tif")
names(B5Sp) = "B5Sp"
B6Sp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B6_Sp.tif")
names(B6Sp) = "B6Sp"
BUSp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/BU_Sp.tif")
names(BUSp) = "BUSp"
DEM = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/DEM.tif")
LSTSp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/LST_Sp.tif")
names(LSTSp) = "LSTSp"
NDBISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDBI_Sp.tif")
names(NDBISp) = "NDBISp"
NDMISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDMI_Sp.tif")
names(NDMISp) = "NDMISp"
NDVISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDVI_Sp.tif")
names(NDVISp) = "NDVISp"
distoroad = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/RdDen.tif")
SAVISp = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/SAVI_Sp.tif")
names(SAVISp) = "SAVISp"
Slope = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Slope.tif")
names(Slope) = "Slope"
Study_area = shapefile("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/CASESTUDY/BosniaP.shp")
#dis_to_road = mask(crop(distoroad, Study_area), Study_area)


# stack multiple raster files

Rasters= stack(B2Sp,B3Sp,B4Sp,B5Sp,B6Sp,BUSp,LSTSp,NDMISp,
                NDVISp,SAVISp,Slope)
plot(Rasters$LSTSp)
names(Rasters)



# 6-1-1 Convert rasters to dataframe with Long-Lat -----------------------

#Convert raster to dataframe with Long-Lat
Rasters.df = as.data.frame(Rasters, xy = TRUE, na.rm = TRUE)
head(Rasters.df,1)
#Rasters.df=Rasters.df[,c(-6)] #

# Now:Prediction using imported Rasters
Rasters.df_N = Rasters.df[,c(-1,-2)] # remove x, y

# Data Normalization
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
# Keep the "LevelAve" variables since ita????s the response variable that needs to be predicted.
names(original)
Rasters.df_N_Nor = as.data.frame(lapply(Rasters.df_N, normalize))    
str(Rasters.df_N_Nor)


#Convert Dataframe back to raster with Long-Lat


# PRODUCE PROBABILITY MAP
p3_nb_spr=as.data.frame(predict(fit.nbAll, Rasters.df_N_Nor, type = "prob"))
summary(p3_nb_spr)
Rasters.df$Levels_Moderate = p3_nb_spr$Moderate
Rasters.df$Levels_UHealSen = p3_nb_spr$UHealSen

x=SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_UHealSen = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHealSen")])
proj4string(r_ave_UHealSen)=CRS(projection(NDVISp))

r_ave_Moderate = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_Moderate")])
proj4string(r_ave_Moderate)=CRS(projection(NDVISp))

# Plot Maps
spplot(r_ave_Moderate, main="Moderate NB")
writeRaster(r_ave_Moderate,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_Moderate_NB_Spr.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_UHealSen, main="UHealSen NB")
writeRaster(r_ave_UHealSen,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHealSen_NB_Spr.tif", format="GTiff", overwrite=TRUE) 


# PRODUCE CLASSIFICATION MAP
#Prediction at grid location
p3_nb_spr=as.data.frame(predict(fit.nbAll, Rasters.df_N_Nor, type = "prob"))
summary(p3)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_ave<-p3_nb_spr$`predict(fit.nbAll, Rasters.df_N_Nor, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID=read.csv("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images//Levels_key.csv", header = T)

# Join landuse ID
grid.new=join(Rasters.df, ID, by="Levels_ave", type="inner") 
# Omit missing values
grid.new.na=na.omit(grid.new)    
head(grid.new.na, n=2)

#Convert to raster
x=SpatialPointsDataFrame(as.data.frame(grid.new.na)[, c("x", "y")], data = grid.new.na)
r_ave_NB = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])


# borrow the projection from Raster data
proj4string(r_ave_NB)=CRS(projection(SAVISp)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
spplot(r_ave_NB)
writeRaster(r_ave,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Classification_Map NB Spring.tif", format="GTiff", overwrite=TRUE) 

#### KNN Model ########
#default search#####
#Caret can provide for you random parameter if you do not declare for them. 
control = trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3)
param_grid = expand.grid(k = seq(1, 31, by = 2))
set.seed(1)
knn_grid = train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                   NDVISp+SAVISp+Slope, 
                 data=train.data,
                 method = "knn",
                 trControl = control,
                 tuneGrid = param_grid)
plot(knn_grid, main="KNN with different K values For Spring")
plot(varImp(knn_grid))
knn_grid$results
summaryRes=knn_grid$results
summary(summaryRes)
head(summaryRes[order(summaryRes$Accuracy, decreasing = TRUE),],n=6)
#Plot Variable Importance
# Extract variable importance from the trained model
var_imp = varImp(knn_grid, scale = TRUE)
# Create a bar chart of variable importance
ggplot(var_imp, aes(x = reorder(Var2, Importance), y = Importance)) + 
  geom_bar(stat = "identity") + 
  xlab("Variable") + ylab("Importance") + 
  ggtitle("Variable Importance from KNN Model Autumn") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
# Evaluate the model
p1_knn_grid<-predict(knn_grid, test.data, type = "raw")
confusionMatrix(p1_knn_grid, as.factor(test.data_labels))


#Fitted Parameter
set.seed(1)
param_grid = expand.grid(k=3)
knn_default = train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                      NDVISp+SAVISp+Slope,
                    data=train.data,
                    method = "knn",
                    trControl = control,
                    tuneGrid = param_grid)
knn_default
plot(knn_default)
plot(varImp(knn_default), main="KNN DEFAULT")

# Evaluate the model
p1_knn_default=predict(knn_default, test.data, type = "raw")
confusionMatrix(p1_knn_default, as.factor(test.data_labels))  # using more deep tree, the accuracy linearly increases! 
# the model is used to predict the test data. However, you should ask for type="prob" here
predictions_KNN_Sp = as.data.frame(predict(knn_default, test.data, type = "prob"))

##  Since you have probabilities, use them to get the most-likely class.
# predict class and then attach test class
predictions_KNN_Sp$predict = names(predictions_KNN_Sp)[1:2][apply(predictions_KNN_Sp[,1:2], 1, which.max)]
predictions_KNN_Sp$observed = test.data_labels
head(predictions_KNN_Sp)


# 1 ROC curve, 
roc.UHealSen = roc(ifelse(predictions_KNN_Sp$observed== "UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_KNN_Sp$UHealSen))
roc.Moderate = roc(ifelse(predictions_KNN_Sp$observed=="Moderate", "Moderate", "non-Moderate"), as.numeric(predictions_KNN_Sp$Moderate))
roc.KNN_Sp = roc(ifelse(predictions_KNN_Sp$observed== "UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_KNN_Sp$UHealSen))

plot(roc.KNN_Sp)
# Plot the ROC for the three model
plot(roc.KNN_Sp, col = "red", lwd = 2, main = "ROC Plot for Spring Season")
plot(roc.XGB_Sp, add = T, col = "green", lwd = 2)
plot(roc.NB_Sp, add = T, col = "blue", lwd = 2)
# Add AUC values
auc_KNN = auc(roc.KNN_Sp)
auc_XGB = auc(roc.XGB_Sp)
auc_NB = auc(roc.NB_Sp)
legend("bottomright", c(paste("KNN (AUC =", round(auc_KNN, 2),")"),
                        paste("XGB (AUC =", round(auc_XGB, 2),")"),
                        paste("NB (AUC =", round(auc_NB, 2),")")),
       col=c("red", "green", "blue"), lty=1, lwd=2)

original.n$PM=original$Av_Lev_Spr

set.seed(849)
k_values = c(1, 3, 5)
tune_gridNaive2 = expand.grid(k = k_values)
fit.KNNAll= train(PM~B2Sp+ B3Sp+ B4Sp +B5Sp+B6Sp+ BUSp+ LSTSp+NDMISp+
                     NDVISp+SAVISp+Slope,
                   data=original.n,
                   method = "knn",
                   metric= "Accuracy",
                   tuneGrid=param_grid,
                   preProc = c("center", "scale"), 
                   trControl = myControl)

fit.KNNAll$results
X.KNNAll = varImp(fit.KNNAll)
plot(X.KNNAll, main="varImportance All NB tuned")

#### PRODUCE PROBABILITY MAP ####
# PRODUCE PROBABILITY MAP
p3_knn_spr=as.data.frame(predict(fit.KNNAll, Rasters.df_N_Nor, type = "prob"))
summary(p3_knn_spr)
Rasters.df$Levels_Moderate = p3_knn_spr$Moderate
Rasters.df$Levels_UHealSen = p3_knn_spr$UHealSen

x=SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_UHealSen = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHealSen")])
proj4string(r_ave_UHealSen)=CRS(projection(NDVISp))

r_ave_Moderate = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_Moderate")])
proj4string(r_ave_Moderate)=CRS(projection(NDVISp))

# Plot Maps
spplot(r_ave_Moderate, main="Moderate KNN")
writeRaster(r_ave_Moderate,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_Moderate_KNN_Spr.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_UHealSen, main="UHealSen KNN")
writeRaster(r_ave_UHealSen,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHealSen_KNN_Spr.tif", format="GTiff", overwrite=TRUE) 


# PRODUCE CLASSIFICATION MAP
#Prediction at grid location
p3_knn_spr=as.data.frame(predict(fit.KNNAll, Rasters.df_N_Nor, type = "prob"))
summary(p3_knn_spr)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_ave=p3_knn_spr$`predict(fit.KNNAll, Rasters.df_N_Nor, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID<-read.csv("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images//Levels_key.csv", header = T)

# Join landuse ID
grid.new=join(Rasters.df, ID, by="Levels_ave", type="inner") 
# Omit missing values
grid.new.na=na.omit(grid.new)    
head(grid.new.na, n=2)

#Convert to raster
x=SpatialPointsDataFrame(as.data.frame(grid.new.na)[, c("x", "y")], data = grid.new.na)
r_ave = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# borrow the projection from Raster data
proj4string(r_ave)=CRS(projection(SAVISp)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
spplot(r_ave)
writeRaster(r_ave,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Classification_Map KNN Spr.tif", format="GTiff", overwrite=TRUE) 
