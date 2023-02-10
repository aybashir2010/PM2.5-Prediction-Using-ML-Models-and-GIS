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
library(shap)
library(Information)


# Import training and testing data ----
list.files( pattern = "csv$", full.names = TRUE)
original =  read.csv("./Stat_Wi_no_Rd.csv", header = T,stringsAsFactors = FALSE)
road =  read.csv("./Stat_Wi_Rd.csv", header = T)
original$dis_to_road=road$RdDen
original =(na.omit(original))
original =data.frame(original)  # to remove the unwelcomed attributes

#Step 2: Data Cleaning for model preparation
names(original) 
original = original[c("Av_Lev_Win" ,"B2Wi",   "B3Wi" , "B4Wi" ,"B5Wi", 
                       "B6Wi", "BUWi" , "LSTWi", "NDMIWi",
                       "NDVIWi","SAVIWi","Slope")] # remove unnecessary variables
names(original)
original$Av_Lev_Win=as.factor(original$Av_Lev_Win)

#Step 3: Data Normalization
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

names(original)
original.n = as.data.frame(lapply(original[,2:12], normalize)) # Keep the "Av_Lev_Win" variables since it the target variable that needs to be predicted.                     

##### to predict which variable would be the best one for splitting the Decision Tree, plot a graph that represents the split for each of the 11 variables, ####

#Creating seperate dataframe for '"LevelAve" features which is our target.
number.perfect.splits = apply(X=original.n, MARGIN = 2, FUN = function(col){
  t = table(original$Av_Lev_Win,col)
  sum(t == 0)})

# Descending order of perfect splits
order = order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits <- number.perfect.splits[order]

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
myControl = trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=3,
                          returnResamp='all', 
                          allowParallel=TRUE)

#Parameter for Tree Booster
#In the grid, each algorithm parameter can be specified as a vector of possible values . These vectors combine to define all the possible combinations to try.
# We will follow the defaults proposed by https://xgboost.readthedocs.io/en/latest//parameter.html

tune_grid = expand.grid(nrounds = 200,           # the max number of iterations INCREASE THE PROCESSING TIME COST
                         max_depth = 6,            # depth of a tree EFFECTIVE OPTIMIZATION
                         eta = 0.3,               # control the learning rate
                         gamma = 0,             # minimum loss reduction required
                         colsample_bytree = 1,  # subsample ratio of columns when constructing each tree
                         min_child_weight = 1,     # minimum sum of instance weight (hessian) needed in a child 
                         subsample = 1)          # subsample ratio of the training instance

# Step 5 modeling
set.seed(849)
fit.xgb_train= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                        NDVIWi+SAVIWi+Slope, 
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
                          eta = c(0.05,0.3,1),               # control the learning rate
                          gamma = c(0,0.01,0.1),             # minimum loss reduction required
                          colsample_bytree = c(0.75,1),  # subsample ratio of columns when constructing each tree
                          min_child_weight = c(0,1,2),     # minimum sum of instance weight (hessian) needed in a child 
                          subsample = c(0.5,1))            # subsample ratio of the training instance

set.seed(849)
fit.xgb_train2= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                         NDVIWi+SAVIWi+Slope,  
                       data=train.data,
                       method = "xgbTree",
                       metric= "Accuracy",
                       preProc = c("center", "scale"), 
                       trControl = myControl,
                       tuneGrid = tune_grid2,
                       tuneLength = 10)

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
                          gamma = c(0.01),            # minimum loss reduction required
                          colsample_bytree = c(1),  # subsample ratio of columns when constructing each tree
                          min_child_weight = c(0),     # minimum sum of instance weight (hessian) needed in a child 
                          subsample = c(0.5))            # subsample ratio of the training instance

set.seed(849)
fit.xgb_train3= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                         NDVIWi+SAVIWi+Slope,
                       data=train.data,
                       method = "xgbTree",
                       metric= "Accuracy",
                       preProc = c("center", "scale"), 
                       trControl = myControl,
                       tuneGrid = tune_grid3,
                       tuneLength = 10,
                       importance = TRUE)


fit.xgb_train3$results
xgb.plot.importance(fit.xgb_train3$finalModel)
X.xgb_win = varImp(fit.xgb_train3)
xgb.plot.importance(X.xgb_win)
plot(X.xgb_win)
plot(fit.xgb_train3)

# Extract variable importance from the trained model
var_imp = varImp(fit.xgb_train3, scale = TRUE)

# Create a bar chart of variable importance
ggplot(var_imp, aes(x = reorder(Var2, Importance), y = Importance)) + 
  geom_bar(stat = "identity") + 
  xlab("Variable") + ylab("Importance") + 
  ggtitle("Variable Importance from XGBoost Model Winter") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Confusion Matrix - train data
p2_xgb_train3=predict(fit.xgb_train3, test.data, type = "raw")
confusionMatrix(p2_xgb_train3, as.factor(test.data_labels)) 

# Plot ROC curves
# https://stackoverflow.com/questions/46124424/how-can-i-draw-a-roc-curve-for-a-randomforest-model-with-three-classes-in-r
#install.packages("pROC")
library(pROC)


# the model is used to predict the test data. However, you should ask for type="prob" here
predictions_XGB = as.data.frame(predict(fit.xgb_train3, test.data, type = "prob"))

##  Since you have probabilities, use them to get the most-likely class.
# predict class and then attach test class
predictions_XGB$predict = names(predictions_XGB)[1:2][apply(predictions_XGB[,1:2], 1, which.max)]
predictions_XGB$observed = test.data_labels
head(predictions_XGB)

#    Now, let's see how to plot the ROC curves. For each class, convert the multi-class problem into a binary problem. Also, 
#    call the roc() function specifying 2 arguments: i) observed classes and ii) class probability (instead of predicted class).
# 1 ROC curve,  UHeal,  UHealSen vs non  UHeal non  UHealSen
roc.UHeal = roc(ifelse(predictions_XGB$observed== "UHeal", "UHeal", "non-UHeal"), as.numeric(predictions_XGB$UHeal))
roc.UHealSen = roc(ifelse(predictions_XGB$observed=="UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_XGB$UHealSen))
roc_XGB = roc(ifelse(predictions_XGB$observed=="UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_XGB$UHealSen))

plot(roc.UHeal, col = "#FF9900", main="XGBoost best tune prediction ROC plot using testing data", xlim=c(0.64,0.1))
lines(roc.UHealSen, col = "red")
plot(roc_XGB)



#Train xgbTree model USING aLL dependent data
#We will use the train() function from the of caret package with the "method" parameter "xgbTree" wrapped from the XGBoost package.

original.n$PM=original$Av_Lev_Win
set.seed(849)
fit.xgbAll= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                     NDVIWi+SAVIWi+Slope, 
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


# Plot graph
plot(X.xgbAll,main="varImportance All XB" )
# 3. Close the file
dev.off()

# 6  Produce prediction map using Raster data --------------
#Produce prediction map using Trained model results and Raster layers data

# Load the Raster data
list.files( "C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images",pattern = "tif$", full.names = TRUE)
B2Wi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B2Wi.tif" )
names(B2Wi) = "B2Wi"
B3Wi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B3Wi.tif" )
names(B3Wi) = "B3Wi"
B4Wi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B4Wi.tif")
names(B4Wi) = "B4Wi"
B5Wi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B5Wi.tif")
names(B5Wi) = "B5Wi"
B6Wi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/B6Wi.tif")
names(B6Wi) = "B6Wi"
BUWi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/BUWi.tif")
names(BUWi) = "BUWi"
DEM = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/DEM.tif")
LSTWi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/LSTWi.tif")
names(LSTWi) = "LSTWi"
NDBIWi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDBIWi.tif")
names(NDBIWi) = "NDBIWi"
NDMIWi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDMIWi.tif")
names(NDMIWi) = "NDMIWi"
NDVIWi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/NDVIWi.tif")
names(NDVIWi) = "NDVIWi"
distoroad = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/RdDen.tif")
SAVIWi = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/SAVIWi.tif")
names(SAVIWi) = "SAVIWi"
Slope = raster("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Slope.tif")
names(Slope) = "Slope"
Study_area = shapefile("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/CASESTUDY/BosniaP.shp")
#dis_to_road = mask(crop(distoroad, Study_area), Study_area)


# stack multiple raster files

Rasters= stack(B2Wi,B3Wi,B4Wi,B5Wi,B6Wi,BUWi,LSTWi,NDMIWi,
                  NDVIWi,SAVIWi,Slope)
plot(Rasters$B2Wi)
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


# PRODUCE PROBABILITY MAP
p3<-as.data.frame(predict(fit.xgbAll, Rasters.df_N_Nor, type = "prob"))
summary(p3)
Rasters.df$Levels_UHeal=p3$UHeal
Rasters.df$Levels_UHealSen=p3$UHealSen

x=SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_UHeal = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHeal")])
proj4string(r_ave_UHeal)=CRS(projection(NDVIWi))

r_ave_UHealSen = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHealSen")])
proj4string(r_ave_UHealSen)=CRS(projection(NDVIWi))

# Plot Maps
spplot(r_ave_UHeal, main="UHeal XGB")
writeRaster(r_ave_UHeal,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHeal_XGB_Win.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_UHealSen, main="UHealSen XGB")
writeRaster(r_ave_UHealSen,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHealSen_XGB_Win.tif", format="GTiff", overwrite=TRUE) 

# PRODUCE CLASSIFICATION MAP
#Prediction at grid location
p3=as.data.frame(predict(fit.xgbAll, Rasters.df_N_Nor, type = "raw"))
summary(p3)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_ave=p3$`predict(fit.xgbAll, Rasters.df_N_Nor, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID=read.csv("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images//Levels_key.csv", header = T)

# Join ID
grid.new=join(Rasters.df, ID, by="Levels_ave", type="inner") 
# Omit missing values
grid.new.na=na.omit(grid.new)    
head(grid.new.na, n=2)

#Convert to raster
x=SpatialPointsDataFrame(as.data.frame(grid.new.na)[, c("x", "y")], data = grid.new.na)
r_ave = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# coord. ref. : NA 
# Add coord. ref. system by using the original data info (Copy n Paste).
# borrow the projection from Raster data
proj4string(r_ave)=CRS(projection(NDVIWi)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
writeRaster(r_ave,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Classification_Map XGB Winter.tif", format="GTiff", overwrite=TRUE) 


###### Naive Bayes Method ##############
# Step 1 Run the modelling
# Using default prameters
myControl = trainControl(method="repeatedcv", 
                          number=10, 
                          repeats=3,
                          returnResamp='all', 
                          allowParallel=TRUE)


#Train Na?ve Bayes model
#We will use the train() function of the caret package with the "method" parameter "nb" wrapped from the e1071 package.
set.seed(849)
fit.nb_def = train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                      NDVIWi+SAVIWi+Slope,  
                    data=train.data,
                    method = "nb",
                    metric= "Accuracy",
                    preProc = c("center", "scale"),
                    trControl = myControl)
fit.nb_def$resample 
X.nb_win  = varImp(fit.nb_def)
plot(X.nb_win)

# Plot graph

# Extract variable importance from the trained model
var_imp = varImp(fit.xgb_train3, scale = TRUE)

# Create a bar chart of variable importance
ggplot(var_imp, aes(x = reorder(Var2, Importance), y = Importance)) + 
  geom_bar(stat = "identity") + 
  xlab("Variable") + ylab("Importance") + 
  ggtitle("Variable Importance from XGBoost Model") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


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
fit.nb = train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                  NDVIWi+SAVIWi+Slope,
                data=train.data,
                method = "nb",
                tuneGrid=tune_gridNaive,
                metric= "Accuracy",
                preProc = c("center", "scale"), 
                trControl = myControl,
                importance = TRUE)
fit.nb$results 
summaryRes=fit.nb$results # nrounds was fixed = 210
head(summaryRes)
summary(summaryRes)
head(summaryRes[order(summaryRes$Accuracy, decreasing = TRUE),],n=6)  # sort max to min for first 5 values based on Accuracy


## using best tunned hyperparameters
tune_gridNaive2 = expand.grid(fL= c(0) ,             # (Laplace Correction)
                               usekernel= TRUE,                     #(Distribution Type)
                               adjust= c(1.0)              #(Bandwidth Adjustment)
)

#Train Na?ve Bayes model
#We will use the train() function of the caret package with the "method" parameter "nb" wrapped from the e1071 package.
set.seed(849)
fit.nb2= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                   NDVIWi+SAVIWi+Slope,
                 data=train.data,
                 method = "nb",
                 tuneGrid=tune_gridNaive2,
                 metric= "Accuracy",
                 preProc = c("center", "scale"), 
                 trControl = myControl,
                 importance = TRUE)
fit.nb2$results 
# Extract variable importance from the trained model
var_imp = varImp(fit.nb2, scale = TRUE)
# Create a bar chart of variable importance
ggplot(var_imp, aes(x = reorder(Var2, Importance), y = Importance)) + 
  geom_bar(stat = "identity") + 
  xlab("Variable") + ylab("Importance") + 
  ggtitle("Variable Importance from NB Model Winter") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Confusion Matrix - train data
p1=predict(fit.nb2, test.data, type = "raw")
confusionMatrix(p1, test.data_labels)   


# Plot ROC curves
#install.packages("pROC")
library(pROC)

# the model is used to predict the test data. However, you should ask for type="prob" here
predictions_nb = as.data.frame(predict(fit.nb2, test.data, type = "prob"))
predictions_nb$predict = names(predictions_nb)[1:2][apply(predictions_nb[,1:2], 1, which.max)]
predictions_nb$observed = test.data_labels
head(predictions_nb)


roc.UHeal = roc(ifelse(predictions_nb$observed== "UHeal", "UHeal", "non-UHeal"), as.numeric(predictions_nb$UHeal))
roc.UHealSen = roc(ifelse(predictions_nb$observed=="UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_nb$UHealSen))
roc_NB = roc(ifelse(predictions_nb$observed== "UHeal", "UHeal", "non-UHeal"), as.numeric(predictions_nb$UHeal))
plot(roc_NB)

plot(roc.UHeal, col = "#FF9900", main="Naive Bayes best tune prediction ROC plot using testing data", xlim=c(0.64,0.1))
par(pty='s')
plot(roc.UHeal, xlab='False Positive Percentage', ylab='True Positive Percentage', col='#377eb8', lwd=4)
lines(roc.UHealSen, col = "red")


# calculating the values of AUC for ROC curve
results= c("UHealSen AUC" = roc.UHealSen$auc,"UHeal AUC" = roc.UHeal$auc)
print(results)
legend("topleft",c("UHealSen AUC = 0.82 ","UHeal AUC = 0.82"),fill=c("red","#FF9900"),inset = (0.42))

# Step 5 Train nb model USING aLL data
original.n$PM=original$Av_Lev_Win

set.seed(849)
fit.nbAll= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                    NDVIWi+SAVIWi+Slope,
                  data=original.n,
                  method = "nb",
                  metric= "Accuracy",
                  tuneGrid=tune_gridNaive2,
                  preProc = c("center", "scale"), 
                  trControl = myControl)

fit.nbAll$results
X.nbAll = varImp(fit.nbAll)
plot(X.nbAll, main="varImportance All NB tuned")

# Plot graph
# 1. Open jpeg file
jpeg("varImportance All NB.jpg", width = 800, height = 500)
# 2. Create the plot
plot(X.nbAll,main="varImportanceAll NB" )
# 3. Close the file
dev.off()

#### PRODUCE PROBABILITY MAP ####
p3=as.data.frame(predict(fit.nbAll, Rasters.df_N_Nor, type = "prob"))
summary(p3)
Rasters.df$Levels_UHeal=p3$UHeal
Rasters.df$Levels_UHealSen=p3$UHealSen

x=SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_UHeal = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHeal")])
proj4string(r_ave_UHeal)=CRS(projection(SAVIWi))

r_ave_UHealSen = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHealSen")])
proj4string(r_ave_UHealSen)=CRS(projection(SAVIWi))


# Plot Maps
spplot(r_ave_UHeal, main="UHeal NB")
writeRaster(r_ave_UHeal,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHeal_NB_Win.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_UHealSen, main="UHealSen NB")
writeRaster(r_ave_UHealSen,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHealSen_NB_Win", format="GTiff", overwrite=TRUE) 

#### PRODUCE CLASSIFICATION MAP ####

#Prediction at grid location
p3=as.data.frame(predict(fit.nbAll , Rasters.df_N_Nor, type = "raw"))
summary(p3)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_ave=p3$`predict(fit.nbAll, Rasters.df_N_Nor, type = "raw")`
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
r_ave = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# coord. ref. : NA 
# Add coord. ref. system by using the original data info (Copy n Paste).
# borrow the projection from Raster data
proj4string(r_ave)=CRS(projection(SAVIWi)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
writeRaster(r_ave,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Classification_Map NB Winter.tif", format="GTiff", overwrite=TRUE) 

#### KNN Model ########
#default search#####
#Caret can provide for you random parameter if you do not declare for them. 
control = trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3)
param_grid = expand.grid(k = seq(1, 31, by = 2))
set.seed(1)
knn_grid = train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                   NDVIWi+SAVIWi+Slope,
                 data=train.data,
                 method = "knn",
                 trControl = control,
                 tuneGrid = param_grid)
plot(knn_grid, main="KNN with different K values")
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
  ggtitle("Variable Importance from KNN Model Winter") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Evaluate the model
p1_knn_grid=predict(knn_grid, test.data, type = "raw")
confusionMatrix(p1_knn_grid, as.factor(test.data_labels))  # using more deep tree, the accuracy linearly increases! 

#Fitted Parameter

set.seed(1)
param_grid = expand.grid(k=1)
knn_fit = train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                      NDVIWi+SAVIWi+Slope, 
                    data=train.data,
                    method = "knn",
                    trControl = control,
                    tuneGrid = param_grid)
knn_fit
plot(knn_fit)
plot(varImp(knn_fit), main="KNN DEFAULT")

# Evaluate the model
p1_knn_fit=predict(knn_fit, test.data, type = "raw")
confusionMatrix(p1_knn_fit, as.factor(test.data_labels))  # using more deep tree, the accuracy linearly increases! 
# the model is used to predict the test data. However, you should ask for type="prob" here

predictions_KNN = as.data.frame(predict(knn_fit, test.data, type = "prob"))

##  Since you have probabilities, use them to get the most-likely class.
# predict class and then attach test class
predictions_KNN$predict = names(predictions_KNN)[1:2][apply(predictions_KNN[,1:2], 1, which.max)]
predictions_KNN$observed = test.data_labels
head(predictions_KNN)

#pred_KNN = predict(knn_fit, newdata = test.data, type = "prob")[,2]
#pred_NB = predict(fit.nb2, newdata = test.data, type = "prob")[,2]
#pred_XGB = predict(fit.xgb_train3, newdata = test.data, type = "prob")[,2]

#perf_KNN = performance(prediction(pred_KNN, test.data_labels), "tpr", "fpr")
#perf_NB = performance(prediction(pred_NB, test.data_labels), "tpr", "fpr")
#perf_XGB = performance(prediction(pred_XGB, test.data_labels), "tpr", "fpr")

# Plot ROC curves
#plot(perf_KNN, col = "red", lwd = 2, main = "ROC Curves")
#plot(perf_NB, add = T, col = "green", lwd = 2)
#plot(perf_XGB, add = T, col = "blue", lwd = 2)

roc.UHeal = roc(ifelse(predictions_KNN$observed== "UHeal", "UHeal", "non-UHeal"), as.numeric(predictions_KNN$UHeal))
roc.UHealSen = roc(ifelse(predictions_KNN$observed=="UHealSen", "UHealSen", "non-UHealSen"), as.numeric(predictions_KNN$UHealSen))
roc_KNN = roc(ifelse(predictions_KNN$observed== "UHeal", "UHeal", "non-UHeal"), as.numeric(predictions_KNN$UHeal))

plot(roc_KNN, col = "red", lwd = 2, main = "ROC Plot for Winter Season")
plot(roc_XGB, add = T, col = "green", lwd = 2)
plot(roc_NB, add = T, col = "blue", lwd = 2)
# Add AUC values
auc_KNN = auc(roc_KNN)
auc_XGB = auc(roc_XGB)
auc_NB = auc(roc_NB)
legend("bottomright", c(paste("KNN (AUC =", round(auc_KNN, 2),")"),
                      paste("XGB (AUC =", round(auc_XGB, 2),")"),
                     paste("NB (AUC =", round(auc_NB, 2),")")),
            col=c("red", "green", "blue"), lty=1, lwd=2)


# calculating the values of AUC for ROC curve
results= c("UHealSen AUC" = roc.UHealSen$auc,"UHeal AUC" = roc.UHeal$auc)
print(results)
legend("topleft",c("UHealSen AUC = 0.96","UHeal AUC = 0.96"),fill=c("red","#FF9900"),inset = (0.42))


# To produce the map
original.n$PM=original$Av_Lev_Win

set.seed(849)
fit.KNNAll= train(PM~B2Wi+ B3Wi+ B4Wi +B5Wi+B6Wi+ BUWi+ LSTWi+NDMIWi+
                    NDVIWi+SAVIWi+Slope,
                  data=original.n,
                  method = "nb",
                  metric= "Accuracy",
                  tuneGrid=tune_gridNaive2,
                  preProc = c("center", "scale"), 
                  trControl = myControl)

fit.KNNAll$results
X.KNNAll = varImp(fit.KNNAll)
plot(X.KNNAll, main="varImportance All NB tuned")

# Plot graph
# 1. Open jpeg file
jpeg("varImportance All KNN.jpg", width = 800, height = 500)
# 2. Create the plot
plot(X.KNNAll,main="varImportanceAll KNN")
# 3. Close the file
dev.off()

#### PRODUCE PROBABILITY MAP ####
p3=as.data.frame(predict(fit.KNNAll, Rasters.df_N_Nor, type = "prob"))
summary(p3)
Rasters.df$Levels_UHeal=p3$UHeal
Rasters.df$Levels_UHealSen=p3$UHealSen

x=SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_UHeal = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHeal")])
proj4string(r_ave_UHeal)=CRS(projection(SAVIWi))

r_ave_UHealSen = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_UHealSen")])
proj4string(r_ave_UHealSen)=CRS(projection(SAVIWi))


# Plot Maps
spplot(r_ave_UHeal, main="UHeal KNN")
writeRaster(r_ave_UHeal,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHeal_KNN_Win.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_UHealSen, main="UHealSen KNN")
writeRaster(r_ave_UHealSen,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Probability_Map_UHealSen_KNN_Win.tif", format="GTiff", overwrite=TRUE) 




#### PRODUCE CLASSIFICATION MAP ####

#Prediction at grid location
p3=as.data.frame(predict(fit.KNNAll , Rasters.df_N_Nor, type = "raw"))
summary(p3)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_ave=p3$`predict(fit.KNNAll, Rasters.df_N_Nor, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID=read.csv("C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Levels_key.csv", header = T)

# Join ID
grid.new=join(Rasters.df, ID, by="Levels_ave", type="inner") 
# Omit missing values
grid.new.na=na.omit(grid.new)    
head(grid.new.na, n=2)

#Convert to raster
x=SpatialPointsDataFrame(as.data.frame(grid.new.na)[, c("x", "y")], data = grid.new.na)
r_ave = rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# coord. ref. : NA 
# Add coord. ref. system by using the original data info (Copy n Paste).
# borrow the projection from Raster data
proj4string(r_ave)=CRS(projection(SAVIWi)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
writeRaster(r_ave,filename="C:/Users/user/Desktop/TROPHEE/Scientific training/Remote sensing method/Landsat_Images/Classification_Map KNN Winter.tif", format="GTiff", overwrite=TRUE) 


