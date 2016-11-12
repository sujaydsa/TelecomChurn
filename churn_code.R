# Set Working Directory

# Install and Load the required packages
library(ggplot2)
library(Hmisc)
library(MASS)
library(car)
library(e1071)
library(ROCR)
library(caret)
library(dummies)
library(caTools)
library(class)
library(pROC)

# Read the given files
churn_data <- read.csv(file.choose(), stringsAsFactors = F)
customer_data <- read.csv(file.choose(), stringsAsFactors = F)
internet_data <- read.csv(file.choose(), stringsAsFactors = F)

str(churn_data)
str(customer_data)
str(internet_data)

##################################################################
# Create the main Dataset
##################################################################
# Collate the 3 files in a single dataset
churn <- merge(churn_data, customer_data, by.x = 'customerID', by.y = 'customerID')
churn <- merge(churn, internet_data, by.x = 'customerID', by.y = 'customerID')

# Understand the structure of the collated dataset
str(churn)
summary(churn)

############################################################################
# EDA Plots to explore potential interesting relationships between variables
############################################################################
ggplot(churn, aes(MonthlyCharges, fill = Churn)) + geom_histogram()+labs(title="Impact of Monthly Charges on Churn",x ="Monthly Charges", y = "Frequency")
ggplot(churn, aes(tenure, fill = Churn)) + geom_bar()+labs(title="Impact of Tenure on Churn",x ="Tenure", y = "Frequency") #Important
ggplot(churn, aes(PhoneService, fill = Churn)) + geom_bar()+labs(title="Impact of Phone Service on Churn",x ="Phone Service", y = "Frequency")
ggplot(churn, aes(Contract, fill = Churn)) + geom_bar()+labs(title="Impact of Contract type on Churn",x ="Contract", y = "Frequency")   #Important
ggplot(churn, aes(gender, fill = Churn)) + geom_bar()+labs(title="Impact of Gender on Churn",x ="Gender", y = "Frequency")
ggplot(churn, aes(MultipleLines, fill = Churn)) + geom_bar()+labs(title="Impact of Multiple Lines on Churn",x ="Multiple Lines", y = "Frequency")
ggplot(churn, aes(TotalCharges, fill = Churn)) + geom_histogram()+labs(title="Impact of Total Charges on Churn",x ="Total Charges", y = "Frequency")  #Important
ggplot(churn, aes(InternetService, fill = Churn)) + geom_bar()+labs(title="Impact of Internet Service on Churn",x ="Internet Service", y = "Frequency")
ggplot(churn, aes(PaymentMethod, fill = Churn)) + geom_bar()+labs(title="Impact of Payment Method on Churn",x ="Payment Method", y = "Frequency")#Important
ggplot(churn, aes(SeniorCitizen, fill = Churn)) + geom_bar()+labs(title="Impact of Senior Citizen on Churn",x ="Senior Citizen", y = "Frequency")
ggplot(churn, aes(OnlineSecurity, fill = Churn)) + geom_bar()+labs(title="Impact of Online Security",x ="Online Security", y = "Frequency")#Important
ggplot(churn, aes(OnlineBackup, fill = Churn)) + geom_bar()+labs(title="Impact of Online Backup on Churn",x ="Online Backup", y = "Frequency")  #Important
ggplot(churn, aes(DeviceProtection, fill = Churn)) + geom_bar()+labs(title="Impact of Device Protection on Churn",x ="Device Protection", y = "Frequency")
ggplot(churn, aes(TechSupport, fill = Churn)) + geom_bar()+labs(title="Impact of Tech Support on Churn",x ="Tech Support", y = "Frequency")
ggplot(churn, aes(StreamingTV, fill = Churn)) + geom_bar()+labs(title="Impact of Streaming TV on Churn",x ="Streaming TV", y = "Frequency")
ggplot(churn, aes(StreamingMovies, fill = Churn)) + geom_bar()+labs(title="Impact of Streaming Movie on Churn",x ="Streaming Movies", y = "Frequency")


##################################################################
# Make Box plots for numeric variables to look for outliers
##################################################################
boxplot(churn$tenure)
boxplot(churn$MonthlyCharges)
boxplot(churn$TotalCharges)

#NOTE : No Outliers detected from the box plots 

##################################################################
# Perform De-Duplication if required
##################################################################
which(duplicated(churn))
which(duplicated(churn$customerID))

# NOTE: No duplicates are present in the dataset
##################################################################
# Bring the variables in the correct format
##################################################################
str(churn)
# Converting all the character variables to factor type
factor_columns <- sapply(churn, FUN = function(x) { is.character(x) })
churn[factor_columns] <- lapply(churn[factor_columns], factor)
#Converting CustomerID back to character as it is a unique field
churn$customerID <- as.character(churn$customerID)
#Converting Senior Citizen to factor type
churn$SeniorCitizen <- as.factor(churn$SeniorCitizen)

# Impute the missing values, and perform the outlier treatment (if required).
sapply(churn, FUN = function(x) { sum(is.na(x)) })
churn_new <- churn[-which(is.na(churn$TotalCharges)),]
sapply(churn_new, FUN = function(x) { sum(is.na(x)) })
str(churn_new)

##################################################################
# Binning Tenure to years for more convenient/clear analysis
##################################################################
churn_cleaned <- churn_new
churn_cleaned$tenureYears <- 0
summary(churn_cleaned$tenure)

churn_cleaned$tenureYears[which(churn_cleaned$tenure >= 0 & churn_cleaned$tenure <= 12)] <- 1
churn_cleaned$tenureYears[which(churn_cleaned$tenure >= 13 & churn_cleaned$tenure <= 24)] <- 2
churn_cleaned$tenureYears[which(churn_cleaned$tenure >= 25 & churn_cleaned$tenure <= 36)] <- 3
churn_cleaned$tenureYears[which(churn_cleaned$tenure >= 37 & churn_cleaned$tenure <= 48)] <- 4
churn_cleaned$tenureYears[which(churn_cleaned$tenure >= 49 & churn_cleaned$tenure <= 60)] <- 5
churn_cleaned$tenureYears[which(churn_cleaned$tenure >= 61 & churn_cleaned$tenure <= 72)] <- 6

churn_cleaned$tenureYears <- as.factor(churn_cleaned$tenureYears)
str(churn_cleaned)
churn_cleaned <- subset(churn_cleaned, select = -c(tenure))
ggplot(churn_cleaned, aes(tenureYears, fill = Churn)) + geom_bar()+labs(title="Impact of Tenure on Churn",x ="Tenure", y = "Frequency")
churn_cleaned$ChargeRatio<-round(churn_cleaned$TotalCharges/churn_cleaned$MonthlyCharges)
ggplot(churn_cleaned, aes(ChargeRatio, fill = Churn)) + geom_bar()+labs(title="Impact of ChargeRatio on Churn",x ="ChargeRatio", y = "Frequency")
churn_cleaned<-churn_cleaned[,-22]

##################################################################
# K-Nearest Neighbour (K-NN) Model
##################################################################

# Bring the data in the correct format to implement K-NN model.
churn_knn <- churn_cleaned
str(churn_knn)

factor_columns <- sapply(churn_knn , FUN = function(x) { is.factor(x) })
factor_columns <- which(factor_columns==TRUE)
as.data.frame(factor_columns)
#Converting the factors to dummy variables
churn_dummy_knn <- dummy.data.frame(churn_knn, names = c("PhoneService","Contract","PaperlessBilling","PaymentMethod",
                                                     "gender","SeniorCitizen","Partner","Dependents",
                                                     "MultipleLines","InternetService","OnlineSecurity",
                                                     "OnlineBackup","DeviceProtection","TechSupport", "StreamingTV", 
                                                     "StreamingMovies", "tenureYears" ))

str(churn_dummy_knn)

#Scaling the numeric values
churn_dummy_knn$MonthlyCharges <- scale(churn_dummy_knn$MonthlyCharges)
churn_dummy_knn$TotalCharges <- scale(churn_dummy_knn$TotalCharges)
str(churn_dummy_knn)

# Implement the K-NN model for optimal K.
set.seed(100)
rows <- sample.split(churn_knn$Churn,SplitRatio = 0.7)
train_knn <- churn_dummy_knn[rows,]
test_knn <- churn_dummy_knn[!rows,]

train_knn <- subset(train_knn, select = -c(customerID))
test_knn <- subset(test_knn, select = -c(customerID))

test_churn_knn <- test_knn$Churn
cl <- train_knn[,"Churn"]
str(train_knn)
str(test_knn)
train_knn1 <- subset(train_knn, select = -c(Churn))
test_knn1 <- subset(test_knn, select = -c(Churn))

knn_3 <- knn(train = train_knn1, test = test_knn1, cl = cl, k = 3, prob = T)

table(knn_3, test_knn$Churn)
confusionMatrix(knn_3, test_knn$Churn, positive="Yes")

# Finding optimum K Value
Knn_model <- train(
  Churn~., 
  data=train_knn,
  method='knn',
  tuneGrid=expand.grid(.k=1:15),
  metric='Accuracy',
  trControl=trainControl(
    method='repeatedcv', 
    number=10, 
    repeats=10))

Knn_model

# Final model is built with a optimum k value of 13
knn_13 <- knn(train = train_knn1, test = test_knn1, cl = cl, k = 13, prob = T)
table(knn_13, test_churn_knn)
confusionMatrix(knn_13, test_churn_knn, positive="Yes")

attr(knn_13,"prob") <- ifelse(knn_13==1,attr(knn_13,"prob"),1 - attr(knn_13,"prob"))

knn_model_score_test <- prediction(predictions = attr(knn_13, "prob"), labels = test_churn_knn)
knn_model_perf_test <- performance(knn_model_score_test, "tpr", "fpr")
knn_auc <- performance(knn_model_score_test, "auc")
knn_auc # AUc Value of 0.7051

plot(knn_model_perf_test, main="KNN AUC")


##################################################################
# Naive Bayes Model
##################################################################

# Bring the data in the correct format to implement Naive Bayes algorithm.

churn_naive <- churn_cleaned
set.seed(100)
rows <- sample.split(churn_knn$Churn,SplitRatio = 0.7)
#Creating dummy variables for the factor type
churn_dummy_nb <- dummy.data.frame(churn_naive, names = c("PhoneService","Contract","PaperlessBilling","PaymentMethod",
                                                         "gender","SeniorCitizen","Partner","Dependents",
                                                         "MultipleLines","InternetService","OnlineSecurity",
                                                         "OnlineBackup","DeviceProtection","TechSupport", "StreamingTV", 
                                                         "StreamingMovies", "tenureYears" ))
# Scaling the numeric values
churn_dummy_nb$MonthlyCharges <- scale(churn_dummy_nb$MonthlyCharges)
churn_dummy_nb$TotalCharges <- scale(churn_dummy_nb$TotalCharges)

train_nb <- churn_dummy_nb[rows,]
test_nb <- churn_dummy_nb[!rows,]

test_churn_nb <- test_nb$Churn

train_nb <- subset(train_nb, select = -c(customerID))
test_nb <- subset(test_nb, select = -c(Churn,customerID))

# Implement the Naive Bayes algorithm.
model <- naiveBayes(formula = Churn ~., data = train_nb)
test_nb$Churn <- predict(model, newdata = test_nb)

nb_predprob <- predict(model, newdata = test_nb, type = "raw")
pred_prob_nb <- nb_predprob[,2]
real_values <- ifelse(test_churn_nb=="Yes",1,0)
confusionMatrix(data = test_nb$Churn,reference =  test_churn_nb, positive = "Yes")

nb_model_score_test <- prediction(pred_prob_nb,real_values)
nb_model_perf_test <- performance(nb_model_score_test, "tpr", "fpr")
nb_auc <- performance(nb_model_score_test, "auc")
nb_auc # AUC is 0.822
plot(nb_model_perf_test, main="Naive-Bayes ROC")

##################################################################
# Logistic Regression Model
##################################################################

# Bring the data in the correct format to implement Logistic regression model.

churn_lr <- churn_cleaned
#Creating dummy variables for the factors
churn_dummy_lr <- dummy.data.frame(churn_lr, names = c("PhoneService","Contract","PaperlessBilling","PaymentMethod",
                                                   "gender","SeniorCitizen","Partner","Dependents",
                                                   "MultipleLines","InternetService","OnlineSecurity",
                                                   "OnlineBackup","DeviceProtection","TechSupport", "StreamingTV", 
                                                   "StreamingMovies","tenureYears" ))
# Scaling the numeric values
churn_dummy_lr$MonthlyCharges <- scale(churn_dummy_lr$MonthlyCharges)
churn_dummy_lr$TotalCharges <- scale(churn_dummy_lr$TotalCharges)

# Split the dataset into into train and test
set.seed(100)
rows <- sample.split(churn_knn$Churn,SplitRatio = 0.7)
train_lr <- churn_dummy_lr[rows,]
test_lr <- churn_dummy_lr[!rows,]

train_lr <- subset(train_lr, select = -c(customerID))
test_lr <- subset(test_lr, select = -c(customerID))

initial_model = glm(Churn ~ ., data = train_lr, family = "binomial")
summary(initial_model)

# Implement the Logistic regression algorithm and use stepwise selection to select final variables

best_model <- step(initial_model, direction = "both")
summary(best_model)
sort(vif(best_model), decreasing = T)

# Select the variables using VIF criterion. 
# Eliminate PhoneServiceNo
model_2 <- glm(formula = Churn ~ `ContractMonth-to-month` + 
                 `ContractOne year` + PaperlessBillingNo + `PaymentMethodElectronic check` + 
                 MonthlyCharges + TotalCharges + SeniorCitizen0 + MultipleLinesNo + 
                 InternetServiceDSL + `InternetServiceFiber optic` + OnlineBackupNo + 
                 DeviceProtectionNo + StreamingTVNo + StreamingMoviesNo + 
                 tenureYears1 + tenureYears2 + tenureYears4, family = "binomial", 
                 data = train_lr)
summary(model_2)
sort(vif(model_2), decreasing = T)

# Eliminate TotalCharges
model_3 <- glm(formula = Churn ~ `ContractMonth-to-month` + 
                 `ContractOne year` + PaperlessBillingNo + `PaymentMethodElectronic check` + 
                 MonthlyCharges + SeniorCitizen0 + MultipleLinesNo + 
                 InternetServiceDSL + `InternetServiceFiber optic` + OnlineBackupNo + 
                 DeviceProtectionNo + StreamingTVNo + StreamingMoviesNo + 
                 tenureYears1 + tenureYears2 + tenureYears4, family = "binomial", 
                 data = train_lr)
summary(model_3)
sort(vif(model_3), decreasing = T)

# Eliminate MonthlyCharges
model_4 <- glm(formula = Churn ~ `ContractMonth-to-month` + 
                 `ContractOne year` + PaperlessBillingNo + `PaymentMethodElectronic check` + 
                 SeniorCitizen0 + MultipleLinesNo + 
                 InternetServiceDSL + `InternetServiceFiber optic` + OnlineBackupNo + 
                 DeviceProtectionNo + StreamingTVNo + StreamingMoviesNo + 
                 tenureYears1 + tenureYears2 + tenureYears4, family = "binomial", 
                 data = train_lr)
summary(model_4)
sort(vif(model_4), decreasing = T)


# Eliminate DeviceProtectionNo
model_5 <- glm(formula = Churn ~ `ContractMonth-to-month` + 
                 `ContractOne year` + PaperlessBillingNo + `PaymentMethodElectronic check` + 
                 SeniorCitizen0 + MultipleLinesNo + 
                 InternetServiceDSL + `InternetServiceFiber optic` + OnlineBackupNo + 
                 StreamingTVNo + StreamingMoviesNo + 
                 tenureYears1 + tenureYears2 + tenureYears4, family = "binomial", 
                data = train_lr)
summary(model_5)
sort(vif(model_5), decreasing = T)

# Eliminate OnlineBackupNo
model_6 <- glm(formula = Churn ~ `ContractMonth-to-month` + 
                 `ContractOne year` + PaperlessBillingNo + `PaymentMethodElectronic check` + 
                 SeniorCitizen0 + MultipleLinesNo + 
                 InternetServiceDSL + `InternetServiceFiber optic` + 
                 StreamingTVNo + StreamingMoviesNo + 
                 tenureYears1 + tenureYears2 + tenureYears4, family = "binomial", 
                 data = train_lr)
summary(model_6)
sort(vif(model_6), decreasing = T)

# Make the final Logistic Regression model.
# The final model is the model obtained from the step function
model_final <- model_6

# Find c-statistic and KS -statistic
train_lr$predicted_prob <- predict(model_final,  type = "response")
rcorr.cens(train_lr$predicted_prob,train_lr$Churn) 
# C statisitic is 0.84 which suggests that the model has a high discriminative power.

test_lr$predicted_prob = predict(model_final, newdata = test_lr,type = "response")
rcorr.cens(test_lr$predicted_prob,test_lr$Churn)
# C statisitic is 0.84 which suggests that the model has a high discriminative power.

model_score <- prediction(train_lr$predicted_prob,train_lr$Churn)
model_perf <- performance(model_score, "tpr", "fpr")
plot(model_perf)
ks_table <- attr(model_perf, "y.values")[[1]] - (attr(model_perf, "x.values")[[1]])
ks = max(ks_table)
ks
which(ks_table == ks)
319/4922 
# ***************************************************************************
# KS Statistic for train data comes out to be 0.06 
# This suggests that the ks-stat lies in the 1st decile itself which implies 
# that the model has good discriminative power
# ***************************************************************************

model_score_test <- prediction(test_lr$predicted_prob,test_lr$Churn)
model_perf_test <- performance(model_score_test, "tpr", "fpr")
lr_auc <- performance(model_score_test, "auc")
lr_auc # AUC is 0.84
plot(model_perf_test)
ks_table_test <- attr(model_perf_test, "y.values")[[1]] - (attr(model_perf_test, "x.values")[[1]])
ks_test <- max(ks_table_test)
ks_test
0.52
which(ks_table_test == ks_test)
195/2110 
# ********************************************************************************
# KS Statistic for test data comes out to be 0.092 
# This suggests that the ks-stat lies in the 1st decile itself which again implies 
# that the model has a good discriminative power
# ********************************************************************************

# Selecting Threshold Value
test_results.roc <- roc(test_lr$Churn,test_lr$predicted_prob)
train_results.roc <- roc(train_lr$Churn,train_lr$predicted_prob)
plot(train_results.roc, print.thres="best", print.thres.best.method="closest.topleft")
plot(test_results.roc, print.thres="best", print.thres.best.method="closest.topleft")

# The threshold value from the train dataset is chosen. 

# Confusion Matrix
train_lr$predicted_churn <- "No"
train_lr$predicted_churn[which(train_lr$predicted_prob > 0.294)] <- "Yes"

test_lr$predicted_churn <- "No"
test_lr$predicted_churn[which(test_lr$predicted_prob > 0.294)] <- "Yes"

confusionMatrix(data = train_lr$predicted_churn,reference = train_lr$Churn, positive = "Yes")
confusionMatrix(data = test_lr$predicted_churn,reference = test_lr$Churn, positive = "Yes")

##################################################################
# SVM Model
##################################################################

# Bring the data in the correct format to implement the SVM algorithm.
churn_svm <- churn_cleaned
churn_dummy_svm <- dummy.data.frame(churn_svm, names = c("PhoneService","Contract","PaperlessBilling","PaymentMethod",
                                                   "gender","SeniorCitizen","Partner","Dependents",
                                                   "MultipleLines","InternetService","OnlineSecurity",
                                                   "OnlineBackup","DeviceProtection","TechSupport", "StreamingTV", 
                                                   "StreamingMovies", "tenureYears" ))
# Scaling the numeric values
churn_dummy_svm$MonthlyCharges <- scale(churn_dummy_svm$MonthlyCharges)
churn_dummy_svm$TotalCharges <- scale(churn_dummy_svm$TotalCharges)

set.seed(100)
rows <- sample.split(churn_dummy_svm$Churn,SplitRatio = 0.7)
train_svm <- churn_dummy_svm[rows,]
test_svm <- churn_dummy_svm[!rows,]

train_svm <- subset(train_svm, select = -c(customerID))
test_svm <- subset(test_svm, select = -c(customerID))

churn_test <- test_svm$Churn

# Implement the SVM algorithm using the optimal cost.

model.svm.0 = svm(Churn ~., data = train_svm, kernel = "linear", cost = 0.1, scale = F)  
summary(model.svm.0)
# TUning the SVM Model to find the optimum cost value
tune.svm = tune(svm, Churn ~., data = train_svm, kernel = "linear", 
                ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 1, 10, 100)))
best.mod = tune.svm$best.model
best.mod

# Best Model obtained after tuning
model.svm.1 = svm(Churn ~., data = train_svm, kernel = "linear", cost = 0.01, scale = F)  
summary(model.svm.1)

best.mod <- model.svm.1
# Predict the Churn value from the model
ypred <- predict(best.mod, test_svm, type="prob")
table(predicted <- ypred, truth = test_svm$Churn)
confusionMatrix(ypred, test_svm$Churn, positive = "Yes")

# Changing the Positive Class to Yes
attr(ypred,"prob") <- ifelse(ypred==1,attr(ypred,"prob"),1 - attr(ypred,"prob"))
# Performance Metrics
svm_model_score_test <- prediction(predictions = attr(knn_13, "prob"), labels = test_churn_knn)
svm_model_perf_test <- performance(svm_model_score_test, "tpr", "fpr")
svm_auc <- performance(svm_model_score_test, "auc")
svm_auc # Auc is 0.70
plot(svm_model_perf_test, main="SVM AUC")
