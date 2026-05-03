# Analytics assignment
# ---- libraries ----
library(caret)
library(glmnet)
library(glmnetUtils)
library(plotmo)
library(knitr)
library(ranger)
library(dplyr)
library(stringr)
library(randomForest)
library(gbm)
library(pROC)
library(ggplot2)
library(vip)
library(patchwork)
library(pdp)
library(gridExtra)
library(lubridate)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ROCR)

data <- read.csv("listings.csv")
des <- read.csv("variable_descriptions.csv")
# selecting features for model
features <- c("host_listings_count", "host_since", "host_acceptance_rate", "host_is_superhost", "neighbourhood_ward", "room_type", "accommodates", "bedrooms", 
              "price", "availability_30", "number_of_reviews")

set.seed(7154)
n <- nrow(data)
train_index <- createDataPartition(data[, 1], p = 0.8, list = FALSE)
train <- data[train_index, features]
test <- data[-train_index, features]

quantile(train$host_listings_count, c(0.25, 0.5, 0.75))

hist(train$host_listings_count, breaks = 30)
abline(v = median(train$host_listings_count), col = "red", lty = 2)
abline(v = mean(train$host_listings_count), col = "blue", lty = 3)

mean(train$host_listings_count)
min(train$host_listings_count)
max(train$host_listings_count)
# ---- PRELIMINARY DATA ANALYSIS ----
data <- read.csv("listings.csv")
des <- read.csv("variable_descriptions.csv")

# selecting features for model
features <- c("host_listings_count", "host_since", "host_acceptance_rate", "host_is_superhost", "neighbourhood_ward", "room_type", "accommodates", "bedrooms", 
              "price", "availability_30", "number_of_reviews")

# splitting data into training and test set (80:20)
set.seed(7154)
n <- nrow(data)
train_index <- createDataPartition(data[, 1], p = 0.8, list = FALSE)
train <- data[train_index, features]
test <- data[-train_index, features]

# determining quantiles
quantile(train$host_listings_count, c(0.25, 0.5, 0.75))

# graph showing distribution of host listings count
hist(train$host_listings_count, breaks = 30)
abline(v = median(train$host_listings_count), col = "red", lty = 2)
abline(v = mean(train$host_listings_count), col = "blue", lty = 3)

# other notable values
mean(train$host_listings_count)
min(train$host_listings_count)
max(train$host_listings_count)
# ---- QUESTION 1 ----

# making highhost variable for train and test (x = 8)
highhost1 <- ifelse(train$host_listings_count > 8, 1, 0)
highhost2 <- ifelse(test$host_listings_count > 8, 1, 0)
sum(highhost1)

# replacing host_listings_count with highhost
train$host_listings_count <- factor(highhost1, levels = c(0,1))
colnames(train)[1] <- "highhost"

test$host_listings_count <- factor(highhost2, levels = c(0,1))
colnames(test)[1] <- "highhost"

# converting host_since date to years (in variable host_since_year)

# training set
train$host_since <- as.Date(train$host_since)
host_since_year <- as.integer(format(train$host_since, "%Y"))
train$host_since <- host_since_year
colnames(train)[2] <- "host_since_year"

# test set
test$host_since <- as.Date(test$host_since)
host_since_year <- as.integer(format(test$host_since, "%Y"))
test$host_since <- host_since_year
colnames(test)[2] <- "host_since_year"

head(test)
# creating the response vector and predictor matrices for the training and validation data
train_res <- train[, 1]
train_pred <- train[, -1]
test_res <- test[, 1]
test_pred <- test[, -1]


# QUESTION 2
# ---- regularised elastic-net log reg ----

# converting training data to matrix
train_predmat <- makeX(train_pred)

# defining alpha range
alphas <- seq(0, 0.99, by = 0.01)

# cross-validated elastic-net model:
# set.seed(7154)
# mod_elastic <- lapply(alphas, function(a){
#               cv.glmnet(train_predmat, train_res,
#               family = "binomial",
#               alpha = a,
#               nfolds = 10,
#               type.measure = "class",
#               keep = TRUE)
# })
load("C:/Users/Topollo/Desktop/analytics/assignment one/mod_elastic.Rdata")
# for lambda.min
errors_min  <- sapply(mod_elastic, function(mod) min(mod$cvm))
lambdas_min <- sapply(mod_elastic, function(mod) mod$lambda.min)

best_index_min <- which.min(errors_min)
best_model_min <- mod_elastic[[best_index_min]]
best_alpha_min <- alphas[best_index_min]
best_lambda_min <- best_model_min$lambda.min
best_error_min <- errors_min[best_index_min]

## for lambda.1se
errors_1se <- sapply(mod_elastic, function(mod) {
  mod$cvm[mod$lambda == mod$lambda.1se]
})
lambdas_1se <- sapply(mod_elastic, function(mod) mod$lambda.1se)

best_index_1se <- which.min(errors_1se)
best_model_1se <- mod_elastic[[best_index_1se]]
best_alpha_1se <- alphas[best_index_1se]
best_lambda_1se <- lambdas_1se[best_index_1se]
best_error_1se <- errors_1se[best_index_1se]

# COMBINED PLOTS
# convert errors to accuracy
accuracy_min <- 1 - errors_min
accuracy_1se <- 1 - errors_1se

y_range <- range(c(accuracy_min, accuracy_1se))

plot(alphas, accuracy_min,
     type = "b", pch = 16, col = "firebrick",
     ylim = y_range,
     xlab = expression(alpha),
     ylab = "Accuracy (Cross Validation)",
     main = "Elastic-net: CV Accuracy Across Alpha")
lines(alphas, accuracy_1se,
      type = "b", pch = 16, col = "purple")

# lambda.min optimal alpha
abline(v = best_alpha_min, lty = 2, col = "firebrick", lwd = 2)
points(best_alpha_min, accuracy_min[best_index_min], pch = 16, col = "firebrick")
text(best_alpha_min, accuracy_min[best_index_min],
     labels = paste0("alpha = ", best_alpha_min),
     pos = 4, cex = 0.85, col = "firebrick")

# lambda.1se optimal alpha
abline(v = best_alpha_1se, lty = 2, col = "purple", lwd = 2)
points(best_alpha_1se, accuracy_1se[best_index_1se], pch = 16, col = "purple")
text(best_alpha_1se, accuracy_1se[best_index_1se],
     labels = paste0("alpha = ", best_alpha_1se),
     pos = 2, cex = 0.85, col = "purple")

legend("bottomright",
       legend = c("lambda.min", "lambda.1se"),
       col = c("firebrick", "purple"),
       lty = 1, pch = c(16, 16), cex = 0.85, bty = "n")

# finding best lambda.1se index, extracting predictions from training
lambda_index_1se <- which.min(abs(best_model_1se$lambda - best_lambda_1se))
probs_1se <- best_model_1se$fit.preval[, lambda_index_1se]
preds_1se <- factor(ifelse(probs_1se >= 0.5, "1", "0"), levels = levels(train_res))

# confusion matrix for elastic-net (1se)
confusion_elastic_1se <- confusionMatrix(data = preds_1se, 
                                         reference = train_res, 
                                         positive  = "1")

# metrics
confusion_elastic_1se$overall["Accuracy"]
confusion_elastic_1se$byClass[c("F1", "Precision", "Sensitivity", "Specificity", "Recall")]
roc_el <- roc(response = train_res, predictor = probs_1se)
auroc_el <- auc(roc_el)

# ---- random forest ----

# conversions to maintain integrity of original train data
train2 <- train
train2$highhost <- factor(train$highhost, levels = c(0, 1), labels = c("no", "yes"))
train2$host_is_superhost <- as.factor(train$host_is_superhost)
train2$room_type <- as.factor(train$room_type)
train2$neighbourhood_ward <- as.factor(train2$neighbourhood_ward)
# forest_grid <- expand.grid(mtry = 2:(ncol(data) - 1),
#                            splitrule = "gini",
#                            min.node.size = c(1, 5, 20))
# 
# forest_ctrl <- trainControl(method="cv", number = 10, 
#                             verboseIter = F, classProbs = T,
#                             summaryFunction = twoClassSummary,
#                             savePredictions = "final")
#         
# set.seed(7154)
# mod_randomforest <- train(highhost ~ .,
#                             data       = train2,
#                             method     = "ranger",
#                             num.trees  = 250,
#                             importance = "impurity",
#                             trControl  = forest_ctrl,
#                             tuneGrid   = forest_grid)
load("C:/Users/Topollo/Desktop/analytics/assignment one/mod_randomforest.RData")
load("C:/Users/Topollo/Desktop/analytics/assignment one/rf_accuracy.RData")

# plot of the accuracy of the random forest
plot(rf_accuracy)

# all predictions on training data (for all hyperparameters)
preds_rf <- mod_randomforest$pred


# filter to best hyperparameters
preds_rf <- preds_rf %>% 
  filter(mtry          == mod_randomforest$bestTune$mtry,
         splitrule     == mod_randomforest$bestTune$splitrule,
         min.node.size == mod_randomforest$bestTune$min.node.size)

confusion_rf <- confusionMatrix(
  data      = preds_rf$pred,
  reference = preds_rf$obs,
  positive  = "yes")

# metrics
confusion_rf$overall["Accuracy"]
confusion_rf$byClass[c("F1", "Precision", "Specificity", "Recall")]  
roc_obj_rf <- roc(response  = preds_rf$obs, predictor = preds_rf$yes) 
auc(roc_obj_rf)








# ---- boosted tree model ----

# gradient boosted model:
# gbm_grid <- expand.grid(n.trees = c(1000, 10000, 20000),
#                         interaction.depth = c(1,2,5),
#                         shrinkage = c(0.1, 0.01),
#                         n.minobsinnode = 10)
# 
# gbm_ctrl <- trainControl(method = "cv", number = 10, verboseIter = F)
# 
# set.seed(7154)
# mod_gbm <- train(highhost ~ .,
#                  data = train2,
#                  method = "gbm",
#                  distribution = "bernoulli",
#                  trControl = gbm_ctrl,
#                  verbose = F,
#                  tuneGrid = gbm_grid)

load('C:/Users/Topollo/Desktop/analytics/assignment one/mod_gbm.Rdata')
load('C:/Users/Topollo/Desktop/analytics/assignment one/gbm_accuracy.Rdata')

# plot of the gbm accuracy
plot(gbm_accuracy)

# predictions of the gbm model
preds_gbm <- mod_gbm$pred

# filter to best hyperparameters
preds_gbm <- preds_gbm %>%
  filter(n.trees           == mod_gbm$bestTune$n.trees,
         interaction.depth == mod_gbm$bestTune$interaction.depth,
         shrinkage         == mod_gbm$bestTune$shrinkage,
         n.minobsinnode    == mod_gbm$bestTune$n.minobsinnode)

# confusion matrix
confusion_gbm <- confusionMatrix(data = preds_gbm$pred,
                                 reference = preds_gbm$obs,
                                 positive  = "yes")

# metrics
confusion_gbm$overall["Accuracy"]
confusion_gbm$byClass[c("F1", "Precision", "Specificity", "Recall")]
roc_obj_gbm <- roc(response  = preds_gbm$obs,
                   predictor = preds_gbm$yes)
auc(roc_obj_gbm)

# ---- k-nearest neighbours ----

# TO DO:
# create new train3 and test3
# from elasti-net, select necessary features (that are not . and 0.000) -> some neighbourhood wards have .
# create new matrix of my data with new columns where each neigh_ward is represented (make it as.factor)
# from notes: scale and rerun!

# selecting features from elastic-net

# coefficients from best elastic (1se)
best_coef_1se <- as.matrix(coef(best_model_1se, s = "lambda.1se"))
coef_table <- data.frame(
  feature = rownames(best_coef_1se),
  coefficient = round(as.numeric(best_coef_1se), 3)
)
print(coef_table)


# non-zero coefficients (* REMOVE INTERCEPT!)
non_zero_cols <- rownames(best_coef_1se)[which(best_coef_1se != 0 & 
                                                 rownames(best_coef_1se) != "(Intercept)")] %>%
  # Simplify factor-level feature names to base column names
  gsub("neighbourhood_ward.*", "neighbourhood_ward", .) %>%
  gsub("host_is_superhost.*", "host_is_superhost", .) %>%
  gsub("room_type.*", "room_type", .) %>%
  unique()
# errors bc you have to make cols match those in train set

# identify wards with zero coefficients
zero_wards <- coef_table %>%
  filter(coefficient == 0,
         str_detect(feature, "neighbourhood_ward")) %>%
  mutate(ward = str_remove(feature, "neighbourhood_ward")) %>%
  pull(ward)

# CREATING TRAIN 3 (make necessary changes for it to work)
# remove 0 and "." variables
train3 <- train2 %>% 
  select(all_of(non_zero_cols), highhost) %>%
  filter(!neighbourhood_ward %in% zero_wards)

train3$neighbourhood_ward <- droplevels(train3$neighbourhood_ward)
train3$host_is_superhost <- droplevels(train3$host_is_superhost)
train3$room_type <- droplevels(train3$room_type)

# dummy variables for categoricals (each considered its own var in elastic!)
dummies <- dummyVars(highhost ~ ., data = train3)

# scale
preproc <- preProcess(predict(dummies, newdata = train3), method = c("center", "scale"))

# apply changes to train, make response a factor
train3_mat <- predict(dummies, newdata = train3)
train3_scaled <- as.data.frame(predict(preproc, train3_mat))
train3_scaled$highhost <- as.factor(train3$highhost)


# fitting knn model
# knn_ctrl <- trainControl(method = "cv", 
#                          number = 10,
#                          classProbs = T,
#                          summaryFunction = twoClassSummary,
#                          savePredictions = "final")
# 
# knn_grid <- expand.grid(k = 1:20)
# 
# set.seed(7154)
# mod_knn <- train(highhost ~ .,
#                  data      = train3_scaled,
#                  method    = "knn",
#                  trControl = knn_ctrl,
#                  tuneGrid  = knn_grid,
#                  metric = "ROC")
load("C:/Users/Topollo/Desktop/analytics/assignment one/knn_accuracy.RData")
load("C:/Users/Topollo/Desktop/analytics/assignment one/mod_knn.Rdata")

# plot
plot(knn_accuracy)

# predictions
preds_knn <- mod_knn$pred %>%
  filter(k == mod_knn$bestTune$k)

# confusion matrix 
confusion_knn <- confusionMatrix(data = preds_knn$pred,
                                 reference = preds_knn$obs,
                                 positive  = "yes")
# metrics
confusion_knn$overall["Accuracy"]
confusion_knn$byClass[c("F1", "Precision", "Sensitivity", "Specificity")]
roc_obj_knn <- roc(response  = preds_knn$obs, predictor = preds_knn$yes)
auc(roc_obj_knn)


# ---- plot of performance across models ----
# Data
metrics_df <- data.frame(
  Model = c("Elastic-net", "Random Forest", "Gradient Boosted", "K-nearest neighbours"),
  Accuracy    = c(0.6975, 0.8664, 0.8566, 0.7689),
  F1_Score    = c(0.0627, 0.7705, 0.7560, 0.6527),
  Precision   = c(0.5549, 0.8077, 0.7845, 0.6394),
  Recall      = c(0.0333, 0.7365, 0.7295, 0.6665),
  Specificity = c(0.9883, 0.9233, 0.9123, 0.8184),
  AUROC       = c(0.7610, 0.9264, 0.9178, 0.8263)
)

# Reshape to long format
metrics_long <- metrics_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Score")

# Plot
ggplot(metrics_long, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75),
           width = 0.65, alpha = 0.9) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_fill_manual(values = c(
    "Elastic-net"          = "#5B6FE8",
    "Random Forest"        = "#1D9E75",
    "Gradient Boosted"     = "#D85A30",
    "K-nearest neighbours" = "#BA7517"
  )) +
  labs(
    title = NULL,
    x = NULL,
    y = "Score",
    fill = "Model"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title   = element_text(size = 14, margin = ggplot2::margin(b = 12)),
    axis.text.x  = element_text(angle = 15, hjust = 1),
    legend.position = "top",
    legend.title = element_text(size = 11),
    panel.grid.major.x = element_blank()
  )

# ---- QUESTION 3 ----
# get predicted probabilities from random forest on the validation set
# select by position — second column is always the probability of the positive class (1)
# align factor levels in test2 to match train2 exactly


# use "no" column and flip the true labels
rf_probs <- preds_rf$yes
rf_true  <- (preds_rf$obs == "yes") * 1


# tau values
taus <- seq(0.01, 0.99, by = 0.01)

# F1 score, for each tau
f1_scores <- sapply(taus, function(tau){
  preds <- ifelse(rf_probs >= tau, 1, 0)
  tp <- sum(preds == 1 & rf_true == 1)
  fp <- sum(preds == 1 & rf_true == 0)
  fn <- sum(preds == 0 & rf_true == 1)
  precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
})

# best f1_score and tau
best_f1_cv <- max(f1_scores)
best_tau   <- taus[which.max(f1_scores)]

# plot (F1 vs tau)
plot(taus, f1_scores,
     type = "l", lwd = 2,
     xlab = expression(tau),
     ylab = "F1 Score (Validation)",
     main = "F1 Score Across Decision Thresholds")
abline(v = best_tau, lty = 2, col = "firebrick", lwd = 2)
points(best_tau, best_f1_cv, pch = 16, col = "firebrick")
text(best_tau, best_f1_cv,
     labels = bquote(tau == .(best_tau) ~ ", F1 =" ~ .(round(best_f1_cv, 4))),
     pos = 4, cex = 0.85)

# TPR, FPR (at optimal tau)
preds_best <- ifelse(rf_probs >= best_tau, 1, 0)
tpr_best   <- sum(preds_best == 1 & rf_true == 1) / sum(rf_true == 1)
fpr_best   <- sum(preds_best == 1 & rf_true == 0) / sum(rf_true == 0)

# plotting (ROC)
pred_rocr <- prediction(rf_probs, rf_true)
perf_rocr <- performance(pred_rocr, "tpr", "fpr")
plot(perf_rocr, col = "red", lwd = 2,
     main = "Validation Set ROC Curve – Random Forest")
lines(c(0,1), c(0,1), col = "gray", lty = 4)
points(fpr_best, tpr_best, pch = 16, col = "firebrick", cex = 1.5)

text(fpr_best, tpr_best,
     labels = bquote("(" ~ tau ~ "=" ~ .(best_tau) ~  ", F1 = 0.7794)"),
     pos = 4, cex = 0.85)

# confusion matrix
final_preds <- as.factor(ifelse(rf_probs >= best_tau, 1, 0))
rf_obs_fac  <- as.factor(rf_true)

confusion_rf_tau <- confusionMatrix(final_preds, rf_obs_fac, positive = "1")

# metrics
confusion_rf_tau$overall["Accuracy"]
confusion_rf_tau$byClass[c("F1", "Precision", "Sensitivity", "Specificity", "Recall")]

# PREDICTIONS
# training, on tau = 0.41
rf_preds_train <- factor(ifelse(preds_rf$yes >= 0.41, "yes", "no"),
                         levels = c("no", "yes"))

# confusion matrix (train)
confusion_rf_train <- confusionMatrix(data = rf_preds_train,
                                      reference = preds_rf$obs,
                                      positive  = "yes")

# testing on tau = 0.41
rf_probs_test <- predict(mod_randomforest,
                         newdata = test,
                         type    = "prob")[, "yes"]

rf_preds_test <- factor(ifelse(rf_probs_test >= 0.41, "1", "0"),
                        levels = c("0", "1"))

# confusion matrix (test)
confusion_rf_test <- confusionMatrix(data = rf_preds_test,
                                     reference = test$highhost,
                                     positive  = "1")

# auroc
auc_train <- auc(roc(response  = preds_rf$obs, predictor = preds_rf$yes))
auc_test  <- auc(roc(response  = test$highhost, predictor = rf_probs_test))

# comparison table
comparison <- data.frame(
  metric   = c("Accuracy", "F1", "Precision", "Recall", "Specificity", "AUROC"),
  training = as.numeric(c(confusion_rf_train$overall["Accuracy"],
                          confusion_rf_train$byClass["F1"],
                          confusion_rf_train$byClass["Precision"],
                          confusion_rf_train$byClass["Sensitivity"],
                          confusion_rf_train$byClass["Specificity"],
                          auc_train)),
  test     = as.numeric(c(confusion_rf_test$overall["Accuracy"],
                          confusion_rf_test$byClass["F1"],
                          confusion_rf_test$byClass["Precision"],
                          confusion_rf_test$byClass["Sensitivity"],
                          confusion_rf_test$byClass["Specificity"],
                          auc_test)))

colnames(comparison)[2:3] <- c("Training (tau = 0.41)", "Test (tau = 0.41)")
kable(comparison, caption = "Random Forest: Training vs Test at Optimised Threshold", digits = 4)


# ---- QUESTION 4 ----
## impurity plots
rf_imp_p <- vip(mod_randomforest) + 
  labs(title = "Impurity Variable Importance",
       x = "Decrease in Splitting Criterion",
       y = NULL) + theme_minimal()

## permutation plot
rf_perm <- ranger(highhost ~ ., data = train2,
                  num.trees = 250,
                  mtry = 5,
                  min.node.size = 5,
                  splitrule = "gini",
                  importance = "permutation",
                  seed = 7154)

rf_per_p <- vip(rf_perm) + 
  labs(title = "Permutation Variable Importance",
       x = "Increase in Prediction Error",
       y = NULL) + theme_minimal()

# partial dependency plots
p1 <- partial(mod_randomforest, pred.var = "price", 
              prob = TRUE, plot = TRUE, rug = TRUE, train = train2)

p3 <- partial(mod_randomforest, pred.var = "host_acceptance_rate", 
              prob = TRUE, plot = TRUE, rug = TRUE, train = train2)

grid.arrange(p1, p3, ncol = 2)

# the false negatives and false positives
results <- test %>% 
  mutate(
    predicted = rf_preds_test,
    true      = highhost
  )

# false positives: predicted highhost = 1, when actually highhost = 0 actually
false_positives <- results %>%
  filter(predicted == 1 & true == "0")

# false negatives: predicted highhost = 0, when actually highhost = 1 actually
false_negatives <- results %>%
  filter(predicted == 0 & true == "1")

set.seed(7154)
fp_sample <- false_positives %>% slice_sample(n = 5)
fn_sample <- false_negatives %>% slice_sample(n = 5)




