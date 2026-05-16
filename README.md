# STA4026Z Assignment 1 – Applied Supervised Learning

## Overview

This repository contains the code, data, models, and report for Assignment 1 of STA4026Z (Statistics Honours Analytics), focusing on applied supervised learning using Cape Town Airbnb listing data.

The assignment involves building and comparing multiple classification models to predict a binary outcome derived from Airbnb listings, with emphasis on model selection, validation, threshold optimisation, and interpretability.

---

## Problem Context

Short-term rental platforms such as Airbnb have notable impacts on housing affordability and rental availability in Cape Town. This project uses supervised learning methods to model Airbnb listing data and classify listings according to a selected binary outcome related to housing market dynamics.

---

## Objectives

The main objectives of this assignment are to:

- Construct a binary classification target variable from Airbnb features  
- Train and compare multiple supervised learning models  
- Perform hyperparameter tuning using cross-validation  
- Evaluate models using multiple classification metrics  
- Optimise decision thresholds for F1 score performance  
- Interpret model outputs and analyse classification errors  

---

## Methods

### Data Preparation
- 80/20 train-test split  
- Feature preprocessing and selection  
- Construction of binary target variable (training set only)

---

### Models Implemented

The following models were developed and compared:

- Elastic Net Logistic Regression  
- Random Forest  
- Gradient Boosting model (e.g. GBM / XGBoost / LightGBM)  
- K-Nearest Neighbours (KNN)

---

### Model Evaluation
Models were evaluated using:

- 10-fold cross-validation  
- Metrics:
  - Accuracy  
  - F1 Score  
  - Precision  
  - Recall  
  - Specificity  
  - AUROC  
- Default decision threshold τ = 0.5  

---

### Threshold Optimisation
- Optimisation of decision threshold τ for maximum F1 score  
- ROC curve analysis  
- Final evaluation on test set using optimal τ  

---

### Interpretation and Error Analysis
- Variable importance plots  
- Partial dependence plots  
- Coefficient interpretation (Elastic Net alternative)  
- Misclassification analysis (false positives and false negatives)  

---

## Repository Structure

```text
.
├── code/
│   ├── Assignment code.R
│   └── analyticsone.R
│
├── data/
│   ├── listings.csv
│   ├── variable_descriptions.csv
│   └── my_data.RData
│
├── images/
│   ├── UCT-logo.png
│   ├── logocirclesss.gif
│   └── stats_logo.png
│
├── models/
│   ├── cv_results.RData
│   ├── forest.Rdata
│   ├── mod_gbm.Rdata
│   ├── mod_knn.Rdata
│   └── mod_randomforest.Rdata
│
├── results/
│   ├── gbm_accuracy.RData
│   ├── knn_accuracy.RData
│   ├── p1.RData
│   ├── p3.RData
│   ├── per_data.RData
│   ├── rf_accuracy.RData
│   ├── rf_probs.RData
│   └── rf_true.RData
│
├── report/
│   ├── MLNNAN007_MKHTOP003_STA4026Z_a1.pdf
│   ├── STA4026Z 2026 Assignment 1.pdf
│   ├── analyticsone.Rnw
│   ├── analyticsone.tex
│   └── analyticsone-concordance.tex
│
└── README.md
