# Project Overview

This repository contains the complete implementation code for a comparative survival analysis study using breast cancer data. The project focuses on model development, tuning, validation, and interpretability, with an emphasis on rigorous nested cross-validation and modern survival machine learning methods.

# Objectives

- Implement a nested cross-validation framework for unbiased survival model evaluation
- Compare classical and machine learning–based survival models
- Tune hyperparameters using inner resampling loops
- Evaluate models using survival-specific metrics
- Apply model explainability techniques to the best-performing model

# Models Implemented

The following survival models are implemented using the tidymodels ecosystem:

- Penalized Cox Proportional Hazards model 

- Random Survival Forests 

- Gradient Boosted Survival Trees

- Survival Decision Trees

# Model Evaluation

All models are evaluated using nested 5×5 cross-validation with stratification on the event indicator. Performance is assessed using:

- Concordance Index (C-index)

- Time-dependent ROC AUC

- Integrated Brier Score

Evaluation is conducted across multiple event-time horizons.

# Model Interpretability

For the best-performing model (Random Survival Forest):

- Survival-specific SHAP values are computed

- Global and local feature effects are analyzed using survex

- Risk and survival function predictions are explicitly defined

This ensures transparent and explainable survival modeling, suitable for clinical and applied research settings.
