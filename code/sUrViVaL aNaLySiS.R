library(tidyverse)
library(readxl)
library(survivalsvm)
library(survivalmodels)
library(reticulate)
library(tidymodels)
library(censored)
library(aorsf)
library(purrr)
library(survAUC)
library(future)
library(purrr)
library(Hmisc)
library(parallel)
library(furrr)
library(survivalROC)
library(progressr)
library(survex)
library(stats)

#Importing the data, converting characters into factors, and creating the Survival object
breast_cancer <- read_csv("Breast Cancer Kiki1.csv") |> 
  mutate(across(where(is.character) & !matches("Patient"),
                factor)) |> 
  select(-Patient_ID, -No) |> 
  mutate(Breast_cancer_Surv = Surv(Event_time, Event),
         .keep = "unused")
  
eval_times <- sort(unique(breast_cancer$Breast_cancer_Surv))[,1]

#Splitting the data into nested folds
set.seed(313)
nested_fold <- nested_cv(breast_cancer,
                         outside = vfold_cv(v = 5, 
                                            strata = breast_cancer$Breast_cancer_Surv[,2]),
                         inside = vfold_cv(v = 5, 
                                           strata = breast_cancer$Breast_cancer_Surv[,2]))
outer_resample <- nested_fold$splits
inner_resample <- nested_fold$inner_resamples

#Data preprocessing
bc_recipe <- recipe(Breast_cancer_Surv ~ ., data = breast_cancer) |>
  step_normalize(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_nzv(all_predictors()) 

#Parallel Processing
plan(multisession)

#######################################################
################ PROPORTIONAL HAZARD ##################
#######################################################

ph_spec <- proportional_hazards(penalty = tune(), 
                                mixture = tune()) |> 
  set_mode("censored regression") |> 
  set_engine("glmnet")

ph_wf <- workflow() |> 
  add_model(ph_spec) |> 
  add_recipe(bc_recipe) 

ph_set <- extract_parameter_set_dials(ph_spec)

ph_res <- map(1: length(outer_resample), \(i){
  
  tuned <- tune_grid(
    ph_wf,
    resamples = inner_resample[[i]],
    param_info = ph_set,
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times,
    control = control_grid(verbose = TRUE, allow_par = F)
  )
  
  best_params <- select_best(tuned, metric = "concordance_survival")
  
  final_wf <- finalize_workflow(ph_wf, best_params)
  
  final_fit <- last_fit(
    final_wf,
    split = outer_resample[[i]],
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times
  )
  
  list(
    final_fit = final_fit,
    best_params = best_params
  )
})

ph_hyper <- map_dfr(ph_res, ~ .x$best_params, 
                     .id = "fold") |>
  summarise(across(c(penalty, mixture), 
                   list(mean = mean,
                        sd = sd,
                        median = median,
                        mad = mad)))

ph_final_metrics <- map_dfr(ph_res, 
                             ~ collect_metrics(.x$final_fit), 
                             .id = "fold") |>
  group_by(.metric) |>
  summarise(
    mean = mean(.estimate, na.rm = TRUE),
    sd = sd(.estimate, na.rm = TRUE),
    median = median(.estimate, na.rm = TRUE),
    se = sd(.estimate, na.rm = TRUE) / sqrt(n()),
    ci_lower = mean - 1.96 * se,
    ci_upper = mean + 1.96 * se
  )

final_ph_mod <- finalize_workflow(ph_wf, ph_hyper |>
                                     transmute(
                                       penalty  = penalty_median,
                                       mixture = mixture_median
                                     )) |> 
  fit(data = breast_cancer) |> 
  extract_fit_engine()


########################################################
################Random survival forest##################
########################################################
rsf_spec <- rand_forest(mtry = tune(), trees = 1e3,
                        min_n = tune()) |> 
  set_engine("aorsf") |> 
  set_mode("censored regression")

rsf_wf <- workflow() |>
  add_model(rsf_spec) |>
  add_recipe(bc_recipe)

rsf_set <- extract_parameter_set_dials(rsf_wf) |> 
  finalize(breast_cancer |> select(-Breast_cancer_Surv))

rsf_res <- map(1: length(outer_resample), \(i){
  
  tuned <- tune_grid(
    rsf_wf,
    resamples = inner_resample[[i]],
    param_info = rsf_set,
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times,
    control = control_grid(verbose = TRUE, allow_par = F)
  )
  
  best_params <- select_best(tuned, metric = "concordance_survival")
  
  final_wf <- finalize_workflow(rsf_wf, best_params)
  
  final_fit <- last_fit(
    final_wf,
    split = outer_resample[[i]],
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times
  )
  
  list(
    final_fit = final_fit,
    best_params = best_params
  )
})

rsf_hyper <- map_dfr(rsf_res, ~ .x$best_params, 
                     .id = "fold") |>
  summarise(across(c(mtry, min_n), 
                   list(mean = mean,
                        sd = sd,
                        median = median,
                        mad = mad)))

rsf_final_metrics <- map_dfr(rsf_res, 
                             ~ collect_metrics(.x$final_fit), 
                             .id = "fold") |>
  group_by(.metric) |>
  summarise(
    mean = mean(.estimate, na.rm = TRUE),
    sd = sd(.estimate, na.rm = TRUE),
    median = median(.estimate, na.rm = TRUE),
    se = sd(.estimate, na.rm = TRUE) / sqrt(n()),
    ci_lower = mean - 1.96 * se,
    ci_upper = mean + 1.96 * se
  )

final_rsf_mod <- finalize_workflow(rsf_wf, rsf_hyper |>
                                     transmute(
                                       mtry  = mtry_median,
                                       min_n = min_n_median
                                     )) |> 
  fit(data = breast_cancer) |> 
  extract_fit_engine()

########################################################
######### Gradient Boosted Survival Trees ##############
########################################################
xgb_spec <- boost_tree(mtry = tune(), trees = 1e3, 
                       min_n = tune(), tree_depth = tune(),
                       loss_reduction = tune()) |> 
  set_mode("censored regression") |> 
  set_engine("mboost")

xgb_wf <- workflow() |>
  add_model(xgb_spec) |>
  add_recipe(bc_recipe)

xgb_set <- extract_parameter_set_dials(xgb_wf) |> 
  finalize(breast_cancer |> select(-Breast_cancer_Surv))

xgb_res <- map(1: length(outer_resample), \(i){
  
  tuned <- tune_grid(
    xgb_wf,
    resamples = inner_resample[[i]],
    param_info = xgb_set,
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times,
    control = control_grid(verbose = TRUE, allow_par = F)
  )
  
  best_params <- select_best(tuned, metric = "concordance_survival")
  
  final_wf <- finalize_workflow(xgb_wf, best_params)
  
  final_fit <- last_fit(
    final_wf,
    split = outer_resample[[i]],
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times
  )
  
  list(
    final_fit = final_fit,
    best_params = best_params
  )
})

xgb_hyper <- map_dfr(xgb_res, ~ .x$best_params, 
                     .id = "fold") |>
  summarise(across(c(mtry, min_n, tree_depth, loss_reduction), 
                   list(mean = mean,
                        sd = sd,
                        median = median,
                        mad = mad)))

xgb_final_metrics <- map_dfr(xgb_res, 
                             ~ collect_metrics(.x$final_fit), 
                             .id = "fold") |>
  group_by(.metric) |>
  summarise(
    mean = mean(.estimate, na.rm = TRUE),
    sd = sd(.estimate, na.rm = TRUE),
    median = median(.estimate, na.rm = TRUE),
    se = sd(.estimate, na.rm = TRUE) / sqrt(n()),
    ci_lower = mean - 1.96 * se,
    ci_upper = mean + 1.96 * se
  )

final_xgb_mod <- finalize_workflow(xgb_wf, xgb_hyper |>
                                     transmute(
                                       mtry  = mtry_median,
                                       min_n = min_n_median,
                                       tree_depth = tree_depth_median,
                                       loss_reduction = loss_reduction_median
                                     )) |> 
  fit(data = breast_cancer_clean) |> 
  extract_fit_engine()

#######################################################
##################### DECISION TREE ###################
#######################################################
tree_spec <- decision_tree(cost_complexity = tune(),
                           tree_depth = tune(),
                           min_n = tune()) |> 
  set_mode("censored regression")

tree_wf <- workflow() |>
  add_model(tree_spec) |>
  add_recipe(bc_recipe)

tree_set <- extract_parameter_set_dials(tree_wf) 

tree_res <- map(1: length(outer_resample), \(i){
  
  tuned <- tune_grid(
    tree_wf,
    resamples = inner_resample[[i]],
    param_info = tree_set,
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times,
    control = control_grid(verbose = TRUE, allow_par = F)
  )
  
  best_params <- select_best(tuned, metric = "concordance_survival")
  
  final_wf <- finalize_workflow(tree_wf, best_params)
  
  final_fit <- last_fit(
    final_wf,
    split = outer_resample[[i]],
    metrics = metric_set(concordance_survival,
                         roc_auc_survival,
                         brier_survival_integrated),
    eval_time = eval_times
  )
  
  list(
    final_fit = final_fit,
    best_params = best_params
  )
})

tree_hyper <- map_dfr(tree_res, ~ .x$best_params, 
                     .id = "fold") |>
  summarise(across(c(tree_depth, min_n, cost_complexity), 
                   list(mean = mean,
                        sd = sd,
                        median = median,
                        mad = mad)))

tree_final_metrics <- map_dfr(tree_res, 
                             ~ collect_metrics(.x$final_fit), 
                             .id = "fold") |>
  group_by(.metric) |>
  summarise(
    mean = mean(.estimate, na.rm = TRUE),
    sd = sd(.estimate, na.rm = TRUE),
    median = median(.estimate, na.rm = TRUE),
    se = sd(.estimate, na.rm = TRUE) / sqrt(n()),
    ci_lower = mean - 1.96 * se,
    ci_upper = mean + 1.96 * se
  )

final_tree_mod <- finalize_workflow(tree_wf, xgb_hyper |>
                                     transmute(
                                       mtry  = mtry_median,
                                       min_n = min_n_median,
                                       tree_depth = tree_depth_median,
                                       loss_reduction = loss_reduction_median
                                     )) |> 
  fit(data = breast_cancer) |> 
  extract_fit_engine()

#RSF MODEL IS THE BEST
rsf_exp <- explain_survival(
  final_rsf_mod,
  data = bc_recipe |> prep() |> bake(new_data = breast_cancer) |> 
    select(-Breast_cancer_Surv),
  y = breast_cancer$Breast_cancer_Surv,
  verbose = FALSE,
  times = eval_times,
  predict_function = \(model, newdata) {
    as.numeric(predict(model, newdata, pred_type = "risk"))
  }, 
  predict_survival_function = \(model, newdata, times) {
    pred <- predict(model, 
                    newdata, 
                    pred_horizon = times,  
                    pred_type = "surv")
    
    if (is.vector(pred)) pred <- matrix(pred, nrow = nrow(newdata))
    pred
  }
)

obs_multiple <- bc_recipe |> 
  prep() |> 
  bake(new_data = breast_cancer |> 
         select(-Breast_cancer_Surv))
  

shap_multiple <- predict_parts(rsf_exp, 
                               new_observation = obs_multiple,
                               type = "survshap")
plot(shap_multiple)
